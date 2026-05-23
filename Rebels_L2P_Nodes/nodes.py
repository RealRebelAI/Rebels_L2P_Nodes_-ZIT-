import os
import re
import glob
import numpy as np
import torch
from PIL import Image

import folder_paths

try:
    from diffsynth.pipelines.z_image_L2P import ZImagePipeline, ModelConfig
    _L2P_IMPORT_ERROR = None
except Exception as e:  # noqa
    ZImagePipeline = ModelConfig = None
    _L2P_IMPORT_ERROR = e

_NODE_DIR = os.path.dirname(os.path.abspath(__file__))
_BUNDLED_TOK = os.path.join(_NODE_DIR, "tokenizer")
_TOK_REPO = "Tongyi-MAI/Z-Image-Turbo"
_DTYPE_MAP = {"fp8_e4m3fn": getattr(torch, "float8_e4m3fn", None),
              "fp8_e5m2": getattr(torch, "float8_e5m2", None), "bf16": None}
_MAIN_FOLDERS = ("diffusion_models", "unet", "checkpoints")
_TE_FOLDERS = ("text_encoders", "clip")
_PIPE_CACHE = {}


def _pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _tok_present(p):
    return os.path.isdir(p) and (os.path.exists(os.path.join(p, "tokenizer_config.json"))
                                 or os.path.exists(os.path.join(p, "tokenizer.json")))


def _ensure_tokenizer():
    if _tok_present(_BUNDLED_TOK):
        return _BUNDLED_TOK
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=_TOK_REPO, allow_patterns=["tokenizer/*"], local_dir=_NODE_DIR)
    except Exception as e:  # noqa
        raise RuntimeError(
            f"Tokenizer missing at '{_BUNDLED_TOK}' and download failed ({e}). Run once:\n"
            f'  huggingface-cli download {_TOK_REPO} --include "tokenizer/*" --local-dir "{_NODE_DIR}"')
    if not _tok_present(_BUNDLED_TOK):
        raise RuntimeError(f"Tokenizer download did not populate '{_BUNDLED_TOK}'.")
    return _BUNDLED_TOK


def _list_from(folders, ph):
    names = []
    for f in folders:
        try:
            names += folder_paths.get_filename_list(f)
        except Exception:  # noqa
            pass
    return list(dict.fromkeys(names)) or [ph]


def _resolve_one(name, folders, what):
    if name.startswith("<<<"):
        raise FileNotFoundError(f"No {what} found. {name.strip('< >')}")
    for f in folders:
        try:
            p = folder_paths.get_full_path(f, name)
            if p:
                return p
        except Exception:  # noqa
            pass
    raise FileNotFoundError(f"Could not resolve {what}: {name}")


def _resolve_te_paths(name):
    p = _resolve_one(name, _TE_FOLDERS, "text encoder")
    base = os.path.basename(p)
    m = re.match(r"^(.*)-\d+-of-\d+\.safetensors$", base)
    if m:
        sh = sorted(glob.glob(os.path.join(os.path.dirname(p), f"{m.group(1)}-*-of-*.safetensors")))
        return sh or [p]
    return [p]


def _dtype_cfg(dtype, offload, device):
    """
    Always stream from CPU. Storage stays fp8 only if explicitly chosen;
    otherwise bf16 — no per-step fp8<->bf16 cast, which is what kills speed on Ampere.
    """
    storage = dtype if dtype is not None else torch.bfloat16
    return dict(
        offload_device=("cpu" if offload else device), 
        offload_dtype=storage,
        onload_device=device,      
        onload_dtype=torch.bfloat16,
        preparing_device=device,   
        preparing_dtype=torch.bfloat16,
        computation_device=device, 
        computation_dtype=torch.bfloat16,
    )


class RebelL2PPipelineLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "main_model": (_list_from(_MAIN_FOLDERS, "<<< put DiT in models/diffusion_models >>>"),),
            "text_encoder": (_list_from(_TE_FOLDERS, "<<< put encoder in models/text_encoders >>>"),),
            "main_model_dtype": (["fp8_e4m3fn", "fp8_e5m2", "bf16"], {"default": "fp8_e4m3fn"}),
            "text_encoder_dtype": (["fp8_e4m3fn", "fp8_e5m2", "bf16"], {"default": "fp8_e4m3fn"}),
            "offload_to_cpu": ("BOOLEAN", {"default": True}),
            "vram_limit_gb": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 256.0, "step": 0.5}),
        }}

    RETURN_TYPES = ("L2P_PIPELINE",)
    RETURN_NAMES = ("l2p_pipeline",)
    FUNCTION = "load"
    CATEGORY = "Rebel Nodes/L2P"

    def load(self, main_model, text_encoder, main_model_dtype, text_encoder_dtype,
             offload_to_cpu, vram_limit_gb):
        if ZImagePipeline is None:
            raise RuntimeError(
                "Could not import diffsynth.pipelines.z_image_L2P. Install:\n"
                "  git clone https://github.com/TencentYoutuResearch/T2I-L2P\n"
                "  cd T2I-L2P && pip install -e .\n"
                f"Import error: {_L2P_IMPORT_ERROR}")

        main_path = _resolve_one(main_model, _MAIN_FOLDERS, "main DiT model")
        te_paths = _resolve_te_paths(text_encoder)
        tok_dir = _ensure_tokenizer()
        device = _pick_device()

        dit_dtype = _DTYPE_MAP.get(main_model_dtype)
        te_dtype = _DTYPE_MAP.get(text_encoder_dtype)
        for label, sel, dt in (("main_model_dtype", main_model_dtype, dit_dtype),
                               ("text_encoder_dtype", text_encoder_dtype, te_dtype)):
            if sel.startswith("fp8") and dt is None:
                raise RuntimeError(f"{sel} unavailable in this torch build; pick bf16 for {label}.")
        vram_limit = float(vram_limit_gb) if vram_limit_gb and vram_limit_gb > 0 else None

        key = (os.path.abspath(main_path), tuple(os.path.abspath(p) for p in te_paths),
               device, main_model_dtype, text_encoder_dtype, bool(offload_to_cpu), vram_limit)
        if key in _PIPE_CACHE:
            return (_PIPE_CACHE[key],)

        main_cfg = ModelConfig(path=[main_path], **_dtype_cfg(dit_dtype, offload_to_cpu, device))
        te_cfg = ModelConfig(path=te_paths, **_dtype_cfg(te_dtype, offload_to_cpu, device))

        pipe = ZImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16, device=device,
            model_configs=[main_cfg, te_cfg],
            tokenizer_config=ModelConfig(path=tok_dir),
            vram_limit=vram_limit,
        )
        _PIPE_CACHE[key] = pipe
        return (pipe,)


class RebelL2PGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "l2p_pipeline": ("L2P_PIPELINE",),
            "prompt": ("STRING", {"default": "", "multiline": True}),
            "negative_prompt": ("STRING", {"default": "", "multiline": True}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "steps": ("INT", {"default": 8, "min": 1, "max": 100}),
            "cfg_scale": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 20.0, "step": 0.1}),
            "width": ("INT", {"default": 512, "min": 256, "max": 4096, "step": 64}),
            "height": ("INT", {"default": 512, "min": 256, "max": 4096, "step": 64}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 16}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "Rebel Nodes/L2P"

    def _to_comfy_image(self, result):
        imgs = result if isinstance(result, (list, tuple)) else [result]
        out = []
        for im in imgs:
            if isinstance(im, Image.Image):
                arr = np.array(im.convert("RGB")).astype(np.float32) / 255.0
            elif isinstance(im, torch.Tensor):
                t = im.detach().float().cpu()
                if t.dim() == 4:
                    t = t[0]
                if t.dim() == 3 and t.shape[0] in (1, 3):
                    t = t.permute(1, 2, 0)
                arr = (t / 255.0 if t.max() > 1.5 else t).numpy()
            else:
                arr = np.array(im).astype(np.float32) / 255.0
            out.append(torch.from_numpy(np.ascontiguousarray(arr)))
        return torch.stack(out, dim=0)

    def generate(self, l2p_pipeline, prompt, negative_prompt, seed, steps,
                 cfg_scale, width, height, batch_size):
        device = _pick_device()
        rand_device = "cuda" if device == "cuda" else "cpu"
        out = []
        for i in range(int(batch_size)):
            kwargs = dict(prompt=prompt, seed=int(seed) + i, rand_device=rand_device,
                          num_inference_steps=int(steps), cfg_scale=float(cfg_scale),
                          height=int(height), width=int(width))
            if negative_prompt and negative_prompt.strip():
                try:
                    result = l2p_pipeline(negative_prompt=negative_prompt, **kwargs)
                except TypeError:
                    result = l2p_pipeline(**kwargs)
            else:
                result = l2p_pipeline(**kwargs)
            out.append(self._to_comfy_image(result))
        return (torch.cat(out, dim=0),)


NODE_CLASS_MAPPINGS = {"RebelL2PPipelineLoader": RebelL2PPipelineLoader,
                       "RebelL2PGenerate": RebelL2PGenerate}
NODE_DISPLAY_NAME_MAPPINGS = {"RebelL2PPipelineLoader": "L2P Pipeline Loader (Rebel)",
                              "RebelL2PGenerate": "L2P Generate (Rebel)"}