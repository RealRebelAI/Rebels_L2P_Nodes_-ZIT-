# Rebels L2P Nodes (Z-Image-Turbo) for ComfyUI
<img width="1496" height="898" alt="Screenshot (125)" src="https://github.com/user-attachments/assets/b770f6f0-8d71-45f3-9afc-b3632661e05e" />


A custom node suite for running Latent-to-Pixel (L2P) generation using Z-Image-Turbo directly inside ComfyUI. 

These nodes are heavily optimized for local execution on RTX 30-series (Ampere) hardware with 8GB VRAM, leveraging system RAM offloading to handle massive pixel-space calculations without crashing.

## ✨ Features
* **Ampere-Optimized fp8 Compute:** Bypasses the crippling CPU-casting slowdowns on 30-series cards. Models are stored in `fp8_e4m3fn` to save system RAM, but computation is forced to `bf16` the moment it hits the GPU.
* **Memory-Efficient SDPA:** Hardcoded Scaled Dot-Product Attention (SDPA) to prevent VRAM spiking during high-resolution pixel-space math.
* **Aggressive Garbage Collection:** Sweeps system RAM between batches and steps to maximize the 16GB offload space.

## 🛠️ Installation

### 1. Install the Nodes
Navigate to your ComfyUI `custom_nodes` folder and clone this repository:
```bash
cd ComfyUI/custom_nodes
git clone [https://github.com/RealRebelAI/Rebels_L2P_Nodes_-ZIT-.git](https://github.com/RealRebelAI/Rebels_L2P_Nodes_-ZIT-.git)
2. Install Dependencies
This node relies on the diffsynth L2P backend. If you are using the ComfyUI Windows Portable version, open your terminal in your root ComfyUI folder and run:

Bash
.\python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\Rebels_L2P_Nodes_-ZIT-\requirements.txt
(If you are on a standard Python environment, simply use pip install -r requirements.txt)

3. Model Placement
Because this uses a non-native text encoder setup, your models must be placed in the correct directories:

Main DiT Model: Place your merged .safetensors file (e.g., model-1k-merge.safetensors) into:
ComfyUI/models/diffusion_models/

https://huggingface.co/zhen-nan/L2P/blob/main/model-1k-merge.safetensors


Text Encoders (Qwen): Create a folder for the shards and place them here:
ComfyUI/models/text_encoders/Z-Image-Turbo/text_encoder/
(Note: You must use the official split shards, e.g., model-00001-of-00003.safetensors. Pruned single-file encoders missing the lm_head will crash the pipeline).

https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/tree/main/text_encoder


🚀 Step-by-Step Usage Guide
Load the Pipeline: Add the L2P Pipeline Loader (Rebel) node to your workspace.

Configure Data Types: * Set both main_model_dtype and text_encoder_dtype to fp8_e4m3fn.

Ensure offload_to_cpu is set to True.

Set vram_limit_gb to 4.0 or 5.0 to leave headroom for the generation canvas.

Connect the Generator: Add the L2P Generate (Rebel) node and connect the l2p_pipeline.

Dial in the Generation Settings: For the merged model format, you must use these specific settings to get clean text and coherent faces:

Steps: 30

CFG Scale: 2.0

Resolution & Upscaling (Crucial): Pixel-space generation calculates attention across every single pixel. Running native 1024x1024 on an 8GB card will cause severe PCIe bus bottlenecks (15+ minute render times) as data shuffles between the GPU and system RAM.

Recommended Workflow: Set your L2P Generate width/height to 512x512. Then, route the IMAGE output into a standard ComfyUI Latent Upscale workflow to achieve your 1024x1024 final image cleanly and quickly.

⚠️ Known Limitations

Hardware Requirements: Generating with these nodes requires a minimum of 16GB of system RAM for offloading, alongside a healthy Windows Pagefile (40GB+ recommended) to prevent memory-spill crashing.
