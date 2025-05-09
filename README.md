# Multi-view PBR Material Diffusion Model (MV-PBRMat-Diffusion)
PyTorch implementation of Multi-view PBR Material Diffusion Model, a small step towards the reproduction of [CLAY](https://sites.google.com/view/clay-3dlm)-like material synthesis architecture.

## 🎸 Install for inference
- You can install the required dependencies via *requirements-inference.txt*.
```bash
pip install -r requirements-inference.txt
```

- (Optional) You can also install PyTorch at first via [PyTorch Official](https://pytorch.org/get-started/previous-versions/), then install other dependencies via *requirements-inference.txt*.
```bash
# CUDA 11.6
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia

# remember to remove packages related to torch
pip install -r requirements-inference.txt
```

## 🎺 Download checkpoints
### Pre-trained weights
- MVDream & MVControlNet

For MVDream base model and Multi-view ControlNet, we use a third-party [diffusers implementation](https://huggingface.co/SnowflakeWang/MV-PBRMat-Diffusion) inherited from the [HuggingFace repo](https://huggingface.co/lzq49/mvdream-sd21-diffusers) of [Controllable Text-to-3D Generation via Surface-Aligned Gaussian Splatting](https://lizhiqi49.github.io/MVControl/), instead of the official implementation.

Click the links to download: [MVDream](https://huggingface.co/SnowflakeWang/MV-PBRMat-Diffusion), [MVControlNet](https://huggingface.co/lzq49/mvcontrol-4v-normal)

- CLIP

Click the link to download: [CLIP-ViT-H-14-laion2B-s32B-b79K](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K)

### (Optional) Fine-tuned checkpoints
- All modules are unified in a single checkpoint.

Click the link to download: [pytorch_model.bin]().

### Final saved fine-tuned checkpoints
- IP-Adapter & Image Projection Model

Click the links to download: [ip_adapter.pt](https://huggingface.co/SnowflakeWang/MV-PBRMat-Diffusion/resolve/main/ip_adapter.pt?download=true), [image_proj_model.pt](https://huggingface.co/SnowflakeWang/MV-PBRMat-Diffusion/resolve/main/image_proj_model.pt?download=true)

- UNet LoRA & Multi-branch

Click the link to download: [unet.pt](https://huggingface.co/SnowflakeWang/MV-PBRMat-Diffusion/resolve/main/unet.pt?download=true), LoRA and Multi-branch have been unified in it.

## 🎹 Run inference
### Prepare input
- You can organize your input as the following structure. We put one folder in *data/input_mv_normal* as an example.
```bash
|-- example1
    |-- front.png # rendered normal from the front view of the mesh
    |-- right.png # right view normal
    |-- back.png # back view normal
    |-- left.png # left view normal
    |-- top.png # top view normal
    |-- bottom.png # bottom view normal
|-- example2
...
```

### Run code
- You can regard *run.sh* as an example.
```bash
# in run.sh

python inference.py --ctrlnet_seed=42 --mvdiff_seed=42 \ # seeds for normal-conditioned ControlNet and Multi-view PBR Diffusion
    --controlnet_cond_mode="image" \ # normal-conditioned ControlNet can take a text/image prompt as input, "text" for text prompt, "image" for image prompt
    --prompt="/path/to/image_prompt.png" \ # if controlnet_cond_mode=="text", input text prompt. if controlnet_cond_mode=="image", input /path/to/image
    --controlnet_normal_path="/path/to/front_normal.png" \ # normal condition for ControlNet, recommend to use the front view normal of the untextured mesh
    --input_path="/path/to/multi_view_normals_folder" \ # the folder should contain 6-view normals: front.png, right.png, back.png, left.png, top.png, bottom.png
    --out_path="/path/to/inference_results" \ # output path
    --do_mv_super_res # whether do multi-view super-resolution or not
```
- Run *run.sh* for inference.
```bash
bash run.sh
```

## 🎻 Visualization
### Example 1
- Image prompt

<img src="assets/ironman_rgba.png" width="150" height="150" alt="">

- Multi-view PBR material generation results

<img src="assets/eg1_results.png" alt="">

### Example 2
- Text prompt

*"Red Queen, full body, CGI, 8K, HD"*

- Multi-view PBR material generation results

<img src="assets/eg2_results.png" alt="">

### Example 3
- Text prompt

*"Mandalorian, full body, CGI, 8K, HD"*

- Multi-view PBR material generation results

<img src="assets/eg3_results.png" alt="">

## 💎 Acknowledgments
- [CLAY: A Controllable Large-scale Generative Model for Creating High-quality 3D Assets](https://sites.google.com/view/clay-3dlm)
- [MVDream: Multi-view Diffusion for 3D Generation](https://mv-dream.github.io/)
- [Controllable Text-to-3D Generation via Surface-Aligned Gaussian Splatting](https://lizhiqi49.github.io/MVControl/)
- [IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models](https://ip-adapter.github.io/)
- [HyperHuman: Hyper-Realistic Human Generation with Latent Structural Diffusion](https://snap-research.github.io/HyperHuman/)
- [Objaverse: A Universe of Annotated 3D Objects](https://objaverse.allenai.org/)
- [G-buffer Objaverse: High-Quality Rendering Dataset of Objaverse](https://aigc3d.github.io/gobjaverse/)
- [Material Anything: Generating Materials for Any 3D Object via Diffusion](https://xhuangcv.github.io/MaterialAnything/)
- [Unique3D: High-Quality and Efficient 3D Mesh Generation from a Single Image](https://wukailu.github.io/Unique3D/)