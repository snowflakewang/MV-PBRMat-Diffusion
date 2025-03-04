import argparse
import contextlib
import gc
import logging
import math
import os
import random
import shutil
from pathlib import Path

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from transformers import CLIPVisionModelWithProjection, CLIPVisionModel, CLIPImageProcessor

import diffusers
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    DDIMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available, convert_state_dict_to_diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from utils import cast_training_params

# import for MVDream
from extern.mvcontrol.pipeline_mvcontrol import load_mvcontrol_pipeline
from extern.mvcontrol.pipeline_mvcontrol_multi_branch import MVControlPipeline
from extern.mvcontrol.utils.camera import get_camera, get_top_bottom_camera
from extern.mvcontrol.mvcontrolnet import MultiViewControlNetModel
from extern.mvcontrol.attention import XFormersCrossViewAttnProcessor, CrossViewAttnProcessor, set_self_attn_processor
from extern.ip_adapter.ip_adapter import ImageProjModel, FullImageTokenProjModel
from extern.ip_adapter.attention_processors import IPAttnProcessor
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
import einops
import imageio.v2 as imageio
import copy
import re
import pdb

from utils import make_image_grid, split_image

from mvsr.app.all_models import model_zoo
from ctrlnet_ip.ip_adapter import IPAdapter

def print_green(text):
    print(f"\033[92m{text}\033[0m")

class MVPBRMatDiffusionWrapper(torch.nn.Module):
    def __init__(
        self,
        unet,
        controlnet,
        image_proj_model,
        ip_adapter,
        lora_layers,
        add_on_layers,
        camera_proj,
    ):
        super().__init__()
        self.unet = unet
        self.controlnet = controlnet
        self.image_proj_model = image_proj_model

        self.ip_adapter = ip_adapter
        self.lora_layers = lora_layers
        self.add_on_layers = add_on_layers
        self.camera_proj = camera_proj
    
    def forward(
        self,
        noisy_latents,
        noisy_latents_rmo,
        encoder_hidden_states,
        clip_image_embeds,
        controlnet_image,
        c2ws,
        timesteps,
        num_views,
        only_train_albedo,
    ):
        c2ws = c2ws.to(self.camera_proj.device, dtype=self.camera_proj.dtype)
        cam_mtx_flattened = einops.rearrange(c2ws, 'b n l -> (b n) l') # shape=[2,4,16]->[8,16]
        cam_mtx_emb = self.camera_proj(cam_mtx_flattened) # [8,16]->[8,1280]

        timesteps = timesteps.repeat_interleave(num_views) if timesteps.ndim == 1 else timesteps

        clip_image_embeds = self.image_proj_model(clip_image_embeds)
        clip_image_embeds = torch.cat([encoder_hidden_states, clip_image_embeds], dim=1)

        encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_views, dim=0)
        clip_image_embeds = clip_image_embeds.repeat_interleave(num_views, dim=0)

        # 1st. train with albedo
        down_block_res_samples, mid_block_res_samples, temb_residual = self.controlnet(
            sample=noisy_latents, # noisy_latents.shape=[bsz,num_views,4,32,32]
            controlnet_cond=controlnet_image, # controlnet_image.shape=[bsz*num_views,3,256,256], value_range=[0,1], 即为输入pipe的hint
            timestep=timesteps, # timesteps.shape=[bsz]
            #encoder_hidden_states=clip_image_embeds, #encoder_hidden_states, # empty prompt's encoder_hidden_states.shape=[batch_size, sequence_length, hidden_size=1024]
            encoder_hidden_states=encoder_hidden_states, # controlnet no need to use ip-adapter
            camera_matrix_embeds=cam_mtx_emb, # cam_mtx_emb.shape=[bsz*num_views,1280]
            conditioning_scale=1.0, # 1.0
            return_dict=False,
            forward_2d=False, # False
        )

        cam_mtx_emb = temb_residual

        noisy_latents = einops.rearrange(noisy_latents, 'b n c h w -> (b n) c h w')

        model_pred = self.unet( # noise_pred.shape=[bsz*num_views,4,32,32]
            noisy_latents,
            timesteps,
            encoder_hidden_states=clip_image_embeds,#encoder_hidden_states, #encoder_hidden_states.shape=[bsz, num_tokens, embed_size=1024]
            camera_matrix_embeds=cam_mtx_emb,
            cross_attention_kwargs=None,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_samples,
            sample_type='albedo',
        ).sample

        model_pred = einops.rearrange(model_pred, '(b n) c h w -> b n c h w', n=num_views)
        del down_block_res_samples

        # 2nd. train with rmo
        if not only_train_albedo:
            down_block_res_samples, mid_block_res_samples, temb_residual = self.controlnet(
                sample=noisy_latents_rmo, # noisy_latents.shape=[bsz,num_views,4,32,32]
                controlnet_cond=controlnet_image, # controlnet_image.shape=[bsz*num_views,3,256,256], value_range=[0,1], 即为输入pipe的hint
                timestep=timesteps, # timesteps.shape=[bsz]
                #encoder_hidden_states=clip_image_embeds, #encoder_hidden_states, # empty prompt's encoder_hidden_states.shape=[batch_size, sequence_length, hidden_size=1024]
                encoder_hidden_states=encoder_hidden_states, # controlnet no need to use ip-adapter
                camera_matrix_embeds=cam_mtx_emb, # cam_mtx_emb.shape=[bsz*num_views,1280]
                conditioning_scale=1.0, # 1.0
                return_dict=False,
                forward_2d=False, # False
            )
            cam_mtx_emb = temb_residual

            noisy_latents_rmo = einops.rearrange(noisy_latents_rmo, 'b n c h w -> (b n) c h w')

            model_pred_rmo = self.unet( # noise_pred.shape=[bsz*num_views,4,32,32]
                noisy_latents_rmo,
                timesteps,
                encoder_hidden_states=clip_image_embeds,#encoder_hidden_states, #encoder_hidden_states.shape=[bsz, num_tokens, embed_size=1024]
                camera_matrix_embeds=cam_mtx_emb,
                cross_attention_kwargs=None,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_samples,
                sample_type='rmo',
            ).sample

            model_pred_rmo = einops.rearrange(model_pred_rmo, '(b n) c h w -> b n c h w', n=num_views)
            del down_block_res_samples

            return model_pred, model_pred_rmo
        
        return model_pred

class MVPBRMatDiffusionWrapper_2(torch.nn.Module):
    def __init__(
        self,
        unet,
        controlnet,
        image_proj_model,
        ip_adapter,
        lora_layers,
        add_on_layers,
        camera_proj,
    ):
        super().__init__()
        self.unet = unet
        self.controlnet = controlnet
        self.image_proj_model = image_proj_model

        self.ip_adapter = ip_adapter
        self.lora_layers = lora_layers
        self.add_on_layers = add_on_layers
        self.camera_proj = camera_proj
    
    def forward(
        self,
        noisy_latents,
        noisy_latents_r,
        noisy_latents_m,
        encoder_hidden_states,
        clip_image_embeds,
        controlnet_image,
        c2ws,
        timesteps,
        num_views,
        only_train_albedo,
    ):
        c2ws = c2ws.to(self.camera_proj.device, dtype=self.camera_proj.dtype)
        cam_mtx_flattened = einops.rearrange(c2ws, 'b n l -> (b n) l') # shape=[2,4,16]->[8,16]
        cam_mtx_emb = self.camera_proj(cam_mtx_flattened) # [8,16]->[8,1280]

        timesteps = timesteps.repeat_interleave(num_views) if timesteps.ndim == 1 else timesteps
        clip_image_embeds = self.image_proj_model(clip_image_embeds)
        clip_image_embeds = torch.cat([encoder_hidden_states, clip_image_embeds], dim=1)

        encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_views, dim=0)
        clip_image_embeds = clip_image_embeds.repeat_interleave(num_views, dim=0)

        # 1st. train with albedo
        down_block_res_samples, mid_block_res_samples, temb_residual = self.controlnet(
            sample=noisy_latents, # noisy_latents.shape=[bsz,num_views,4,32,32]
            controlnet_cond=controlnet_image, # controlnet_image.shape=[bsz*num_views,3,256,256], value_range=[0,1], 即为输入pipe的hint
            timestep=timesteps, # timesteps.shape=[bsz]
            #encoder_hidden_states=clip_image_embeds, #encoder_hidden_states, # empty prompt's encoder_hidden_states.shape=[batch_size, sequence_length, hidden_size=1024]
            encoder_hidden_states=encoder_hidden_states, # controlnet no need to use ip-adapter
            camera_matrix_embeds=cam_mtx_emb, # cam_mtx_emb.shape=[bsz*num_views,1280]
            conditioning_scale=1.0, # 1.0
            return_dict=False,
            forward_2d=False, # False
        )

        cam_mtx_emb = temb_residual

        noisy_latents = einops.rearrange(noisy_latents, 'b n c h w -> (b n) c h w')

        model_pred = self.unet( # noise_pred.shape=[bsz*num_views,4,32,32]
            noisy_latents,
            timesteps,
            encoder_hidden_states=clip_image_embeds,#encoder_hidden_states, #encoder_hidden_states.shape=[bsz, num_tokens, embed_size=1024]
            camera_matrix_embeds=cam_mtx_emb,
            cross_attention_kwargs=None,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_samples,
            sample_type='albedo',
        ).sample

        model_pred = einops.rearrange(model_pred, '(b n) c h w -> b n c h w', n=num_views)
        del down_block_res_samples

        # 2nd. train with rmo
        if not only_train_albedo:
            # roughness branch
            down_block_res_samples, mid_block_res_samples, temb_residual = self.controlnet(
                sample=noisy_latents_r, # noisy_latents.shape=[bsz,num_views,4,32,32]
                controlnet_cond=controlnet_image, # controlnet_image.shape=[bsz*num_views,3,256,256], value_range=[0,1], 即为输入pipe的hint
                timestep=timesteps, # timesteps.shape=[bsz]
                #encoder_hidden_states=clip_image_embeds, #encoder_hidden_states, # empty prompt's encoder_hidden_states.shape=[batch_size, sequence_length, hidden_size=1024]
                encoder_hidden_states=encoder_hidden_states, # controlnet no need to use ip-adapter
                camera_matrix_embeds=cam_mtx_emb, # cam_mtx_emb.shape=[bsz*num_views,1280]
                conditioning_scale=1.0, # 1.0
                return_dict=False,
                forward_2d=False, # False
            )
            cam_mtx_emb = temb_residual

            noisy_latents_r = einops.rearrange(noisy_latents_r, 'b n c h w -> (b n) c h w')

            model_pred_r = self.unet( # noise_pred.shape=[bsz*num_views,4,32,32]
                noisy_latents_r,
                timesteps,
                encoder_hidden_states=clip_image_embeds,#encoder_hidden_states, #encoder_hidden_states.shape=[bsz, num_tokens, embed_size=1024]
                camera_matrix_embeds=cam_mtx_emb,
                cross_attention_kwargs=None,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_samples,
                sample_type='roughness',
            ).sample

            model_pred_r = einops.rearrange(model_pred_r, '(b n) c h w -> b n c h w', n=num_views)
            del down_block_res_samples

            # metallic branch
            down_block_res_samples, mid_block_res_samples, temb_residual = self.controlnet(
                sample=noisy_latents_m, # noisy_latents.shape=[bsz,num_views,4,32,32]
                controlnet_cond=controlnet_image, # controlnet_image.shape=[bsz*num_views,3,256,256], value_range=[0,1], 即为输入pipe的hint
                timestep=timesteps, # timesteps.shape=[bsz]
                #encoder_hidden_states=clip_image_embeds, #encoder_hidden_states, # empty prompt's encoder_hidden_states.shape=[batch_size, sequence_length, hidden_size=1024]
                encoder_hidden_states=encoder_hidden_states, # controlnet no need to use ip-adapter
                camera_matrix_embeds=cam_mtx_emb, # cam_mtx_emb.shape=[bsz*num_views,1280]
                conditioning_scale=1.0, # 1.0
                return_dict=False,
                forward_2d=False, # False
            )
            cam_mtx_emb = temb_residual

            noisy_latents_m = einops.rearrange(noisy_latents_m, 'b n c h w -> (b n) c h w')

            model_pred_m = self.unet( # noise_pred.shape=[bsz*num_views,4,32,32]
                noisy_latents_m,
                timesteps,
                encoder_hidden_states=clip_image_embeds,#encoder_hidden_states, #encoder_hidden_states.shape=[bsz, num_tokens, embed_size=1024]
                camera_matrix_embeds=cam_mtx_emb,
                cross_attention_kwargs=None,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_samples,
                sample_type='metallic',
            ).sample

            model_pred_m = einops.rearrange(model_pred_m, '(b n) c h w -> b n c h w', n=num_views)
            del down_block_res_samples

            return model_pred, model_pred_r, model_pred_m
        
        return model_pred

def multi_inference(
    vae, text_encoder, tokenizer, image_encoder, image_processor, image_proj_model, ip_adapter,
    camera_proj, scheduler, unet, controlnet, device, 
    seed, guidance_scale, num_inference_steps, validation_prompts, validation_images, num_validation_images, output_path,
    use_feature_full_token, condition_mode, multi_branch_type, do_mv_super_res,
):  
    if multi_branch_type == 'v1':
        from extern.mvcontrol.pipeline_mvcontrol_multi_branch import MVControlPipeline
    elif multi_branch_type == 'v2':
        from extern.mvcontrol.pipeline_mvcontrol_multi_branch_2 import MVControlPipeline
    else:
        raise NotImplementedError(f"[INFO] Invalid multi_branch_type: {multi_branch_type}")
    
    pipeline = MVControlPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        image_encoder=image_encoder,
        image_processor=image_processor,
        unet=unet,
        camera_proj=camera_proj,
        scheduler=scheduler,
        controlnet=controlnet,
        safety_checker=None,
        device=unet.device,
        image_proj_model=image_proj_model,
        ip_adapter=ip_adapter,
        use_feature_full_token=use_feature_full_token,
    )
    
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.set_progress_bar_config(disable=True)

    if seed is None:
        generator = None
    else:
        generator = torch.Generator(device=device).manual_seed(seed)

    image_logs = []
    inference_ctx = contextlib.nullcontext()

    os.makedirs(output_path, exist_ok=True)
    for validation_prompt, validation_image in tqdm.tqdm(zip(validation_prompts, validation_images)):
        mv_validation_images = []
        for v in ['front', 'right', 'back', 'left', 'top', 'bottom',]:
            tmp = Image.open(f'{validation_image}/{v}.png').convert("RGB")
            mv_validation_images.append(tmp)
        ip_image = Image.open(f'{validation_image}/ip.png').convert("RGB")

        c2ws = get_camera(num_frames=4, elevation=0, azimuth_start=0, azimuth_span=360) # cameras of left, front, right, and back
        c2ws_top_bottom = get_top_bottom_camera(top_elev_azi={'elev':90, 'azi': 0}, bottom_elev_azi={'elev': -90, 'azi': 180}) # cameras of top and bottom
        c2ws = torch.cat([c2ws, c2ws_top_bottom], dim=0)

        if unet.is_multi_branch:
            images, rs, ms, rmos = [], [], [], []
        else:
            images = []

        for gs in guidance_scale:
            for _ in range(num_validation_images):
                with inference_ctx:
                    pipe_out = pipeline(
                        validation_prompt, ip_image, mv_validation_images, c2ws, num_inference_steps=num_inference_steps, generator=generator, guidance_scale=gs,
                        condition_mode=condition_mode,
                    )
                    if multi_branch_type == 'v1':
                        image, rmo = pipe_out.images, pipe_out.rmo
                    elif multi_branch_type == 'v2':
                        image, image_r, image_m = pipe_out.images, pipe_out.roughness, pipe_out.metallic
                    image = make_image_grid(image, rows=2)

                images.append(image)

                if unet.is_multi_branch:
                    if multi_branch_type == 'v1':
                        roughness, metalness, rm = [], [], []
                        for tmp in rmo:
                            rmo_np = np.asarray(tmp)
                            roughness_np, metalness_np = rmo_np[...,0:1], rmo_np[...,1:2]
                            rm_np = np.concatenate([roughness_np, metalness_np, np.zeros_like(roughness_np)], axis=-1)
                            roughness_np, metalness_np = np.concatenate([roughness_np] * 3, axis=-1), np.concatenate([metalness_np] * 3, axis=-1)
                            rm.append(Image.fromarray(rm_np))
                            roughness.append(Image.fromarray(roughness_np))
                            metalness.append(Image.fromarray(metalness_np))
                        roughness = make_image_grid(roughness, rows=2)
                        metalness = make_image_grid(metalness, rows=2)
                        rm = make_image_grid(rm, rows=2)

                        rs.append(roughness)
                        ms.append(metalness)
                        rmos.append(rm)
                    elif multi_branch_type == 'v2':
                        roughness, metalness, rm = [], [], []
                        for tmp_r, tmp_m in zip(image_r, image_m):
                            roughness_np, metalness_np = np.asarray(tmp_r, dtype=np.float32), np.asarray(tmp_m, dtype=np.float32)
                            roughness_np, metalness_np = roughness_np.mean(axis=-1, keepdims=True), metalness_np.mean(axis=-1, keepdims=True)
                            rm_np = np.concatenate([roughness_np, metalness_np, np.zeros_like(roughness_np)], axis=-1)
                            roughness_np, metalness_np = np.concatenate([roughness_np] * 3, axis=-1), np.concatenate([metalness_np] * 3, axis=-1)
                            rm.append(Image.fromarray(rm_np.astype(np.uint8)))
                            roughness.append(Image.fromarray(roughness_np.astype(np.uint8)))
                            metalness.append(Image.fromarray(metalness_np.astype(np.uint8)))
                        roughness = make_image_grid(roughness, rows=2)
                        metalness = make_image_grid(metalness, rows=2)
                        rm = make_image_grid(rm, rows=2)

                        rs.append(roughness)
                        ms.append(metalness)
                        rmos.append(rm)

        if unet.is_multi_branch:
            image_logs.append(
                {"mv_validation_image": make_image_grid(mv_validation_images, rows=2), "images": images, "rs": rs, "ms": ms, "rmos": rmos, "validation_prompt": validation_prompt}
            )
        else:
            image_logs.append(
                {"mv_validation_image": make_image_grid(mv_validation_images, rows=2), "images": images, "validation_prompt": validation_prompt}
            )

        full_mv_validation_generated_images = []
        for log in image_logs:
            images = log["images"]
            if unet.is_multi_branch:
                rs, ms, rmos = log["rs"], log["ms"], log["rmos"]
            validation_prompt = log["validation_prompt"]
            mv_validation_image = log["mv_validation_image"]

            formatted_images = []

            if mv_validation_image.size[0] != 256 * 3 or mv_validation_image.size[1] != 256 * 2:
                mv_validation_image = mv_validation_image.resize(size=(256 * 3, 256 * 2))
            formatted_images.append(mv_validation_image)
            if unet.is_multi_branch:
                for image, r, m, rmo in zip(images, rs, ms, rmos):
                    #formatted_images.append(np.asarray(image))
                    formatted_images.append(image)
                    formatted_images.append(r)
                    formatted_images.append(m)
                    formatted_images.append(rmo)
            else:
                for image in images:
                    formatted_images.append(image)
            full_mv_validation_generated_images.append(make_image_grid(formatted_images, rows=1))
        print_green(validation_image.split('/')[-1] + '.png')

        res = np.array(make_image_grid(formatted_images, rows=1))

        # inference result
        output_name = validation_image.split('/')[-1]
        Image.fromarray(res).save(f"{output_path}/{output_name}.png")

        # split images and multi-view super-resolution
        os.makedirs(f"{output_path}/{output_name}", exist_ok=True)
        if do_mv_super_res:
            model_zoo.init_models()
            mv_super_res_pipe = model_zoo.pipe_disney_controlnet_tile_ipadapter_i2i
            mv_super_res_pipe = mv_super_res_pipe.to('cuda') # 'diffusers.pipelines.controlnet.pipeline_controlnet_img2img.StableDiffusionControlNetImg2ImgPipeline'

        for i, j in tqdm.tqdm(zip([1, 2, 3], ['a', 'r', 'm'])):
            rgb_pils, front_pil = [], None
            cut = res[:, 256*3*i:256*3*(i+1), :]

            for h, w, v in zip([0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2], ['front', 'right', 'back', 'left', 'top', 'bottom']):
                cut_sub = cut[256*h:256*(h+1), 256*w:256*(w+1), :]

                if v == 'front':
                    front_pil = cut_sub

                if (v in ['front', 'right', 'back', 'left']) and do_mv_super_res:
                    rgb_pils.append(Image.fromarray(cut_sub))
                else:
                    Image.fromarray(cut_sub).save(f"{output_path}/{output_name}/{output_name}_{j}_{v}.png")

            if do_mv_super_res:
                mv_super_resolution(pipe=mv_super_res_pipe, rgb_pils=rgb_pils, front_pil=front_pil, output_path=output_path, output_name=output_name, material_type=j)

def mv_super_resolution(pipe, rgb_pils, front_pil, output_path, output_name, material_type):
    rgb_pil = make_image_grid(rgb_pils, rows=2)

    prompt = "4views, multiview"
    neg_prompt = "sketch, sculpture, hand drawing, outline, single color, NSFW, lowres, bad anatomy,bad hands, text, error, missing fingers, yellow sleeves, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry,(worst quality:1.4),(low quality:1.4)"
    control_image = rgb_pil.resize((1024, 1024))
    #refined_rgb = refine_lr_with_sd([rgb_pil], [rgba_to_rgb(front_pil)], [control_image], prompt_list=[prompt], neg_prompt_list=[neg_prompt], pipe=model_zoo.pipe_disney_controlnet_tile_ipadapter_i2i, strength=0.2, output_size=(1024, 1024))[0]

    with torch.no_grad():
        refined_rgb = pipe(
            image=[rgb_pil],
            ip_adapter_image=[front_pil],
            prompt=[prompt],
            neg_prompt=[neg_prompt],
            num_inference_steps=50,
            strength=0.2,
            height=1024,
            width=1024,
            control_image=[control_image],
            guidance_scale=5.0,
            controlnet_conditioning_scale=1.0,
            generator=torch.Generator(device="cuda").manual_seed(233),
        ).images[0]
    
    refined_rgbs = split_image(refined_rgb, rows=2)
    for refine_rgb, v in zip(refined_rgbs, ['front', 'right', 'back', 'left']):
        refine_rgb.save(f"{output_path}/{output_name}/{output_name}_{material_type}_{v}_upscale.png")
    
def generate_image_prompt_via_normal_controlnet(normal_path, prompt, controlnet_cond_mode, seed):
    normal = Image.open(normal_path)
        
    if controlnet_cond_mode == 'text':
        sd_ckpt = "/cpfs01/user/wangyitong/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9"
        ctrlnet_ckpt = "/cpfs01/user/wangyitong/.cache/huggingface/hub/models--lllyasviel--control_v11p_sd15_normalbae/snapshots/cb7296e6587a219068e9d65864e38729cd862aa8"

        control_image = normal

        control_mask = np.where(np.array(control_image) > 0, 1.0, 0.0)

        controlnet = ControlNetModel.from_pretrained(ctrlnet_ckpt, torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            pretrained_model_name_or_path=sd_ckpt, 
            controlnet=controlnet, torch_dtype=torch.float16
        )
        print_green("[INFO] Text-guided normal ControlNet is loaded.")

        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()

        generator = torch.manual_seed(seed=seed)
        image = pipe(prompt, num_inference_steps=30, generator=generator, image=control_image).images[0]

        image = Image.fromarray((np.array(image, dtype=np.float32) * control_mask).astype(np.uint8))
        image.save(f"{os.path.dirname(normal_path)}/ip.png")
    
    elif controlnet_cond_mode == 'image':
        sd_ckpt = "/cpfs01/shared/landmark_3dgen/xuxudong_group/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/f03de327dd89b501a01da37fc5240cf4fdba85a1"
        ctrlnet_ckpt = "/cpfs01/user/wangyitong/.cache/huggingface/hub/models--lllyasviel--control_v11p_sd15_normalbae/snapshots/cb7296e6587a219068e9d65864e38729cd862aa8"
        vae_ckpt = "/cpfs01/user/wangyitong/.cache/huggingface/hub/models--stabilityai--sd-vae-ft-mse/snapshots/31f26fdeee1355a5c34592e401dd41e45d25a493"
        image_encoder_ckpt = "/cpfs01/shared/landmark_3dgen/xuxudong_group/huggingface/hub/models--h94--IP-Adapter/snapshots/018e402774aeeddd60609b4ecdb7e298259dc729/models/image_encoder/"
        ip_ckpt = "/cpfs01/shared/landmark_3dgen/xuxudong_group/huggingface/hub/models--h94--IP-Adapter/snapshots/018e402774aeeddd60609b4ecdb7e298259dc729/models/ip-adapter_sd15.safetensors"
        device = "cuda"

        noise_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )
        vae = AutoencoderKL.from_pretrained(vae_ckpt).to(dtype=torch.float16)

        # load controlnet
        controlnet = ControlNetModel.from_pretrained(ctrlnet_ckpt, torch_dtype=torch.float16)

        # load SD pipeline
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            sd_ckpt,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            scheduler=noise_scheduler,
            vae=vae,
            feature_extractor=None,
            safety_checker=None
        )

        # read image prompt
        prompt = Image.open(prompt)
        mask = np.where(np.array(normal, dtype=np.float32) > 0.0, 1.0, 0.0)

        # load ip-adapter
        ip_model = IPAdapter(pipe, image_encoder_ckpt, ip_ckpt, device)
        print_green("[INFO] Image-guided normal ControlNet is loaded.")

        # generate image variations
        images = ip_model.generate(pil_image=prompt, image=normal, num_samples=1, num_inference_steps=30, seed=seed)
        images[0] = Image.fromarray((np.array(images[0], dtype=np.float32) * mask).astype(np.uint8))
        images[0].save(f"{os.path.dirname(normal_path)}/ip.png")

    else:
        NotImplementedError("[INFO] Invalid ControlNet condition mode.")

def inference_multi_branch(pretrained_path, lora_rank, guidance_scale, num_inference_steps, seed, 
                            input_path, out_path, use_feature_full_token, condition_mode, multi_branch_type, 
                            do_mv_super_res, do_single_inference):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print_green(device)

    # base model initialization
    mv_pipe = load_mvcontrol_pipeline(
        pretrained_model_name_or_path="/cpfs01/user/wangyitong/.cache/huggingface/hub/models--lzq49--mvdream-sd21-diffusers/snapshots/0c0f76ed8d4e664b6615cc5b4529df165246fe6e",
        pretrained_controlnet_name_or_path="/cpfs01/user/wangyitong/.cache/huggingface/hub/models--lzq49--mvcontrol-4v-normal/snapshots/379a2e50943d46dcf479c7b33dedb6fb6df74620",
        weights_dtype=torch.float32,
        num_views=6,
        device=device,
        enable_unet_multi_branch=True,
        multi_branch_type=multi_branch_type,
    )
    noise_scheduler = mv_pipe.scheduler
    text_encoder = mv_pipe.text_encoder # process text embeddings in Dataset stage
    vae = mv_pipe.vae
    unet = mv_pipe.unet
    camera_proj = mv_pipe.camera_proj
    tokenizer = AutoTokenizer.from_pretrained(
            "/cpfs01/user/wangyitong/.cache/huggingface/hub/models--lzq49--mvdream-sd21-diffusers/snapshots/0c0f76ed8d4e664b6615cc5b4529df165246fe6e",
            subfolder="tokenizer",
            use_fast=False,
        )

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        pretrained_model_name_or_path='/cpfs01/user/wangyitong/.cache/huggingface/hub/models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K/snapshots/de081ac0a0ca8dc9d1533eed1ae884bb8ae1404b',
    ).to(device=device, dtype=torch.float16)
    image_processor = CLIPImageProcessor.from_pretrained(
        pretrained_model_name_or_path='/cpfs01/user/wangyitong/.cache/huggingface/hub/models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K/snapshots/de081ac0a0ca8dc9d1533eed1ae884bb8ae1404b',
    )
    
    #ip-adapter
    if use_feature_full_token:
        image_proj_model = FullImageTokenProjModel(
            cross_attention_dim=unet.config.cross_attention_dim,
            feature_embeddings_dim=image_encoder.config.hidden_size,
        )
    else:
        image_proj_model = ImageProjModel(
            cross_attention_dim=unet.config.cross_attention_dim,
            clip_embeddings_dim=image_encoder.config.projection_dim,
            clip_extra_context_tokens=4,
        )

    # ip-adapter modules initialization
    if is_xformers_available():
        import xformers
        attn_procs_cls = XFormersCrossViewAttnProcessor
    else:
        attn_procs_cls = CrossViewAttnProcessor
    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = attn_procs_cls(num_views=6)
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            if use_feature_full_token:
                attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, num_tokens=257)
            else:
                attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, num_tokens=4)
            attn_procs[name].load_state_dict(weights)
    unet.set_attn_processor(attn_procs)
    ip_adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())

    for param in unet.parameters():
        param.requires_grad_(False)

    # unet.add_adapter(unet_lora_config) will apply LoRA to all blocks. 
    # However, the duplicated outermost blocks (i.e. multi-branch) need full-param tuning instead of LoRA tuning.
    # So we need to deepcopy these outermost blocks and replace corresponding LoRA-based ones after unet.add_adapter(unet_lora_config)
    if multi_branch_type == 'v1':
        unet.enable_multi_branch()
        tmp_unet_down_blocks_0 = copy.deepcopy(unet.down_blocks[0])
        tmp_unet_down_blocks_0.load_state_dict(unet.down_blocks[0].state_dict())

        tmp_unet_down_blocks_0_rmo = copy.deepcopy(unet.down_blocks_0_rmo)
        tmp_unet_down_blocks_0_rmo.load_state_dict(unet.down_blocks_0_rmo.state_dict())

        tmp_unet_up_blocks_last = copy.deepcopy(unet.up_blocks[-1])
        tmp_unet_up_blocks_last.load_state_dict(unet.up_blocks[-1].state_dict())

        tmp_unet_up_blocks_last_rmo = copy.deepcopy(unet.up_blocks_last_rmo)
        tmp_unet_up_blocks_last_rmo.load_state_dict(unet.up_blocks_last_rmo.state_dict())
    elif multi_branch_type == 'v2':
        unet.enable_multi_branch()
        # albedo
        tmp_unet_down_blocks_0 = copy.deepcopy(unet.down_blocks[0])
        tmp_unet_down_blocks_0.load_state_dict(unet.down_blocks[0].state_dict())
        # roughness
        tmp_unet_down_blocks_0_r = copy.deepcopy(unet.down_blocks_0_r)
        tmp_unet_down_blocks_0_r.load_state_dict(unet.down_blocks_0_r.state_dict())
        # metallic
        tmp_unet_down_blocks_0_m = copy.deepcopy(unet.down_blocks_0_m)
        tmp_unet_down_blocks_0_m.load_state_dict(unet.down_blocks_0_m.state_dict())

        tmp_unet_up_blocks_last = copy.deepcopy(unet.up_blocks[-1])
        tmp_unet_up_blocks_last.load_state_dict(unet.up_blocks[-1].state_dict())

        tmp_unet_up_blocks_last_r = copy.deepcopy(unet.up_blocks_last_r)
        tmp_unet_up_blocks_last_r.load_state_dict(unet.up_blocks_last_r.state_dict())

        tmp_unet_up_blocks_last_m = copy.deepcopy(unet.up_blocks_last_m)
        tmp_unet_up_blocks_last_m.load_state_dict(unet.up_blocks_last_m.state_dict())
    print_green('use multi-branch')

    # add LoRA
    unet_lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=1,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(unet_lora_config)

    # replace LoRA-based outermost blocks with the original ones
    if multi_branch_type == 'v1':
        unet.down_blocks[0] = tmp_unet_down_blocks_0
        unet.down_blocks_0_rmo = tmp_unet_down_blocks_0_rmo
        unet.up_blocks[-1] = tmp_unet_up_blocks_last
        unet.up_blocks_last_rmo = tmp_unet_up_blocks_last_rmo
    elif multi_branch_type == 'v2':
        unet.down_blocks[0] = tmp_unet_down_blocks_0
        unet.down_blocks_0_r = tmp_unet_down_blocks_0_r
        unet.down_blocks_0_m = tmp_unet_down_blocks_0_m
        unet.up_blocks[-1] = tmp_unet_up_blocks_last
        unet.up_blocks_last_r = tmp_unet_up_blocks_last_r
        unet.up_blocks_last_m = tmp_unet_up_blocks_last_m

    controlnet = mv_pipe.controlnet.to(dtype=torch.float32)
    
    if multi_branch_type == 'v1':
        mv_diff_wrapper = MVPBRMatDiffusionWrapper(
            unet=unet, controlnet=controlnet, image_proj_model=image_proj_model,
            ip_adapter=ip_adapter_modules, lora_layers=None, add_on_layers=None, camera_proj=camera_proj,
        )
    elif multi_branch_type == 'v2':
        mv_diff_wrapper = MVPBRMatDiffusionWrapper_2(
            unet=unet, controlnet=controlnet, image_proj_model=image_proj_model,
            ip_adapter=ip_adapter_modules, lora_layers=None, add_on_layers=None, camera_proj=camera_proj,
        )

    if re.match(r"checkpoint-\d+", pretrained_path.split('/')[-1]):
        print_green(f"Use checkpoints: {pretrained_path}")
        mv_diff_wrapper.load_state_dict(torch.load(f'{pretrained_path}/pytorch_model.bin'), strict=False)
        mv_diff_wrapper = mv_diff_wrapper.to(device, dtype=torch.float16)
    else:
        print_green('Use final save')
        image_proj_model.load_state_dict(torch.load(f'{pretrained_path}/image_proj_model.pt'))
        image_proj_model = image_proj_model.to(device=device, dtype=torch.float16)
        print_green('Image Projection Model is loaded.')

        ip_adapter_modules.load_state_dict(torch.load(f'{pretrained_path}/ip_adapter.pt'))
        ip_adapter_modules = ip_adapter_modules.to(device=device, dtype=torch.float16)
        print_green('IP-Adapter is loaded.')

        camera_proj.load_state_dict(torch.load(f'{pretrained_path}/camera_proj.pt'))
        camera_proj = camera_proj.to(device=device, dtype=torch.float16)
        print_green('Camera Projection Model is loaded.')

        unet.load_state_dict(torch.load(f'{pretrained_path}/unet.pt'))
        unet = unet.to(device=device, dtype=torch.float16)
        print_green('UNet LoRA and Add-on Layers are loaded.')

        controlnet = MultiViewControlNetModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_path,
        ).to(device=device, dtype=torch.float16)
        if is_xformers_available():
            import xformers
            attn_procs_cls = XFormersCrossViewAttnProcessor
        else:
            attn_procs_cls = CrossViewAttnProcessor

        set_self_attn_processor(
            controlnet, attn_procs_cls(num_views=6)
        )
        print_green('Multi-view ControlNet is loaded.')

    if do_single_inference:
        validation_images = [input_path]
    else:
        validation_images = sorted(os.listdir(input_path))
        tmp = []
        for v in validation_images:
            if v not in ['__pycache__', '.ipynb_checkpoints',]:
            #if v in ['human_shape_120_610746', 'human_shape_154_783903']:
                tmp.append(f'{input_path}/{v}')
        validation_images = tmp
        del tmp
    validation_prompts = [""] * len(validation_images)

    multi_inference(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        image_encoder=image_encoder,
        image_processor=image_processor,
        image_proj_model=image_proj_model,
        ip_adapter=ip_adapter_modules,
        camera_proj=camera_proj,
        scheduler=noise_scheduler,
        unet=unet,
        controlnet=controlnet,
        device=device,
        seed=seed,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        validation_prompts=validation_prompts,
        validation_images=validation_images,
        num_validation_images=1,
        output_path=out_path,
        use_feature_full_token=use_feature_full_token,
        condition_mode=condition_mode,
        multi_branch_type=multi_branch_type,
        do_mv_super_res=do_mv_super_res,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inference code for Multi-view PBR Material Diffusion Model.")
    parser.add_argument("--ctrlnet_seed", type=int, default=42)
    parser.add_argument("--mvdiff_seed", type=int, default=42)
    parser.add_argument("--controlnet_normal_path", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--controlnet_cond_mode", type=str, default=None)
    parser.add_argument("--input_path", type=str, default=None)
    parser.add_argument("--out_path", type=str, default=None)
    parser.add_argument("--do_mv_super_res", action="store_true")
    args = parser.parse_args()

    generate_image_prompt_via_normal_controlnet(
        normal_path=args.controlnet_normal_path,
        prompt=args.prompt,
        controlnet_cond_mode=args.controlnet_cond_mode,
        seed=args.ctrlnet_seed,
    )

    #pretrained_path = '/cpfs01/shared/landmark_3dgen/xuxudong_group/wangyitong/ai_demo/mvdream_control/trained_weights/gpu32_lora_rank16_alpha1_mat3d_clip_aesthetic_6views_unified_lora_branch_lr2e-4_other_lr1e-4_al0.5_rm0.5_full_image_token/checkpoint-50000'
    #pretrained_path = '/cpfs01/shared/landmark_3dgen/xuxudong_group/wangyitong/ai_demo/mvdream_control/trained_weights/gpu32_lora_rank16_alpha1_mat3d_clip_aesthetic_6views_unified_lora_branch_lr2e-4_other_lr1e-4_al0.75_rm0.25_full_image_token/checkpoint-50000'
    pretrained_path = '/cpfs01/shared/landmark_3dgen/xuxudong_group/wangyitong/ai_demo/mvdream_control/trained_weights/gpu32_lora_rank16_alpha1_mat3d_clip_aesthetic_6views_perturb_front_unified_lora_branch_lr2e-4_other_lr1e-4_al0.5_r0.25_m0.25_full_image_token'

    inference_multi_branch(
        pretrained_path=pretrained_path,
        lora_rank=16,
        guidance_scale=[7.5],
        num_inference_steps=30,
        seed=args.mvdiff_seed, # 3407
        input_path=args.input_path,
        out_path=args.out_path,
        use_feature_full_token=True,
        condition_mode='image', # text, image, text-image
        multi_branch_type='v2',
        do_mv_super_res=args.do_mv_super_res,
        do_single_inference=True, # True: only inference one case, plz provide the path of a single case to input_path. False: inference a folder of multiple cases, plz provide the path of a folder which contains multiple cases.
    )