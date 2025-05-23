from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import os
import inspect
import PIL
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from extern.mvcontrol.unet import UNet2DConditionModel
from extern.mvcontrol.camera_proj import CameraMatrixEmbedding
from extern.mvcontrol.mvcontrolnet import MultiViewControlNetModel
from extern.mvcontrol.scheduler import DDIMScheduler_ as DDIMScheduler
from extern.mvcontrol.utils.typing import *
from extern.ip_adapter.ip_adapter import ImageProjModel

from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor, CLIPVisionModel, CLIPModel, CLIPVisionModelWithProjection
import diffusers
from diffusers import (
    ModelMixin,
    AutoencoderKL,
    ControlNetModel,
    DDIMScheduler,
    # UNet2DConditionModel,
    DiffusionPipeline,
)
from diffusers.loaders import TextualInversionLoaderMixin, LoraLoaderMixin
from diffusers.utils import (
    PIL_INTERPOLATION,
    is_accelerate_available,
    is_accelerate_version,
    is_xformers_available,
    logging,
    BaseOutput
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.stable_diffusion import (
    StableDiffusionSafetyChecker, 
)
import pdb

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class MVControlPipelineOutput(BaseOutput):
    """
    Output class for Stable Diffusion pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
        nsfw_content_detected (`List[bool]`)
            List of flags denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, or `None` if safety checking could not be performed.
    """

    images: Union[List[PIL.Image.Image], np.ndarray, torch.Tensor]


class MVControlPipeline(DiffusionPipeline):
    r"""
    Copied diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_controlnet.StableDiffusionControlNetPipeline
    and rewrite to adapt to our modifications
    """
    _optional_components = ["feature_extractor", "image_encoder"]

    def __init__(
        self,
        vae: AutoencoderKL,
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTextModel,
        image_processor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection,
        unet: UNet2DConditionModel,
        camera_proj: CameraMatrixEmbedding,
        scheduler: DDIMScheduler,
        controlnet: MultiViewControlNetModel = None,
        unet_2d: UNet2DConditionModel = None,
        safety_checker: StableDiffusionSafetyChecker = None,
        feature_extractor: CLIPImageProcessor = None,
        requires_safety_checker: bool = False,
        device = None,
        # ip-adapter
        image_proj_model: ImageProjModel = None,
        ip_adapter = None,
    ): 
        super().__init__()

        self.register_modules(
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            image_processor=image_processor,
            image_encoder=image_encoder,
            unet=unet,
            controlnet=controlnet,
            camera_proj=camera_proj,
            scheduler=scheduler,
            unet_2d=unet_2d,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_proj_model=image_proj_model,
            ip_adapter=ip_adapter,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

        self._device = device
        

     # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_slicing
    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding.

        When this option is enabled, the VAE will split the input tensor in slices to compute decoding in several
        steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_slicing
    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously invoked, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
        text_encoder, vae, controlnet, and safety checker have their state dicts saved to CPU and then are moved to a
        `torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
        Note that offloading happens on a submodule basis. Memory savings are higher than with
        `enable_model_cpu_offload`, but performance is lower.
        """
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.condition_encoder, self.vae, self.controlnet]:
            cpu_offload(cpu_offloaded_model, device)

        if self.safety_checker is not None:
            cpu_offload(self.safety_checker, execution_device=device, offload_buffers=True)

    def enable_model_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
        to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
        method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
        `enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError("`enable_model_cpu_offload` requires `accelerate v0.17.0` or higher.")

        device = torch.device(f"cuda:{gpu_id}")

        hook = None
        for cpu_offloaded_model in [self.condition_encoder, self.unet, self.vae]:
            _, hook = cpu_offload_with_hook(cpu_offloaded_model, device, prev_module_hook=hook)

        if self.safety_checker is not None:
            # the safety checker can offload the vae again
            _, hook = cpu_offload_with_hook(self.safety_checker, device, prev_module_hook=hook)

        # control net hook has be manually offloaded as it alternates with unet
        cpu_offload_with_hook(self.controlnet, device)

        # We'll offload the last model manually.
        self.final_offload_hook = hook

    @property
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._execution_device
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device


    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt
    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            # untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            # if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            #     text_input_ids, untruncated_ids
            # ):
            #     removed_text = self.tokenizer.batch_decode(
            #         untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
            #     )
            #     logger.warning(
            #         "The following part of your input was truncated because CLIP can only handle sequences up to"
            #         f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            #     )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds
    
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline
    def _clip_encode_image(self, image, device, num_images_per_prompt, do_classifier_free_guidance, output_hidden_states=None):
        dtype = next(self.image_encoder.parameters()).dtype
        image = self.image_processor(image, return_tensors="pt").pixel_values # [bsz, 3, h, w]->[bsz, 3, 224, 224]
        image = image.to(device=device, dtype=dtype)

        embed_type = 'image_embeds'

        if embed_type == 'image_embeds':
            embeds = self.image_encoder(image, output_hidden_states=output_hidden_states).image_embeds
            if do_classifier_free_guidance:
                uncond_embeds = self.image_encoder(torch.zeros_like(image), output_hidden_states=True).image_embeds
                # For classifier free guidance, we need to do two forward passes.
                # Here we concatenate the unconditional and text embeddings into a single batch
                # to avoid doing two forward passes
                embeds = torch.cat([uncond_embeds, embeds])

            embeds = embeds.unsqueeze(1)
        
        elif embed_type == 'hidden_states':
            hidden_states = self.image_encoder(_image, output_hidden_states=output_hidden_states).hidden_states # hidden_states[i].shape=[bsz, 257, 1280]
            embeds = hidden_states[-2]

            del hidden_states
        
        else:
            raise Exception('[INFO] Invalid embed_type.')
        
        return embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker
    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        else:
            has_nsfw_concept = None
        return image, has_nsfw_concept
    
    def encode_image(self, images, ):
        posterior = self.vae.encode(images).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents
    def decode_latents(self, latents, post_procs=True):
        # Maybe use inner batch
        if latents.ndim == 5:
            use_inner_batch = True
            B, inner_B = latents.shape[:2]
            latents = rearrange(latents, 'b n c h w -> (b n) c h w')
        else:
            use_inner_batch = False
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample

        if post_procs:
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        if use_inner_batch:
            image = image.reshape(B, inner_B, *image.shape[-3:])

        return image

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_image(self, image, prompt, prompt_embeds):
        image_is_pil = isinstance(image, PIL.Image.Image)
        image_is_tensor = isinstance(image, torch.Tensor)
        image_is_pil_list = isinstance(image, list) and isinstance(image[0], PIL.Image.Image)
        image_is_tensor_list = isinstance(image, list) and isinstance(image[0], torch.Tensor)

        if not image_is_pil and not image_is_tensor and not image_is_pil_list and not image_is_tensor_list:
            raise TypeError(
                "image must be passed and be one of PIL image, torch tensor, list of PIL images, or list of torch tensors"
            )

        if image_is_pil:
            image_batch_size = 1
        elif image_is_tensor:
            image_batch_size = image.shape[0]
        elif image_is_pil_list:
            image_batch_size = len(image)
        elif image_is_tensor_list:
            image_batch_size = len(image)

        if prompt is not None and isinstance(prompt, str):
            prompt_batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            prompt_batch_size = len(prompt)
        elif prompt_embeds is not None:
            prompt_batch_size = prompt_embeds.shape[0]

        if image_batch_size != 1 and image_batch_size != prompt_batch_size:
            raise ValueError(
                f"If image batch size is not 1, image batch size must be same as prompt batch size. image batch size: {image_batch_size}, prompt batch size: {prompt_batch_size}"
            )
        
    def prepare_rays(self, rays: Dict[str, torch.FloatTensor], do_classifier_free_guidance):
        """
        rays: Dict['rays_o': (B, N, 3), 'rays_d': (B, N, 3)]
        """
        if do_classifier_free_guidance:
            rays_o, rays_d = rays['rays_o'], rays['rays_d']
            rays_o = torch.cat([rays_o] * 2)
            rays_d = torch.cat([rays_d] * 2)
            rays['rays_o'] = rays_o
            rays['rays_d'] = rays_d
        else:
            ...

        return rays


    def prepare_hint(
        self, 
        image, 
        width, 
        height, 
        batch_size, 
        num_images_per_prompt, 
        device, 
        dtype, 
        do_classifier_free_guidance=False,
        guess_mode=False,
        blind_control=False,
    ):
        if not isinstance(image, torch.Tensor):
            if isinstance(image, PIL.Image.Image):
                image = [image]

            if isinstance(image[0], PIL.Image.Image):
                images = []

                for image_ in image:
                    image_ = image_.convert("RGB")
                    image_ = image_.resize((width, height), resample=PIL_INTERPOLATION["lanczos"])
                    image_ = np.array(image_)
                    image_ = image_[None, :]
                    images.append(image_)

                image = images

                image = np.concatenate(image, axis=0)
                image = np.array(image).astype(np.float32) / 255.0
                image = image.transpose(0, 3, 1, 2)
                image = torch.from_numpy(image)
            elif isinstance(image[0], torch.Tensor):
                image = torch.cat(image, dim=0)

        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            neg_hint = torch.zeros_like(image) if blind_control else image
            image = torch.cat([neg_hint, image])

        return image
    
    def prepare_mv_hint(
        self, 
        image, 
        width, 
        height, 
        batch_size, 
        num_images_per_prompt, 
        device, 
        dtype, 
        do_classifier_free_guidance=False,
        guess_mode=False,
        blind_control=False,
    ):
        if not isinstance(image, torch.Tensor):
            if isinstance(image, PIL.Image.Image):
                image = [image]

            if isinstance(image[0], PIL.Image.Image):
                images = []

                for image_ in image:
                    image_ = image_.convert("RGB")
                    image_ = image_.resize((width, height), resample=PIL_INTERPOLATION["lanczos"])
                    image_ = np.array(image_)
                    image_ = image_[None, :]
                    images.append(image_)

                image = images

                image = np.concatenate(image, axis=0)
                image = np.array(image).astype(np.float32) / 255.0
                image = image.transpose(0, 3, 1, 2)
                image = torch.from_numpy(image)
            elif isinstance(image[0], torch.Tensor):
                image = torch.cat(image, dim=0)

        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            neg_hint = torch.zeros_like(image) if blind_control else image
            image = torch.cat([neg_hint, image])

        return image

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(
        self, 
        batch_size, 
        num_views,
        num_channels_latents, 
        height, 
        width, 
        dtype, 
        device, 
        generator, 
        from_same_latent=True,
        latents=None
    ):
        shape = (batch_size, num_views, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        if from_same_latent:
            latents = latents[:, :1].expand_as(latents)
        return latents

    def _default_height_width(self, height, width, image):
        # NOTE: It is possible that a list of images have different
        # dimensions for each image, so just checking the first image
        # is not _exactly_ correct, but it is simple.
        while isinstance(image, list):
            image = image[0]

        if height is None:
            if isinstance(image, PIL.Image.Image):
                height = image.height
            elif isinstance(image, torch.Tensor):
                height = image.shape[2]

            height = (height // 8) * 8  # round down to nearest multiple of 8

        if width is None:
            if isinstance(image, PIL.Image.Image):
                width = image.width
            elif isinstance(image, torch.Tensor):
                width = image.shape[3]

            width = (width // 8) * 8  # round down to nearest multiple of 8

        return height, width

    # override DiffusionPipeline
    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        safe_serialization: bool = False,
        variant: Optional[str] = None,
    ):
        if isinstance(self.controlnet, ControlNetModel):
            super().save_pretrained(save_directory, safe_serialization, variant)
        else:
            raise NotImplementedError("Currently, the `save_pretrained()` is not implemented for Multi-ControlNet.")
    
    # @replace_example_docstring(EXAMPLE_DOC_STRING)
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: Union[torch.FloatTensor, PIL.Image.Image, List[torch.FloatTensor], List[PIL.Image.Image]] = None,
        hint: Union[torch.FloatTensor, PIL.Image.Image, List[torch.FloatTensor], List[PIL.Image.Image]] = None,
        c2ws: Optional[torch.FloatTensor] = None,
        height: Optional[int] = 256,
        width: Optional[int] = 256,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        start_from_same_latent: bool = False,
        guess_mode: bool = False,
        blind_control: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        blind_control_until_step: Optional[int] = None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            image (`torch.FloatTensor`, `PIL.Image.Image`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`,
                    `List[List[torch.FloatTensor]]`, or `List[List[PIL.Image.Image]]`):
                The ControlNet input condition. ControlNet uses this input condition to generate guidance to Unet. If
                the type is specified as `Torch.FloatTensor`, it is passed to ControlNet as is. `PIL.Image.Image` can
                also be accepted as an image. The dimensions of the output image defaults to `image`'s dimensions. If
                height and/or width are passed, `image` is resized according to them. If multiple ControlNets are
                specified in init, images must be passed as a list such that each element of the list can be correctly
                batched for input to a single controlnet.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
            controlnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):
                The outputs of the controlnet are multiplied by `controlnet_conditioning_scale` before they are added
                to the residual in the original unet. If multiple ControlNets are specified in init, you can set the
                corresponding scale as a list.
        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        # 0. Default height and width to unet
        height, width = self._default_height_width(height, width, hint)

        # !!! Skip check_inputs for efficiency, so must ensure the inputs correct
        # 1. Check inputs. Raise error if not correct
        # self.check_inputs(
        #     prompt,
        #     image,
        #     height,
        #     width,
        #     callback_steps,
        #     negative_prompt,
        #     prompt_embeds,
        #     negative_prompt_embeds,
        #     controlnet_conditioning_scale,
        # )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._device
        # device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        # print(guidance_scale)
        do_classifier_free_guidance = guidance_scale > 1.0

        # if isinstance(self.controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
        #     controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(self.controlnet.nets)

        # 4. Prepare image
        # if hint is not None and self.controlnet is not None:
        hint = self.prepare_hint(
            image=hint,
            width=width,
            height=height,
            batch_size=batch_size * num_images_per_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            dtype=self.controlnet.dtype,
            do_classifier_free_guidance=do_classifier_free_guidance,
            guess_mode=guess_mode,
            blind_control=True
        )
        if do_classifier_free_guidance:
            hint_uncond, hint_text = hint.chunk(2) # hint_uncond.shape=[1,3,256,256], value=0
        

        # Prepare hidden states for cross attention
        encoder_hidden_states = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds
        )
        '''
        if do_classifier_free_guidance:
            encoder_hidden_states_uncond, encoder_hidden_states = encoder_hidden_states.chunk(2)
            encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states], dim=0)
        '''

        clip_image_embeds = self._clip_encode_image(image, device, num_images_per_prompt, do_classifier_free_guidance, output_hidden_states=True)

        clip_image_embeds = self.image_proj_model(clip_image_embeds)
        clip_image_embeds = torch.cat([encoder_hidden_states, clip_image_embeds], dim=1)

        # if do_classifier_free_guidance:
            # negative_image = torch.zeros_like(image) if negative_image is None else negative_image
            # image = torch.cat([image, negative_image], dim=0)
        
        # Prepare rays
        if c2ws is not None:
            if c2ws.ndim == 2:
                c2ws = c2ws.unsqueeze(0)
            if c2ws.shape[0] == 1:
                c2ws = c2ws.repeat(batch_size, 1, 1)
            assert c2ws.size(0) == batch_size
            batch_size, num_views = c2ws.shape[:2]
            c2ws = c2ws.repeat_interleave(num_images_per_prompt, dim=0)
            c2ws = c2ws.to(device, dtype=self.camera_proj.dtype)
            if do_classifier_free_guidance and not guess_mode:
                c2ws = torch.cat([c2ws]*2, dim=0)

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps.to(device)

        # 6. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
 
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_views,
            num_channels_latents,
            height,
            width,
            encoder_hidden_states.dtype,
            device,
            generator,
            start_from_same_latent,
            latents
        )

        # prompt_embeds_unet = prompt_embeds.repeat_interleave(inner_B, dim=0) if use_inner_batch else prompt_embeds

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                # do_classifier_free_guidance=True, latent_model_input.shape=[2,4,4,32,32], latents.shape=[1,4,4,32,32]
                
                if blind_control_until_step is not None and i < blind_control_until_step: # None
                    hint = torch.cat([hint_uncond, hint_text], dim=0)
                else:
                    if do_classifier_free_guidance:
                        hint = torch.cat([hint_text, hint_text], dim=0)
                
                noise_pred = self._forward( # noise_pred.shape=[2,4,4,32,32]
                    latent_model_input=latent_model_input,
                    controlnet_cond=hint, # hint.shape=[2,3,256,256], value=[0,1]
                    t=t, # tensor(981, device='cuda:0')
                    encoder_hidden_states=encoder_hidden_states, #encoder_hidden_states, # shape=[2,77,1024]
                    ip_adapter_embeds=clip_image_embeds,
                    controlnet_conditioning_scale=controlnet_conditioning_scale, # 1.0
                    guess_mode=guess_mode, # False
                    do_classifier_free_guidance=do_classifier_free_guidance, # True
                    c2ws=c2ws, # c2ws.shape=[2,4,16]
                    cross_attention_kwargs=cross_attention_kwargs, # None
                )

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2) # noise_pred_uncond, noise_pred_text shape=[1,4,4,32,32]
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # if use_inner_batch:
                #     noise_pred = rearrange(noise_pred, '(b v) c h w -> b v c h w', v=inner_B)
                
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        latents = rearrange(latents, 'b n c h w -> (b n) c h w')
        # If we do sequential model offloading, let's offload unet and controlnet
        # manually for max memory savings
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.unet.to("cpu")
            self.controlnet.to("cpu")
            torch.cuda.empty_cache()

        if output_type == "latent":
            image = latents
            # has_nsfw_concept = None
        elif output_type == "pil" or output_type == "numpy":
            # 8. Post-processing
            image = self.decode_latents(latents)

            # 9. Run safety checker
            # image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = self.decode_latents(latents, post_procs=False)

            image = rearrange(image, '(b n) c h w -> b n c h w', n=num_views)
        
        if output_type == "pil":
            image = self.numpy_to_pil(image)

            # 9. Run safety checker
            # image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            # return (image, has_nsfw_concept)
            return image

        return MVControlPipelineOutput(images=image)

    
    def _forward(
        self, 
        latent_model_input: Float[Tensor, "B N C H W"],
        controlnet_cond: Float[Tensor, "B C H W"],
        t: Union[torch.LongTensor, int],
        encoder_hidden_states: Float[Tensor, "B L F"],
        ip_adapter_embeds,
        controlnet_conditioning_scale: float = 1.0,
        guess_mode: bool = False,
        do_classifier_free_guidance: bool = False,
        c2ws: Optional[Float[Tensor, "B N 16"]] = None,
        cross_attention_kwargs = None,
        forward_2d: bool = False,
    ):  
        batch_size, num_views = latent_model_input.shape[:2]
        if c2ws is not None:
            cam_mtx_flattened = rearrange(c2ws, 'b n l -> (b n) l') # shape=[2,4,16]->[8,16]
            cam_mtx_emb = self.camera_proj(cam_mtx_flattened) # [8,16]->[8,1280]
        else:
            cam_mtx_emb = None

        if cam_mtx_emb is not None:
            cam_mtx_emb = cam_mtx_emb.to(dtype=self.unet.dtype)

        if (
            controlnet_cond is not None 
            and self.controlnet is not None
        ):
            # controlnet(s) inference
            if guess_mode and do_classifier_free_guidance:
                # Infer controlnet only for condition batch
                controlnet_prompt_embeds = encoder_hidden_states.chunk(2)[1]
                controlnet_sample = latent_model_input.chunk(2)[1]
            else:
                controlnet_prompt_embeds = encoder_hidden_states # controlnet_prompt_embeds.shape=[2,77,1024]
                controlnet_sample = latent_model_input # controlnet_sample.shape=[2,4,4,32,32]
                
            down_block_res_samples, mid_block_res_sample, temb_residual = self.controlnet( # controlnet.dtype=torch.float16
                sample=controlnet_sample.to(self.controlnet.dtype), # controlnet_sample.shape=[2,4,4,32,32]
                controlnet_cond=controlnet_cond.to(self.controlnet.dtype), # controlnet_cond.shape=[2,3,256,256], value_range=[0,1], 2张相同的normal图，即为输入pipe的hint
                timestep=t, # tensor(981, device='cuda:0')
                encoder_hidden_states=controlnet_prompt_embeds.to(self.controlnet.dtype), # controlnet_prompt_embeds.shape=[2,77,1024]
                camera_matrix_embeds=cam_mtx_emb, # cam_mtx_emb.shape=[8,1280]
                conditioning_scale=controlnet_conditioning_scale, # 1.0
                return_dict=False,
                forward_2d=forward_2d # False
            )
            cam_mtx_emb = temb_residual
            ''' # len(down_block_res_samples)=12, every element is a torch.Tensor
            torch.Size([8, 320, 32, 32])
            torch.Size([8, 320, 32, 32])
            torch.Size([8, 320, 32, 32])
            torch.Size([8, 320, 16, 16])
            torch.Size([8, 640, 16, 16])
            torch.Size([8, 640, 16, 16])
            torch.Size([8, 640, 8, 8])
            torch.Size([8, 1280, 8, 8])
            torch.Size([8, 1280, 8, 8])
            torch.Size([8, 1280, 4, 4])
            torch.Size([8, 1280, 4, 4])
            torch.Size([8, 1280, 4, 4])
            '''
            ''' # len(mid_block_res_sample)=8, every element is a torch.Tensor
            torch.Size([1280, 4, 4])
            torch.Size([1280, 4, 4])
            torch.Size([1280, 4, 4])
            torch.Size([1280, 4, 4])
            torch.Size([1280, 4, 4])
            torch.Size([1280, 4, 4])
            torch.Size([1280, 4, 4])
            torch.Size([1280, 4, 4])
            '''

            if guess_mode and do_classifier_free_guidance:
                # Infered ControlNet only for the conditional batch.
                # To apply the output of ControlNet to both the unconditional and conditional batches,
                # add 0 to the unconditional batch to keep it unchanged.
                down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])
                cam_mtx_emb = torch.cat([cam_mtx_emb] * 2)
        else:
            down_block_res_samples = None
            mid_block_res_sample = None


        # Prepare the shapes of variables for inner batch
        latent_model_input = rearrange(latent_model_input, 'b n c h w -> (b n) c h w')
        t = t.repeat_interleave(num_views) if t.ndim == 1 else t
        #encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_views, dim=0)
        ip_adapter_embeds = ip_adapter_embeds.repeat_interleave(num_views, dim=0)

        noise_pred = self.unet( # noise_pred.shape=[8,4,32,32], unet.dtype=torch.float16
            latent_model_input.to(dtype=self.unet.dtype),
            t,
            encoder_hidden_states=ip_adapter_embeds.to(dtype=self.unet.dtype),
            camera_matrix_embeds=cam_mtx_emb,
            cross_attention_kwargs=cross_attention_kwargs,
            down_block_additional_residuals=down_block_res_samples, # down_block_res_samples[i].dtype=torch.float16
            mid_block_additional_residual=mid_block_res_sample, # mid_block_res_sample[i].dtype=torch.float16
        ).sample

        noise_pred = rearrange(noise_pred, '(b n) c h w -> b n c h w', n=num_views)
        del down_block_res_samples
        
        return noise_pred

from extern.mvcontrol.attention import set_self_attn_processor, XFormersCrossViewAttnProcessor, CrossViewAttnProcessor

def load_mvcontrol_custom_pipeline(
    pretrained_model_name_or_path,
    pretrained_controlnet_name_or_path,
    weights_dtype,
    num_views,
    device,
):

    controlnet = MultiViewControlNetModel.from_pretrained(
        pretrained_controlnet_name_or_path, torch_dtype=weights_dtype
    ).to(device)

    pipe = MVControlPipeline.from_pretrained(
        pretrained_model_name_or_path,
        controlnet=controlnet,
        torch_dtype=weights_dtype,
        device=device
    ).to(device)
    
    if is_xformers_available():
        import xformers
        attn_procs_cls = XFormersCrossViewAttnProcessor
    else:
        attn_procs_cls = CrossViewAttnProcessor
    
    set_self_attn_processor(
        pipe.unet, attn_procs_cls(num_views=num_views)
    )

    set_self_attn_processor(
        pipe.controlnet, attn_procs_cls(num_views=num_views)
    )
    
    return pipe