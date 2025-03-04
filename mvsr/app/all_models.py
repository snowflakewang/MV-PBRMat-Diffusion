import torch
from mvsr.scripts.sd_model_zoo import load_common_sd15_pipe
from diffusers import StableDiffusionControlNetImg2ImgPipeline, StableDiffusionPipeline


class MyModelZoo:
    _pipe_disney_controlnet_lineart_ipadapter_i2i: StableDiffusionControlNetImg2ImgPipeline = None
    
    #base_model = "runwayml/stable-diffusion-v1-5"
    base_model = "/cpfs01/user/wangyitong/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9"

    def __init__(self, base_model=None) -> None:
        if base_model is not None:
            self.base_model = base_model

    @property
    def pipe_disney_controlnet_tile_ipadapter_i2i(self):
        return self._pipe_disney_controlnet_lineart_ipadapter_i2i
    
    def init_models(self):
        self._pipe_disney_controlnet_lineart_ipadapter_i2i = load_common_sd15_pipe(base_model=self.base_model, ip_adapter=True, plus_model=False, controlnet="mvsr/ckpt/controlnet-tile", pipeline_class=StableDiffusionControlNetImg2ImgPipeline)

model_zoo = MyModelZoo()
