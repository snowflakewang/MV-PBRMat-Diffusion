#!/bin/bash
python inference.py --ctrlnet_seed=42 --mvdiff_seed=1234 \
    --controlnet_cond_mode="image" \
    --prompt="ctrlnet_ip/image_prompts/ironman_rgba.png" \
    --controlnet_normal_path="data/validate_renderedobjaverse/Human_Shape_120_610746/front.png" \
    --input_path="data/validate_renderedobjaverse/Human_Shape_120_610746" \
    --out_path="data/validate_renderedobjaverse_inference_out" \
    --do_mv_super_res