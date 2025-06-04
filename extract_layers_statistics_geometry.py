""" 
This script extracts and records the adain features of generated images using style aligned.
The output of this script can be used woth "grade_layers.py" to grade the layers for style balance.
"""
import os
import argparse
from PIL import Image
import torch

from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL

from handlers import adain_handler_sa
from models.sdxl_controlnet_stylealigned_analysis import StableDiffusionXLControlNetStyleAlignedPipeline

from utils import prepare_canny_image

THREE_CATS_CANNY_VALUES = (50, 100)
SERIES_NAME = "style_analysis_geometry_cat"
SUBJECT = "cat"

def main(args):
    input_dir = f'analysis_files/geometry/{args.subject}'
    output_dir = f'outputs/statistics/{args.series_name}'
    os.makedirs(output_dir, exist_ok=True)

    output_image_dir = os.path.join(output_dir, 'images')
    output_canny_dir = os.path.join(output_dir, 'cannies')
    output_style_dir = os.path.join(output_dir, 'styles')
    output_stats_dir = os.path.join(output_dir, 'stats')

    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_canny_dir, exist_ok=True)
    os.makedirs(output_style_dir, exist_ok=True)
    os.makedirs(output_stats_dir, exist_ok=True)

    #### Load DDPM'S ###########################################################
    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0",
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16
    ).to("cuda")

    # load both base & refiner
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to("cuda")

    pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        vae=vae,
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16,
        # scheduler=scheduler
    ).to("cuda")


    sa_args = adain_handler_sa.StyleAlignedArgs(share_group_norm=False,
                                        share_layer_norm=False,
                                        share_attention=True,
                                        adain_queries=True,
                                        adain_keys=True,
                                        adain_values=False,
                                        )



    model = StableDiffusionXLControlNetStyleAlignedPipeline(pipeline, sa_args)

    #### Prepare Inputs ###########################################################


    layer_names_file = os.path.join(output_dir, "layer_names")

    style_prompts = 'in line art style'
    reference_prompts = 'An ink painting'
    main_prompts = 'A photo of a sitting cat'

    target_prompt = f"{main_prompts}, {style_prompts}"
    reference_prompt = f"{reference_prompts}, {style_prompts}"
    prompts = [reference_prompt, target_prompt]

    files = os.listdir(input_dir)

    style_dict = {artist: sorted([f_name for f_name in files if artist in f_name]) for artist in set([f_name.split("_")[0] for f_name in files])}
    num_styles = len(style_dict.keys())
    num_per_style = len(list(style_dict.values())[0])


    for i, k in enumerate(style_dict.keys()):
        assert len(style_dict[k]) == num_per_style
        for j, file in enumerate(style_dict[k]):
            torch.manual_seed(10)

            img = Image.open(os.path.join(input_dir, file)).convert("RGB")
            canny_map = prepare_canny_image(img, THREE_CATS_CANNY_VALUES)

            images, stats, layer_names = model.call(prompts,
                                                    image=canny_map,
                                                    num_inference_steps=50,
                                                    num_images_per_prompt=1,
                                                    controlnet_conditioning_scale=0.8,
                                                    guidance_scale=5
                                                    )
            
            # save images and stats
            images[0].save(os.path.join(output_style_dir, f"style_{k}.png"))
            images[1].save(os.path.join(output_image_dir, f"gen_{k}_{j}.png"))
            canny_map.save(os.path.join(output_canny_dir, f"canny_{k}_{j}.png"))
            torch.save(stats, os.path.join(output_stats_dir, f"stats_{i}_{j}"))

            if not os.path.exists(layer_names_file):
                torch.save(layer_names, layer_names_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--series_name", type=str, default=SERIES_NAME)
    parser.add_argument("--subject", type=str, default=SUBJECT)
    args = parser.parse_args()
    main(args)