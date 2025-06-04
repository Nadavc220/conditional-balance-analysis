""" 
This script extracts and records the adain features of generated images using style aligned.
The output of this script can be used woth "grade_layers.py" to grade the layers for style balance.
"""
import os
import yaml
from PIL import Image
import torch
import argparse

from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL

from handlers import adain_handler_sa
from models.sdxl_controlnet_stylealigned_analysis import StableDiffusionXLControlNetStyleAlignedPipeline

SERIES_NAME = "style_analysis_texture_rabbit"
CONTENT_FILE = 'content_texture_rabbit.yaml'
STYLE_FILE = 'style_prompts_texture.yaml'

def main(args):

    #### Experiment init ###########################################################
    input_dir = "analysis_files"
    output_dir = f"outputs/statistics/image_series/{args.series_name}"
    os.makedirs(output_dir, exist_ok=True)

    output_image_dir = os.path.join(output_dir, 'images')
    output_style_dir = os.path.join(output_dir, 'styles')
    output_stats_dir = os.path.join(output_dir, 'stats')

    os.makedirs(output_image_dir, exist_ok=True)
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
    ).to("cuda")


    sa_args = adain_handler_sa.StyleAlignedArgs(share_group_norm=False,
                                        share_layer_norm=False,
                                        share_attention=True,
                                        adain_queries=True,
                                        adain_keys=True,
                                        adain_values=True,
                                        )



    model = StableDiffusionXLControlNetStyleAlignedPipeline(pipeline, sa_args)

    #### Prepare Inputs ###########################################################



    layer_names_file = os.path.join(output_dir, "layer_names")

    with open(os.path.join(input_dir, 'style_prompts', args.style_file)) as f:
        prompts = yaml.safe_load(f)
        style_prompts = prompts["style_prompts"]
        reference_prompts = prompts["reference_prompts"]

    with open(os.path.join(input_dir, 'contents', args.content_file)) as f:
        content_info = yaml.safe_load(f)
        canny_map = Image.open(content_info["canny_path"])
        main_prompts = content_info["main_prompt"]

    if isinstance(main_prompts, str):
        main_prompts = [main_prompts] * len(style_prompts)
    elif isinstance(style_prompts, str) and isinstance(reference_prompts, str):
        style_prompts = [style_prompts] * len(main_prompts)
        reference_prompts = [reference_prompts] * len(main_prompts)

    num_per_style = 5

    for i in range(len(style_prompts)):
        torch.manual_seed(10)
        target_prompt = f"{main_prompts[i]}, {style_prompts[i]}"
        reference_prompt = f"{reference_prompts[i]}, {style_prompts[i]}"
        prompts = [reference_prompt, target_prompt]

        
        for j in range(num_per_style):
            images, stats, layer_names = model.call(prompts,
                                                    image=canny_map,
                                                    num_inference_steps=50,
                                                    num_images_per_prompt=1,
                                                    controlnet_conditioning_scale=0.8,
                                                    guidance_scale=5
                                                    )
            
            # save images and stats
            images[0].save(os.path.join(output_style_dir, f"style_{i}.png"))
            images[1].save(os.path.join(output_image_dir, f"{i}_{j}.png"))
            torch.save(stats, os.path.join(output_stats_dir, f"stats_{i}_{j}"))

            if not os.path.exists(layer_names_file):
                torch.save(layer_names, layer_names_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--series_name", type=str, default=SERIES_NAME)
    parser.add_argument("--content_file", type=str, default=CONTENT_FILE)
    parser.add_argument("--style_file", type=str, default=STYLE_FILE)
    args = parser.parse_args()
    main(args)
