# generate_style_image.py

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os

# Load the Stable Diffusion model
model_id = "CompVis/stable-diffusion-v1-4"

# Ensure the pipeline runs on GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    """
    Loads the Stable Diffusion model pipeline.
    Returns the loaded pipeline.
    """
    pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    pipeline = pipeline.to(device)
    return pipeline

# Load the model only once to improve performance
pipeline = load_model()

def generate_image(prompt, output_dir="generated_images", filename="style_image.png"):
    """
    Generates an image based on the provided text prompt using Stable Diffusion.

    Args:
        prompt (str): The text prompt to generate the style image.
        output_dir (str): Directory to save the generated images.
        filename (str): The name of the output image file.

    Returns:
        str: The path to the generated image.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Generate the image
    with torch.no_grad():
        image = pipeline(prompt).images[0]

    # Save the image
    output_path = os.path.join(output_dir, filename)
    image.save(output_path)
    return output_path

if __name__ == "__main__":
    # Example usage
    #prompt = "A futuristic cityscape in the style of cyberpunk"
    #output_path = generate_image(prompt)
    #print(f"Image generated and saved at: {output_path}")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, help="text description of style/scene for the product")
    parser.add_argument("--save_dir", type=str, default="generated_images", help="directory to save generated style image")
    parser.add_argument("--filename", type=str, default="style_image.png", help="filename of generated style image")
    args = parser.parse_args()

    output_path = generate_image(args.prompt, args.save_dir, args.filename)
    print(f"Style Image generated and saved at: {output_path}")
