import torch

from diffusers import AnimateDiffSparseControlNetPipeline
from diffusers.models import AutoencoderKL, MotionAdapter, SparseControlNetModel
from diffusers.schedulers import DPMSolverMultistepScheduler
from diffusers.utils import export_to_gif, load_image
import os


model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
motion_adapter_id = "guoyww/animatediff-motion-adapter-v1-5-3"
controlnet_id = "guoyww/animatediff-sparsectrl-rgb"
lora_adapter_id = "guoyww/animatediff-motion-lora-v1-5-3"
vae_id = "stabilityai/sd-vae-ft-mse"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

motion_adapter = MotionAdapter.from_pretrained(motion_adapter_id, torch_dtype=torch_dtype).to(device)
controlnet = SparseControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch_dtype).to(device)
vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=torch_dtype).to(device)
scheduler = DPMSolverMultistepScheduler.from_pretrained(
    model_id,
    subfolder="scheduler",
    beta_schedule="linear",
    algorithm_type="dpmsolver++",
    use_karras_sigmas=True,
)
pipe = AnimateDiffSparseControlNetPipeline.from_pretrained(
    model_id,
    motion_adapter=motion_adapter,
    controlnet=controlnet,
    vae=vae,
    scheduler=scheduler,
    torch_dtype=torch_dtype,
).to(device)
pipe.load_lora_weights(lora_adapter_id, adapter_name="motion_lora")

def image_to_video(image=None, prompt='', save_dir='./'):
    if isinstance(image, list):
        image = image[:]
        image = [load_image(img_path) for img_path in image]
    else:
        image = load_image(image)

    video = pipe(
        prompt=prompt,
        negative_prompt="low quality, worst quality",
        num_inference_steps=25,
        conditioning_frames=image,
        controlnet_frame_indices=[0],
        controlnet_conditioning_scale=1.0,
        generator=torch.Generator().manual_seed(42),
    ).frames[0]
    save_path = os.path.join(save_dir, 'output.gif')
    export_to_gif(video, save_path)
    print (f'Saved {save_path}')
    return save_path

if __name__ == "__main__":
    # upscaled image with image style and prompt
    # prompt = "A sleek, modern ad showcasing a closeup photo luxury watch in a futuristic cityscape"
    # image=['./upscaled_watch.jpg', './static/stock_images/style-3D.jpg']
    # image_to_video(image=image, prompt=prompt)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="image path of enhanced photo of product")
    parser.add_argument("--prompt", type=str, help="text prompt for AnimateDiff to create motion")
    parser.add_argument("--save_dir", type=str, default='generated_videos', help="save folder for .gif video")
    args = parser.parse_args()
    image = args.image
    prompt = args.prompt
    video_path = image_to_video(image=image, prompt=prompt)
    print (f'Video motion saved at: {video_path}')
