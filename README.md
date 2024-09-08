# AI-Powered Video Marketing Solution

## Description

This open-source repository contains the codebase, documentation, and tools required to build an AI-powered video marketing solution that utilizes cutting-edge technologies like Stable Diffusion and AnimateDiff. The solution enables users to generate, enhance, and customize videos with AI-driven voiceovers, music, and high-quality image enhancements, creating an engaging and professional marketing experience.

## Technologies Used

* Python
* TensorFlow/PyTorch
* OpenCV
* FFmpeg
* Stable Diffusion
* AnimateDiff

## Setup Instructions

### Clone the Repository

```console
git clone https://github.com/rahkuan/AI-Powered-Video-Marketing
cd AI-Powered-Video-Marketing
```

### Install Anaconda

Install Anaconda by following instruction in [Anaconda Install](https://docs.anaconda.com/anaconda/install/)

### Create a new Anaconda environment named aivideo with Python 3.9

```console
conda create -n aivideo python=3.9
conda activate aivideo
```

### Install Required Python Libraries

Install the necessary Python libraries within the created environment:

```console
# Install Flask and Flask-RESTful for app development
pip install flask flask-restful

# Install PyTorch and Torchvision
pip install torch==2.0.1 torchvision==0.15.2

# Install Real-ESRGAN for image enhancement and upscaling to higher quality
git clone https://github.com/xinntao/Real-ESRGAN.git
cd Real-ESRGAN

pip install basicsr
pip install facexlib
pip install gfpgan
pip install -r requirements.txt
python setup.py develop

pip install numpy==1.26.4
# Download pre-trained weight for image enhancement model
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P weights

cd ..

# Install Transformers for AI-driven music and VoiceOver generation
pip install transformers diffusers pillow
pip install accelerate
pip install peft

# Install additional libraries for audio and video processing
pip install soundfile moviepy coqui-tts pydub librosa

# For personalized content, use OpenAI GPT model to recommend voiceover and music type
pip install -qU langchain-openai
```

## Usage Guide

### Terminal Command

```console
# Enhance image and upscale to higher quality. The input image is a watch at size 750x750, output image size is 3000x3000
python enhance_image.py --image_path uploaded_image.png

# Generate style image with text
python generate_style_image.py --prompt "A futuristic cityscape in the style of cyberpunk"

# Generate motion video with text and from product image, gif file in generated_videos folder
python generate_motion.py --image ./upscaled_watch.jpg --prompt "A sleek, modern ad showcasing a closeup photo luxury watch in a futuristic cityscape"

# Enhance ad video with voiceover and background music
python audio_generator.py --input_video ./generated_videos/output-upscaled-with-prompt.gif --voiceover "With every tick, experience the harmony of masterful engineering and exquisite design. A watch that doesn’t just keep time but elevates every moment." --music_type classical

# Innovation in Video Marketing with personalized content, information of person (geographic, web surfing,...) that marketing ad target to is used by LLM (OpenAI GPT) model
# to recommmend voiceover content and music type, additionally, product description is also paly a role starting first with simple product name
# you also need OPENAI_API_KEY to use this model
# Enter 'export OPENAI_API_KEY=your-api-key-here' in terminal before you run below command
python personalized_content_generator.py --product_description "watch" --person_description "female, living in Asia, age 40"

# Output:
# personalized_voiceover_text
# "Attention all stylish women in Asia over the age of 40. Are you tired of constantly checking your phone for the time? Say goodbye to that habit with our elegant and functional watch. Made specifically for the # modern woman, our watch combines fashion and practicality, making it the perfect accessory for any occasion. Don't miss out on this must-have timepiece. Get yours today!"

# personalized_music_style
# Classical or jazz music

python personalized_content_generator.py --product_description "watch" --person_description "male, checking electronics at an ecommerce site"

# Output:
# personalized_voiceover_text
# "Introducing the perfect watch for the tech-savvy man. With its sleek design and advanced features, this watch is the ultimate accessory for checking electronics on your favorite ecommerce site. Stay connected # and stylish with our new watch. Order now."

# personalized_music_style
# I recommended using upbeat and modern genres such as electronic or rock music for a male checking electronics at an ecommerce site.
```

### Web application

An alternative way to interact with all component is through UI interaction on web app

```console
python app.py
```
Open web browser and go to `http://127.0.0.1:5000`

## Development Guidelines:

- Contributing to the repository:
  - Video edit is now simply repeating a clip, more effects in transition needed to make fluent shifts in video
  - Prompt engineering needed to conserve details from product photo after transformed into motion animation
  - AnimateDIff currently require both text and conditional image (product image) to animate, a modified pipeline to use only image as input without requirement for prompt would be flexible in solution that start only with a photo of product
- Codes tested in Colab GPU T4 (RAM 12GB, Storage 110GB, GPU-RAM 16GB), need diverse tests on cpu and determined OS
- Modifications to functionality is done seamlessly by replace model names because the flow of solution use pipeline and mostly take pre-trained models in HuggingFace. The proposed 3 steps (Step 1: Input Image Processing, Step 2: Video Generation, Step 3: Innovation in Video Marketing) are implemented in highly moduled manner in Python. Integration or variants of whole solution for automation is practically feasible.
  - Multiple variations of a marketing video tailored for different platforms (e.g., Instagram, YouTube, TikTok) require high-resolution of final video at diverse aspect ratios is still an open-ended direction for future developemnt.
