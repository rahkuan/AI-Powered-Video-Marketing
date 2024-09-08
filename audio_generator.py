# audio_generator.py

import os
import torch
import random
import numpy as np
import librosa
import soundfile as sf
from TTS.api import TTS
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip, concatenate_videoclips
import moviepy.video.fx.all as vfx
from pydub import AudioSegment
from diffusers import StableDiffusionPipeline
from transformers import pipeline
import scipy


# Directory paths
voiceover_dir = "generated_voiceovers"
music_dir = "generated_music"
output_dir = "generated_videos"

# Load the TTS model
tts_model_name = "tts_models/en/ljspeech/tacotron2-DDC"  # Replace with your desired TTS model
tts = TTS(tts_model_name)

# Load the music generation pipeline
music_model_name = "facebook/musicgen-small"
music_pipe = pipeline("text-to-audio", model=music_model_name)

def generate_music(style="classical", duration=10, output_filename=None):
    """
    Generates AI music based on the chosen style.

    Args:
        style (str): The style of music to generate (e.g., 'ambient', 'jazz', 'classical').
        duration (int): Duration of the music in seconds.
        output_filename (str, optional): The name of the output file.

    Returns:
        str: The path to the generated music file.
    """
    if music_pipe is None:
        return "musicgen model could not be loaded. Ensure the model ID is correct and dependencies are installed."

    # Ensure output directory exists
    os.makedirs(music_dir, exist_ok=True)

    try:
        # Generate music prompt based on the desired style
        text = f"A {style} music track, instrumental, suitable for background ambiance."
        
        forward_params = {"max_new_tokens": 512} # extend audio to longer time

        # Generate the music with musicgen model
        output = music_pipe(text, forward_params=forward_params)

        # Save the generated audio to the output file
        output_filename = output_filename or f"music_{style}_{random.randint(1000, 9999)}.wav"
        output_path = os.path.join(music_dir, output_filename)

        # Export the audio
        scipy.io.wavfile.write(output_path, rate=output["sampling_rate"], data=output["audio"])
        print(f"Music generated successfully: {output_path}")
        return output_path

    except Exception as e:
        error_message = f"Error generating music: {e}"
        print(error_message)
        return error_message

def generate_voiceover(text, output_filename=None):
    """
    Generates a voiceover from the provided text using AI TTS.

    Args:
        text (str): The input text for the voiceover.
        output_filename (str, optional): The name of the output file.

    Returns:
        str: The path to the generated voiceover file.
    """
    # Ensure output directory exists
    os.makedirs(voiceover_dir, exist_ok=True)

    output_filename = output_filename or f"voiceover_{random.randint(1000, 9999)}.wav"
    output_path = os.path.join(voiceover_dir, output_filename)

    try:
        # Generate voiceover without specifying speaker for single-speaker models
        tts.tts_to_file(text=text, file_path=output_path)
        
        print(f"Voiceover generated: {output_path}")
        return output_path

    except ValueError as e:
        print(f"Error: {e}. The model does not support the speaker parameter. Please check the model's capabilities.")
        return None

def enhance_video_with_audio(video_path, voiceover_path=None, music_path=None, voiceover_gain=-10, music_gain=5):
    """
    Enhances the video by adding voiceover and background music with adjusted volume levels.

    Args:
        video_path (str): The path to the input video file.
        voiceover_path (str, optional): Path to the voiceover file.
        music_path (str, optional): Path to the music file.
        voiceover_gain (float): Gain adjustment for the voiceover (negative value to reduce volume).
        music_gain (float): Gain adjustment for the background music (positive value to increase volume).

    Returns:
        str: The path to the enhanced video.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load video
    try:
        video = VideoFileClip(video_path)
        if video.duration == 0:
            print("Error: Video file is empty or corrupted.")
            return None
    except Exception as e:
        print(f"Error loading video file: {e}")
        return None

    original_video_duration = video.duration

    # Load audio files if provided
    voiceover = AudioFileClip(voiceover_path) if voiceover_path else None
    music = AudioFileClip(music_path) if music_path else None

    # Determine the target duration based on the voiceover audio if available, else fallback to longest audio clip
    target_duration = max(
        voiceover.duration if voiceover else 0,
        music.duration if music else 0,
        original_video_duration,
    )

    # Adjust the video duration to match the target duration by repeating it if necessary
    if target_duration > original_video_duration:
        repeat_count = int(target_duration // original_video_duration) + 1
        repeated_clips = [video] * repeat_count
        extended_video = concatenate_videoclips(repeated_clips)
        video = extended_video.subclip(0, target_duration)
    else:
        video = video.subclip(0, target_duration)

    # Adjust audio clips to match the new video duration
    def adjust_audio_length(audio_clip, target_duration):
        if audio_clip.duration > target_duration:
            return audio_clip.subclip(0, target_duration)
        elif audio_clip.duration < target_duration:
            return audio_clip.fx(vfx.loop, duration=target_duration)
        else:
            return audio_clip

    if voiceover:
        voiceover = adjust_audio_length(voiceover, target_duration).volumex(10**(voiceover_gain / 20))  # Adjust voiceover volume
    if music:
        music = adjust_audio_length(music, target_duration).volumex(10**(music_gain / 20))  # Adjust music volume

    # Combine audio
    combined_audio_clips = []
    if voiceover:
        combined_audio_clips.append(voiceover)
    if music:
        combined_audio_clips.append(music)

    # Combine all audio clips into a composite audio
    composite_audio = CompositeAudioClip(combined_audio_clips) if combined_audio_clips else None

    # Apply the composite audio to the video
    final_video = video.set_audio(composite_audio) if composite_audio else video

    try:
        # Save the enhanced video
        output_video_path = os.path.join(output_dir, f"enhanced_{os.path.basename(video_path)}")
        final_video.write_videofile(output_video_path, codec='libx264', audio_codec='aac')
        print(f"Enhanced video saved at: {output_video_path}")
        return output_video_path
    except Exception as e:
        print(f"Error saving video file: {e}")
        return None

def convert_gif_to_mp4(gif_path, output_path=None):
    """
    Converts a GIF file to an MP4 file.

    Args:
        gif_path (str): The path to the input GIF file.
        output_path (str, optional): The path to the output MP4 file. If not provided, it saves with the same name.

    Returns:
        str: The path to the converted MP4 file.
    """
    # Set default output path if not provided
    if output_path is None:
        output_path = gif_path.replace('.gif', '.mp4')

    try:
        # Load the GIF file
        clip = VideoFileClip(gif_path)

        # Write the video file in MP4 format using the H.264 codec
        clip.write_videofile(output_path, codec='libx264', audio=False)

        print(f"Successfully converted {gif_path} to {output_path}")
        return output_path

    except Exception as e:
        print(f"Error converting {gif_path} to MP4: {e}")
        return None

if __name__ == '__main__':
    # Generate a voiceover
    #voiceover_path = generate_voiceover("With every tick, experience the harmony of masterful engineering and exquisite design. A watch that doesn’t just keep time but elevates every moment.")
    
    # Generate background music
    #music_path = generate_music(style="classical", duration=10)
    
    # Convert gif to mp4
    #input_gif_path = './generated_videos/output-upscaled-with-prompt.gif'
    #output_mp4_path = convert_gif_to_mp4(input_gif_path)

    # Enhance a sample video with the generated voiceover and music
    #sample_video_path = output_mp4_path # Replace with the path to your input video
    #enhanced_video_path = enhance_video_with_audio(sample_video_path, voiceover_path, music_path, voiceover_gain=0, music_gain=2)
    #print(f"Enhanced video created: {enhanced_video_path}")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video", type=str, help="path to motion video")
    parser.add_argument("--voiceover", type=str, help="content of ad voice to enhance video")
    parser.add_argument("--music_type", type=str, help="type of music to generate background music to add along with voiceover")
    parser.add_argument("--save_dir", type=str, default='generated_videos', help="ouput would be a .mp4 video saving in this folder")
    args = parser.parse_args()
    voiceover_path = generate_voiceover(args.voiceover)
    music_path = generate_music(style=args.music_type, duration=10)
    output_mp4_path = convert_gif_to_mp4(args.input_video)
    enhanced_video_path = enhance_video_with_audio(output_mp4_path, voiceover_path, music_path, voiceover_gain=0, music_gain=2)
    print(f"Enhanced video created: {enhanced_video_path}")
