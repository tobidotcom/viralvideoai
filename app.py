import streamlit as st
import openai
from pathlib import Path
import moviepy.editor as mp
from urllib.request import urlopen
from io import BytesIO
from PIL import Image
import numpy as np
import replicate
import os

# Set up OpenAI API credentials
openai.api_key = st.secrets["OPENAI_API_KEY"]
client = openai.Client()

# Set the Replicate API token
os.environ["REPLICATE_API_TOKEN"] = st.secrets["REPLICATE_API_TOKEN"]

# Define video styles
video_styles = {
    "Realistic": "photorealistic, highly detailed, natural lighting, true-to-life textures",
    "Cartoon": "cartoon style, vibrant and saturated colors, exaggerated features, flat shading, bold outlines",
    "Vintage": "vintage aesthetic, warm sepia or desaturated tones, aged textures, film grain, nostalgic feel",
    "Minimalist": "minimalist design, flat colors, simple shapes, clean lines, negative space",
    "Cyberpunk": "cyberpunk aesthetic, neon lights, futuristic technology, gritty urban environments, high contrast",
    "Surreal": "surreal and dreamlike, distorted reality, impossible scenarios, melting objects, mind-bending visuals",
    "Steampunk": "steampunk style, Victorian-era inspired, brass and wood textures, intricate machinery, retro-futuristic",
    "Vaporwave": "vaporwave aesthetic, glitch art, retro color palettes, Japanese influences, nostalgic and futuristic"
}

def main():
    st.title("Viral Video Generator")

    # Get user input for the video idea
    video_idea = st.text_input("Enter your video idea:")

    # Select video style
    selected_style = st.selectbox("Choose a Video Style", list(video_styles.keys()))
    style_description = video_styles[selected_style]

    if st.button("Generate Viral Video"):
        # Add a placeholder for the progress bar
        progress_bar = st.progress(0)

        # Generate captivating spoken stories using OpenAI Chat Completions API
        with st.spinner('Weaving your tale...'):
            messages = [
                {"role": "system", "content": "You are an AI assistant that crafts amazing and captivating stories for spoken word performances. Your stories should be rich in imagery, emotion, and depth, with relatable characters and a compelling arc that can deeply engage an audience."},
                {"role": "user", "content": f"Create an amazing story that touches the heart, with vivid descriptions and a powerful message, about {video_idea}"}
            ]
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=500,
                n=1,
                stop=None,
                temperature=0.7
            )
            script = response.choices[0].message.content

        # Update the progress bar
        progress_bar.progress(25)

        # Generate audio for the script using OpenAI Audio API
        with st.spinner('Generating audio...'):
            audio_response = client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=script
            )

            # Save the audio file
            audio_file = Path("script_audio.mp3")
            with open(audio_file, "wb") as f:
                f.write(audio_response.content)

        # Update the progress bar
        progress_bar.progress(50)

        # Generate image prompts from the script using OpenAI Chat Completions API
        with st.spinner('Generating image prompts...'):
            messages = [
                {"role": "system", "content": "You are an AI assistant that generates high-quality, consistent image prompts for a viral video."},
                {"role": "user", "content": f"The video script is as follows: {script}. Based on this script, create 20 detailed image prompts that capture the essence of the video's narrative. Each prompt should be in the '{selected_style}' style, which is characterized by {style_description}. The image prompts should be cohesive and maintain a consistent visual theme throughout, ensuring that the final video has a unified and professional look."}
            ]
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=500,
                n=1,
                stop=None,
                temperature=0.7
            )
            image_prompts = [prompt.strip() for prompt in response.choices[0].message.content.split("\n") if prompt.strip()]

        # Update the progress bar
        progress_bar.progress(75)

        # Generate images using Replicate's SDXL-Lightning API
        with st.spinner('Generating images...'):
            images = []
            for prompt in image_prompts:
                output = replicate.run(
                    "bytedance/sdxl-lightning-4step:5f24084160c9089501c1b3545d9be3c27883ae2239b6f412990e82d4a6210f8f",
                    input={"prompt": prompt}
                )
                image_url = output[0]
                image_data = urlopen(image_url).read()
                image = Image.open(BytesIO(image_data))
                images.append(np.array(image))

        # Update the progress bar to 100%
        progress_bar.progress(100)

        # Create video from images and audio
        clips = [mp.ImageClip(image).set_duration(2) for image in images]
        final_clip = mp.concatenate_videoclips(clips)
        final_clip = final_clip.set_audio(mp.AudioFileClip(str(audio_file)))
        final_clip.fps = 24

        # Save the video to an MP4 file
        video_file = Path("viral_video.mp4")
        final_clip.write_videofile(str(video_file), codec="libx264")

        # Display the script, audio, and video
        st.header("Viral Video Script")
        st.write(script)

        st.header("Audio")
        with audio_file.open("rb") as f:
            audio_bytes = f.read()
        st.audio(audio_bytes, format="audio/mp3")

        st.header("Video")
        with video_file.open("rb") as f:
            video_bytes = f.read()
        st.video(video_bytes)

if __name__ == "__main__":
    main()
