import streamlit as st
import openai
from pathlib import Path
import moviepy.editor as mp
from urllib.request import urlopen
from io import BytesIO
from PIL import Image
import numpy as np
import time
import pysrt
from moviepy.editor import TextClip, CompositeVideoClip
from moviepy.video.tools.subtitles import SubtitlesClip

# Set up OpenAI API credentials
openai.api_key = st.secrets["OPENAI_API_KEY"]
client = openai.Client()

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

        # Generate viral video script using OpenAI Chat Completions API
        with st.spinner('Generating script...'):
            messages = [
                {"role": "system", "content": "You are an AI assistant that generates viral video scripts."},
                {"role": "user", "content": f"Generate a viral video script about {video_idea}"}
            ]
            response = client.chat.completions.create(model="gpt-3.5-turbo",
                                                      messages=messages,
                                                      max_tokens=500,
                                                      n=1,
                                                      stop=None,
                                                      temperature=0.7)
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
                {"role": "system", "content": "You are an AI assistant that generates image prompts for a viral video based on a given script."},
                {"role": "user", "content": f"Here is the script: {script}. Please generate at least 5 image prompts for this viral video, each describing a specific scene or moment from the script, in the {style_description} style. For example, if the style is 'Vintage', the visuals should have a warm sepia tone, aged textures, and a nostalgic feel reminiscent of old films."}
            ]
            response = client.chat.completions.create(model="gpt-3.5-turbo",
                                                      messages=messages,
                                                      max_tokens=500,
                                                      n=1,
                                                      stop=None,
                                                      temperature=0.7)
            image_prompts = [prompt.strip() for prompt in response.choices[0].message.content.split("\n") if prompt.strip()]

        # Update the progress bar
        progress_bar.progress(75)

        # Generate images using OpenAI DALL-E API
        with st.spinner('Generating images...'):
            images = []
            for i, prompt in enumerate(image_prompts):
                # Introduce a delay to avoid rate limiting
                if i > 0 and i % 5 == 0:
                    time.sleep(60)  # Wait for 1 minute

                response = client.images.generate(prompt=prompt,
                                                  n=1,
                                                  size="1024x1024")
                image_url = response.data[0].url
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

        # Generate subtitles using pysrt and TextClip
        subs = pysrt.SubRipFile()
        start_time = 0
        for line in script.split('\n'):
            end_time = start_time + len(line.split()) / 10  # Adjust the duration based on the number of words
            start_time_str = pysrt.SubRipTime.coerce(start_time).to_string()
            end_time_str = pysrt.SubRipTime.coerce(end_time).to_string()
            subs.append(pysrt.SubRipItem(start_time_str, end_time_str, line))
            start_time = end_time

        # Load the Google Font
        font = "Roboto-Regular.ttf"  # Replace with the desired Google Font file name
        subtitles_clip = SubtitlesClip(subs, font)

        # Add subtitles to the video
        final_clip = CompositeVideoClip([final_clip, subtitles_clip])

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
