import streamlit as st
import openai
from pathlib import Path
import moviepy.editor as mp
from urllib.request import urlopen
from io import BytesIO
from PIL import Image
import numpy as np

# Set up OpenAI API credentials
openai.api_key = st.secrets["OPENAI_API_KEY"]
client = openai.Client()

def main():
    st.title("Viral Video Generator")

    # Get user input for the video idea
    video_idea = st.text_input("Enter your video idea:")

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
                {"role": "user", "content": f"Here is the script: {script}. Please generate at least 5 image prompts for this viral video, each describing a specific scene or moment from the script."}
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
            for prompt in image_prompts:
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

        # Set the fps attribute on the final VideoClip object
        final_clip.fps = 24

        # Save the video to an MP4 file
        final_clip.write_videofile("viral_video.mp4", codec="libx264")

        # Display the script, audio, and video
        st.header("Viral Video Script")
        st.write(script)

        st.header("Audio")
        st.audio(audio_file, format="audio/mp3")

        st.header("Video")
        video_file = open("viral_video.mp4", "rb").read()
        st.video(video_file)

if __name__ == "__main__":
    main()
