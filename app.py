import streamlit as st
from openai import OpenAI
import moviepy.editor as mp
import numpy as np
import io
from dotenv import load_dotenv
import os
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

# Set up OpenAI API key from environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Streamlit app title
st.title("Viral Video Generator with AI")

# User input for video idea
video_idea = st.text_input("Enter your video idea")

if video_idea:
    # Generate viral video script using OpenAI Chat Completions API
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

    # Generate audio for the script using OpenAI Audio API
    audio_response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=script
    )

    # Save the audio file
    audio_file = Path("script_audio.mp3")
    with open(audio_file, "wb") as f:
        f.write(audio_response.content)

    # Load the audio file using moviepy
    audio_clip = mp.AudioFileClip(str(audio_file))

    # Generate a blank video clip with the same duration as the audio
    def make_frame(t):
        return np.zeros((720, 1280, 3), dtype=np.uint8)  # Replace with desired resolution and color

    video_clip = mp.VideoClip(make_frame=make_frame, duration=audio_clip.duration)

    # Combine the audio and video clips
    final_clip = video_clip.set_audio(audio_clip)

    # Write the final video clip to a file
    final_clip.write_videofile("viral_video.mp4")

    # Display the video in Streamlit
    st.video("viral_video.mp4")
