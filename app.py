import streamlit as st
import openai
import moviepy.editor as mp
import io
import re
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Set up OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Streamlit app title
st.title("Viral Video Generator with AI")

# User input for video idea
video_idea = st.text_input("Enter your video idea")

if video_idea:
    # Generate viral video script using OpenAI Completions API
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Generate a viral video script about {video_idea}",
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.7,
    )
    script = response.choices[0].text

    # Generate image prompts from the script using OpenAI GPT-3.5 Turbo
    messages = [{"role": "system", "content": "You are an AI assistant that generates image prompts for a viral video based on a given script."},
                {"role": "user", "content": f"Here is the script: {script}. Please generate image prompts for this viral video."}]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=200,
        n=1,
        stop=None,
        temperature=0.7,
    )
    image_prompts = response.choices[0].message.content.split("\n")

    # Generate images using OpenAI DALL-E API
    images = []
    for prompt in image_prompts:
        response = openai.Image.create(
            prompt=prompt,
            n=1,
            size="1024x1024"
        )
        image_url = response["data"][0]["url"]
        images.append(image_url)

    # Display script and images
    st.subheader("Viral Video Script")
    st.write(script)

    st.subheader("Images for the Video")
    for image_url in images:
        st.image(image_url)

    # Generate video from images
    clips = [mp.ImageClip(mp.utils.gif_tools.url_to_gif(image_url)).set_duration(2) for image_url in images]
    final_clip = mp.concatenate_videoclips(clips)

    # Save video to buffer
    video_buffer = io.BytesIO()
    final_clip.write_videofile(video_buffer, codec="libx264", audio=False)
    video_bytes = video_buffer.getvalue()

    # Display video
    st.video(video_bytes)
