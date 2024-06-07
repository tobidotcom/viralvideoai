import openai
import moviepy.editor as mp
import io
from dotenv import load_dotenv
import os
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

# Set up OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

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
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.7,
    )
    script = response.choices[0].message.content

    # Generate audio for the script using OpenAI Audio API
    audio = openai.Audio.create(
        model="tts-1",  # The text-to-speech model to use
        voice="alloy",  # The voice to use for the audio
        input=script,  # The text to convert to speech
        output_format="mp3"  # The output format for the audio file
    )

    # Save the audio file
    audio_file = Path("script_audio.mp3")
    with open(audio_file, "wb") as f:
        f.write(audio.data)

    # Generate image prompts from the script using OpenAI Chat Completions API
    messages = [
        {"role": "system", "content": "You are an AI assistant that generates image prompts for a viral video based on a given script."},
        {"role": "user", "content": f"Here is the script: {script}. Please generate image prompts for this viral video."}
    ]
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

    # Display script and audio
    st.subheader("Viral Video Script")
    st.write(script)
    st.audio(audio_file)

    st.subheader("Images for the Video")
    for image_url in images:
        st.image(image_url)

    # Generate video from images
    clips = [mp.ImageClip(mp.utils.gif_tools.url_to_gif(image_url)).set_duration(2) for image_url in images]
    final_clip = mp.concatenate_videoclips(clips)

    # Add audio to the video
    final_clip = final_clip.set_audio(mp.AudioFileClip(audio_file))

    # Save video to buffer
    video_buffer = io.BytesIO()
    final_clip.write_videofile(video_buffer, codec="libx264")
    video_bytes = video_buffer.getvalue()

    # Display video
    st.video(video_bytes)
