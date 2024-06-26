import os
import streamlit as st
import openai
from pathlib import Path
import moviepy.editor as mp
from urllib.request import urlopen
from io import BytesIO
from PIL import Image
import replicate

def generate_video(prompts, num_frames=1200, enhance=True, image_guidance=3.0, model_type="text-to-video"):
    video_urls = []
    if model_type == "text-to-video":
        for prompt in prompts:
            input = {
                "prompt": prompt,
                "enhance": enhance,
                "num_frames": num_frames // len(prompts),
                "image_guidance": image_guidance
            }

            output = replicate.run(
                "camenduru/streaming-t2v:1fe245aad4bb7f209074a231142ac3eceb3b1f2adc9cf77b46e8ffa2662323cf",
                input=input
            )

            video_urls.extend(output)
    elif model_type == "text-to-image":
        for prompt in prompts:
            output = replicate.run(
                "bytedance/sdxl-lightning-4step:5f24084160c9089501c1b3545d9be3c27883ae2239b6f412990e82d4a6210f8f",
                input={"prompt": prompt}
            )
            video_urls.append(output[0])

    return video_urls

def main():
    st.title("Viral Video Generator")

    # Get user input for OpenAI API key
    openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")

    # Check if the user has provided an API key
    if not openai_api_key:
        st.error("Please enter your OpenAI API key.")
        return

    # Set up OpenAI API credentials
    openai.api_key = openai_api_key
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

    # Define available languages and their corresponding voice IDs
    languages = {
        "English": "onyx",
        "German": "onyx",
        "French": "onyx",
        "Spanish": "onyx",
        "Italian": "onyx"
    }

    # Get user input for the video type
    video_type = st.selectbox("Select Video Type", ["Story", "Informational", "Promotional", "Educational"])

    # Get user input for the video idea
    video_idea = st.text_input(f"Enter your {video_type.lower()} video idea:")

    # Select video style
    selected_style = st.selectbox("Choose a Video Style", list(video_styles.keys()))
    style_description = video_styles[selected_style]

    # Select language
    selected_language = st.selectbox("Choose a Language", list(languages.keys()))
    voice_id = languages[selected_language]

    # Select Replicate model
    model_type = st.selectbox("Select Replicate Model", ["Text-to-Video", "Text-to-Image"])

    if st.button("Generate Viral Video"):
        # Add a placeholder for the progress bar
        progress_bar = st.progress(0)

        # Generate content based on the selected video type
        with st.spinner(f'Generating {video_type.lower()} content...'):
            if video_type == "Story":
                messages = [
                    {"role": "system", "content": f"You are an AI assistant that crafts amazing and captivating stories for spoken word performances in {selected_language}. Your stories should be rich in imagery, emotion, and depth, with relatable characters and a compelling arc that can deeply engage an audience."},
                    {"role": "user", "content": f"Create an amazing story that touches the heart, with vivid descriptions and a powerful message, about {video_idea}"}
                ]
            elif video_type == "Informational":
                messages = [
                    {"role": "system", "content": f"You are an AI assistant that creates informative and educational content for video presentations in {selected_language}. Your content should be factual, well-researched, and presented in a clear and engaging manner."},
                    {"role": "user", "content": f"Create an informative and engaging video script that provides detailed information about {video_idea}"}
                ]
            elif video_type == "Promotional":
                messages = [
                    {"role": "system", "content": f"You are an AI assistant that generates persuasive and compelling promotional content for video marketing in {selected_language}. Your content should highlight the key features and benefits of the product or service, while also creating a sense of excitement and desire in the viewer."},
                    {"role": "user", "content": f"Create a promotional video script that effectively markets and promotes {video_idea}"}
                ]
            else:  # Educational
                messages = [
                    {"role": "system", "content": f"You are an AI assistant that develops educational and instructional content for video tutorials in {selected_language}. Your content should be easy to understand, well-structured, and designed to effectively teach and explain complex concepts or processes."},
                    {"role": "user", "content": f"Create an educational video script that teaches and explains {video_idea} in a clear and engaging manner"}
                ]

            response = client.chat.completions.create(
                model="gpt-4o",
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
                voice=voice_id,
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
                model="gpt-4o",
                messages=messages,
                max_tokens=500,
                n=1,
                stop=None,
                temperature=0.7
            )
            image_prompts = [prompt.strip() for prompt in response.choices[0].message.content.split("\n") if prompt.strip()]

        # Update the progress bar
        progress_bar.progress(75)

        # Generate video using the selected Replicate model
        with st.spinner('Generating video...'):
            video_urls = generate_video(image_prompts, num_frames=1200, enhance=True, image_guidance=3.0, model_type=model_type.lower())

        # Update the progress bar to 100%
        progress_bar.progress(100)

        if model_type == "Text-to-Video":
            clips = []
            for url in video_urls:
                video_data = urlopen(url).read()
                with BytesIO(video_data) as video_buffer:
                    clip = mp.VideoFileClip(video_buffer)
                    clips.append(clip)

            final_clip = mp.concatenate_videoclips(clips)
            final_clip = final_clip.set_audio(mp.AudioFileClip(str(audio_file)))
            final_clip.fps = 24

            video_file = "viral_video.mp4"
            final_clip.write_videofile(video_file, codec="libx264")

            with open(video_file, "rb") as f:
                video_bytes = f.read()

            st.video(video_bytes)
        else:  # Text-to-Image
            for url in video_urls:
                image_data = urlopen(url).read()
                image = Image.open(BytesIO(image_data))
                st.image(image, caption="Generated Image", use_column_width=True)

        # Display the script and audio
        st.header(f"{video_type} Video Script ({selected_language})")
        st.write(script)

        st.header("Audio")
        with audio_file.open("rb") as f:
            audio_bytes = f.read()
        st.audio(audio_bytes, format="audio/mp3")

if __name__ == "__main__":
    main()


