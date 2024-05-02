import streamlit as st
from PIL import Image
import numpy as np
import requests
import os

url = "http://localhost:9000/inference/"


def main():
    st.title("VAMDub")

    st.sidebar.header("Upload Files")
    audio_file = st.sidebar.file_uploader("Upload Audio File", type=["mp3", "wav"])
    image_file = st.sidebar.file_uploader(
        "Upload Image File", type=["jpg", "png", "jpeg"]
    )

    if audio_file is not None:
        st.audio(audio_file, format="audio/wav", start_time=0)
        with open(os.path.join("/tmp", audio_file.name), "wb") as fp:
            fp.write(audio_file.getbuffer())
        audio_file_op = ("audio.wav", open(os.path.join("/tmp", audio_file.name), "rb"))

    if image_file is not None:
        image_format = image_file.type.split("/")[1]
        image = Image.open(image_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        with open(os.path.join("/tmp", image_file.name), "wb") as fp:
            fp.write(image_file.getbuffer())
        image_file_op = (
            f"image.{image_format}",
            open(os.path.join("/tmp", image_file.name), "rb"),
        )

    if st.button("Submit") and audio_file is not None and image_file is not None:
        files = [("files", audio_file_op), ("files", image_file_op)]
        response = requests.post(url, files=files)
        # Check if the request was successful
        if response.status_code == 200:
            # Save the received audio file
            with open("received_audio.wav", "wb") as f:
                f.write(response.content)
            print("Audio file received successfully.")
        else:
            print("Error:", response.text)
        st.audio("received_audio.wav", format="audio/wav", start_time=0)


if __name__ == "__main__":
    main()
