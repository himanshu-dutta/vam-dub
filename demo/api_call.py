import requests

# Specify the URL of the FastAPI endpoint
url = "http://localhost:9000/inference/"

# Define files to upload
audio_file = ("audio.wav", open("/home/cv_it.mp3", "rb"))
image_file = ("image.jpg", open("/home/check_data/house.jpeg", "rb"))

# Make the request with both files
# files = {"audio": audio_file, "image": image_file}
files = [("files", audio_file), ("files", image_file)]

response = requests.post(url, files=files)

# Check if the request was successful
if response.status_code == 200:
    # Save the received audio file
    with open("received_audio.wav", "wb") as f:
        f.write(response.content)
    print("Audio file received successfully.")
else:
    print("Error:", response.text)
