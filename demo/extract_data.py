import pickle

import numpy as np
from scipy.io.wavfile import write

from matplotlib import pyplot as plt
import PIL


with open("/home/data/vam/test-seen/JeFG25nYj2p/1188-133604-0005.pkl", "rb") as fp:
    obj = pickle.load(fp)

print(obj.keys(), type(obj))
# input()
# print(type(obj["source_audio"]), obj["source_audio"].shape, obj["source_audio"][:10])
# print(type(obj["rgb"]), type(obj["rgb"][0]), obj["rgb"][0].shape)
# print(type(obj["rgb"]), type(obj["rgb"][0]), obj["rgb"][17].shape)

img = np.concatenate(obj["rgb"], axis=1)
img = PIL.Image.fromarray(np.uint8(img), "RGB")
img.save(f"check_data/img.png", "PNG")

rate = 16000
source_audio_data = obj["source_audio"]
scaled = np.int16(source_audio_data / np.max(np.abs(source_audio_data)) * 32767)
write("check_data/source_audio.wav", rate, scaled)

rate = 16000
receiver_audio_data = obj["receiver_audio"]
scaled = np.int16(receiver_audio_data / np.max(np.abs(receiver_audio_data)) * 32767)
write("check_data/receiver_audio.wav", rate, scaled)
