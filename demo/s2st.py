import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5,4"

import torch
from transformers import (
    AutoProcessor,
    SeamlessM4TModel,
    SeamlessM4Tv2Model,
    SeamlessM4Tv2ForSpeechToSpeech,
)
import librosa
import soundfile as sf

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input-audio", type=str)
parser.add_argument("-o", "--output-audio", type=str)
args = parser.parse_args()

device = "cuda:0" if torch.cuda.is_available() else "cpu"


processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2ForSpeechToSpeech.from_pretrained(
    "facebook/seamless-m4t-v2-large"
).to(device)

audio, _ = librosa.load(args.input_audio, sr=16000)
audio_inputs = processor(audios=audio, return_tensors="pt").to(device)
audio_array_from_audio = (
    model.generate(**audio_inputs, tgt_lang="eng")[0].cpu().numpy().squeeze()
)

print(audio_array_from_audio.shape)
sf.write(args.output_audio, audio_array_from_audio, 16000)
