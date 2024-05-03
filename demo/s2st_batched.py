import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5,4"

from transformers import (
    AutoProcessor,
    SeamlessM4Tv2ForSpeechToSpeech,
)
import librosa
import soundfile as sf
import torch

import os
import argparse
import math
from tqdm import tqdm

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def load_model_processor():
    processor = AutoProcessor.from_pretrained(
        "facebook/seamless-m4t-v2-large",
    )
    model = SeamlessM4Tv2ForSpeechToSpeech.from_pretrained(
        "facebook/seamless-m4t-v2-large",
    )
    return processor, model.to(device)


def get_file_name_without_extension(file_path):
    file_name_with_extension = os.path.basename(file_path)
    file_name_without_extension, _ = os.path.splitext(file_name_with_extension)
    return file_name_without_extension


def find_files(directory, fmt):
    pickle_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(fmt):
                pickle_files.append(os.path.join(root, file))
    return pickle_files


def load_audio(path, sr):
    audio, _ = librosa.load(path, sr=sr)
    return audio


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", type=str, required=True)
    parser.add_argument("-o", "--output-dir", type=str, required=True)
    parser.add_argument("-s", "--src-lang", type=str, default="ita")
    parser.add_argument("-t", "--tgt-lang", type=str, default="eng")
    parser.add_argument("-b", "--batch-size", type=int, default=16)
    parser.add_argument("-r", "--sampling-rate", type=int, default=16000)

    args = parser.parse_args()

    processor, model = load_model_processor()
    src_files = find_files(args.input_dir, ".mp3") + find_files(args.input_dir, ".wav")
    print(f"Loaded {len(src_files)} files.")

    for b_idx in tqdm(range(math.ceil(len(src_files) / args.batch_size))):
        st_idx = b_idx * args.batch_size
        en_idx = st_idx + args.batch_size
        src_files_batch = src_files[st_idx : min((en_idx, len(src_files)))]

        audios = [
            load_audio(src_file, sr=args.sampling_rate) for src_file in src_files_batch
        ]
        audio_inputs = processor(
            audios=audios,
            src_lang=args.src_lang,
            return_tensors="pt",
            sampling_rate=args.sampling_rate,
        ).to(device)
        audio_array_from_audio, lens = model.generate(
            **audio_inputs, tgt_lang=args.tgt_lang
        )

        audio_array_from_audio = [
            aud[: int(l)].squeeze()
            for aud, l in zip(list(audio_array_from_audio.cpu().numpy()), list(lens))
        ]

        for src_file, aud in zip(src_files_batch, audio_array_from_audio):
            file_name = get_file_name_without_extension(src_file)
            sf.write(
                os.path.join(args.output_dir, file_name + ".wav"),
                aud,
                args.sampling_rate,
            )
