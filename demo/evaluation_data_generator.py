import os
import glob
import argparse
import numpy as np
import matplotlib
import PIL
import pickle
from scipy.io.wavfile import write
import random
from tqdm import tqdm

RATE = 16000


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


def read_vam_data(path):
    with open(path, "rb") as fp:
        obj = pickle.load(fp)

    img = np.concatenate(obj["rgb"], axis=1)
    img = PIL.Image.fromarray(np.uint8(img), "RGB")

    source_audio_data = obj["source_audio"]
    source_audio_data = np.int16(
        source_audio_data / np.max(np.abs(source_audio_data)) * 32767
    )

    receiver_audio_data = obj["receiver_audio"]
    receiver_audio_data = np.int16(
        receiver_audio_data / np.max(np.abs(receiver_audio_data)) * 32767
    )

    return {
        "img": img,
        "source_audio": source_audio_data,
        "receiver_audio": receiver_audio_data,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser("evaluation data generator")
    parser.add_argument("-d", "--data", type=str, required=True)
    parser.add_argument("-p", "--pickle-files-dir", type=str, required=True)
    parser.add_argument("-i", "--img-save-path", type=str, required=True)
    parser.add_argument("-a", "--acoustic-save-path", type=str, required=True)

    args = parser.parse_args()

    # Example usage:
    pickle_files_list = find_files(args.pickle_files_dir, ".pkl")
    wav_files_list = find_files(args.data, ".wav")

    print(
        f"Loaded {len(pickle_files_list)} pickle files and {len(wav_files_list)} wav files."
    )

    for wav_file in tqdm(wav_files_list):
        file_name = get_file_name_without_extension(wav_file)
        pkl_file_path = random.choice(pickle_files_list)
        vam_data = read_vam_data(pkl_file_path)

        vam_data["img"].save(
            os.path.join(args.img_save_path, file_name + ".png"), "PNG"
        )
        write(
            os.path.join(args.acoustic_save_path, file_name + ".wav"),
            RATE,
            vam_data["receiver_audio"],
        )
