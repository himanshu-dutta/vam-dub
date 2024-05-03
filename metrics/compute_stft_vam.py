import torch
import torchaudio
import os
import glob
from tqdm.auto import tqdm


mse_acc = []
# Load the WAV files
audio_path_list = glob.glob('en-pred/*.wav')
for audio_path in tqdm(audio_path_list):
    basename = os.path.basename(audio_path)
    filename = basename.replace('.wav', '')

    waveform1, sample_rate1 = torchaudio.load(audio_path)
    waveform2, sample_rate2 = torchaudio.load(f"en-a/{filename}.mp3.wav")

    # Make both waveforms have the same length
    min_length = min(waveform1.size(1), waveform2.size(1))
    waveform1 = waveform1[:, :min_length]
    waveform2 = waveform2[:, :min_length]

    # Compute the STFTs
    stft_transform1 = torch.stft(
        waveform1, n_fft=400, hop_length=160, win_length=400, window=torch.hamming_window(400), return_complex=True
    )
    stft_transform2 = torch.stft(
        waveform2, n_fft=400, hop_length=160, win_length=400, window=torch.hamming_window(400), return_complex=True
    )

    # Compute the element-wise squared difference between the STFTs
    squared_diff = (torch.abs(stft_transform1) - torch.abs(stft_transform2)) ** 2

    # Compute the mean of the squared differences
    mse = torch.mean(squared_diff)
    mse_acc.append(mse.item())

print("STFT Distance (the MSE between the generated and true target audio's magnitude spectrograms): ", sum(mse_acc)/len(mse_acc))
