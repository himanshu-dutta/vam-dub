# from speechbrain.inference.speaker import SpeakerRecognition
from speechbrain.inference.speaker import SpeakerRecognition
import os
import glob
from tqdm.auto import tqdm

verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")

pred_pred_list = []
pred_score_list = []
ac_pred_list = []
ac_score_list = []
# Load the WAV files
audio_path_list = glob.glob('en-pred/*.wav')
for audio_path in tqdm(audio_path_list):
    basename = os.path.basename(audio_path)
    filename = basename.replace('.wav', '')
    
    score, prediction = verification.verify_files(
        f"en-a/{filename}.mp3.wav",
        f"Italian-short/{filename}.mp3"
        )
    ac_score_list.append(score.item())

print('VAM voice preservation score: ', sum(ac_score_list)/len(ac_score_list))
