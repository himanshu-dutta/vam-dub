# from speechbrain.inference.speaker import SpeakerRecognition
from speechbrain.inference.speaker import SpeakerRecognition


verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")
score, prediction = verification.verify_files(
    "en-pred/common_voice_it_17440353.wav", 
    "en-a/common_voice_it_17440353.mp3.wav"
    )