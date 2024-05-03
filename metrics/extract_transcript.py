import pandas as pd
import glob, os, shutil
from tqdm.auto import tqdm


data = pd.read_csv('HindiEngData/Italian-English/test.tsv', sep='\t')
reference_list = []
audio_path = glob.glob('Italian-eng-short/*.wav')

i = 0
for audio in tqdm(audio_path):
    basename = os.path.basename(audio)
    filename = basename.replace('.wav', '')
    # print(filename)
    # print('data: ',data.loc[data['filename']==filename, 'sentence'])
    try:
        reference_sentence = data.loc[data['filename']==filename, 'sentence'].iloc[0]
        reference_list.append(reference_sentence+'\n')
        shutil.copyfile(f"en-pred/{filename.replace('.mp3', '')}.wav", f"en-pred-asrbleau/{i}_pred.wav")
    except IndexError:
        print(filename)
        # os.remove(os.path.join('Italian-short', filename))
        os.remove(os.path.join('Italian-eng-short', basename))
        os.remove(os.path.join('en-pred', filename.replace('.mp3', '.wav')))
    i += 1

with open('reference.txt', 'w') as f:
    f.writelines(reference_list)
