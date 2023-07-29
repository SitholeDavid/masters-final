import os

import librosa
import pandas as pd
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

def prepare_align(config):
    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    speaker = "spanish-single-speaker"
    transcripts =  pd.read_csv( os.path.join(in_dir, 'transcript.txt'), sep='|', header=None)
    transcripts.columns = ['file', 'original script', 'normalized script', 'duration']

    for _, row in tqdm(transcripts.iterrows(), total=len(transcripts)):
        if not isinstance(row['normalized script'], str):
            continue 

        wav_path = os.path.join(in_dir,  row['file'])
        base_name = row['file'].split('/')[-1].replace('.wav', '')
        
        if not os.path.exists(wav_path):
            wav_path = os.path.join(in_dir, f'{base_name}.wav')

        if os.path.exists(wav_path):
            os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
            text = row['normalized script']
            wav, _ = librosa.load(wav_path, sr=sampling_rate)
            wav = wav / max(abs(wav)) * max_wav_value

            wavfile.write(
                os.path.join(out_dir, speaker, "{}.wav".format(base_name)),
                sampling_rate,
                wav.astype(np.int16),
            )
            with open(
                os.path.join(out_dir, speaker, "{}.lab".format(base_name)),
                "w",
            ) as f1:
                f1.write(text)

 
            
           