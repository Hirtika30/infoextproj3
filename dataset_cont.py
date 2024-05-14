import os
import string
import torch
import librosa
from torch.utils.data import Dataset


class AsrDataset(Dataset):
    def __init__(self, scr_file=None, wav_scp=None, wav_dir=None, mfcc=True):
        self.blank = "<blank>"
        self.silence = "<sil>"
        self.label_names = [self.blank, self.silence]

        all_characters = list(string.ascii_lowercase) + [self.blank, self.silence] + list(string.punctuation)
        self.letter_to_idx = {char: idx for idx, char in enumerate(all_characters)}

        self.script = []
        if scr_file and os.path.exists(scr_file):
            with open(scr_file, 'r') as file:
                lines = [line.strip() for line in file.readlines()]
                if lines[0].startswith('jhucsp'):
                    self.script = lines[1:]
                else:
                    self.script = lines
            print(f"Loaded {len(self.script)} scripts from {scr_file}")
        else:
            print(f"Script file not found or not provided: {scr_file}")

        self.wav_files = []
        if wav_scp and os.path.exists(wav_scp):
            with open(wav_scp, 'r') as f:
                self.wav_files = [line.strip() for line in f if not line.startswith('jhucsp')]
            print(f"Loaded {len(self.wav_files)} waveform file names from {wav_scp}")
        else:
            print(f"Wav file list not found or not provided: {wav_scp}")

        self.wav_dir = wav_dir
        self.mfcc = mfcc

    def __len__(self):
        return len(self.wav_files)

    def __getitem__(self, idx):
        word = self.script[idx] if self.script else self.silence  # Return silence if script is not available
        wav_file = self.wav_files[idx]

        wav_path = os.path.join(self.wav_dir, wav_file)
        wav, sr = librosa.load(wav_path, sr=None)

        if self.mfcc:
            features = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=40, hop_length=160, win_length=400).transpose()
        else:
            features = wav  # If not MFCC, use raw wav (or other features if implemented)

        spelling = [self.letter_to_idx.get(letter, self.letter_to_idx[self.silence]) for letter in word]

        return torch.tensor(spelling, dtype=torch.long), torch.tensor(features, dtype=torch.float)

    def compute_mfcc(self, wav_scp, wav_dir):
        features = []
        if wav_scp and os.path.exists(wav_scp):
            with open(wav_scp, 'r') as f:
                for wavfile in f:
                    wavfile = wavfile.strip()
                    if wavfile.startswith('jhucsp'):
                        continue
                    wav, sr = librosa.load(os.path.join(wav_dir, wavfile), sr=None)
                    feats = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=40, hop_length=160, win_length=400).transpose()
                    features.append(feats)
        return features
