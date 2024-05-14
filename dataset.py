import os
import string
import torch
import librosa
from torch.utils.data import Dataset

class AsrDataset(Dataset):
    def __init__(self, scr_file=None, feature_file=None, feature_label_file=None, wav_scp=None, wav_dir=None):
        self.blank = "<blank>"
        self.silence = "<sil>"
        self.label_names = [self.blank, self.silence]

        all_characters = list(string.ascii_lowercase) + [self.blank, self.silence] + list(string.punctuation)
        self.letter_to_idx = {char: idx for idx, char in enumerate(all_characters)}

        if feature_label_file and os.path.exists(feature_label_file):
            with open(feature_label_file, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
                if lines and lines[0].startswith('jhucsp'):
                    lines = lines[1:]
                self.label_names += lines
            print(f"Loaded {len(self.label_names) - 2} label names from {feature_label_file}")
        else:
            print(f"Feature label file not found or not provided: {feature_label_file}")

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

        self.features = []
        if feature_file and os.path.exists(feature_file):
            with open(feature_file, 'r') as file:
                lines = [line.strip() for line in file.readlines()]
                if lines[0].startswith('jhucsp'):
                    self.features = lines[1:]
                else:
                    self.features = lines
            print(f"Loaded {len(self.features)} features from {feature_file}")
        elif feature_label_file and os.path.exists(feature_label_file):
            with open(feature_label_file, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
                if lines and lines[0].startswith('jhucsp'):
                    lines = lines[1:]
                self.features = lines
            print(f"Loaded {len(self.features)} features from {feature_label_file}")
        else:
            print(f"Feature file not found or not provided: {feature_file}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        word = self.script[idx] if self.script else self.silence  # Return silence if script is not available
        feature = self.features[idx]

        spelling = [self.letter_to_idx.get(letter, self.letter_to_idx[self.silence]) for letter in word]
        feature_indices = [self.label_names.index(label) if label in self.label_names else self.label_names.index(self.silence) for label in feature.split()]

        return torch.tensor(spelling, dtype=torch.long), torch.tensor(feature_indices, dtype=torch.long)
