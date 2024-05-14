import torch
import torch.nn as nn

class LSTM_ASR(nn.Module):
    def __init__(self, feature_type="mfcc", input_size=40, hidden_size=256, num_layers=2, output_size=28, label_names=None, blank_index=None):
        super().__init__()
        self.feature_type = feature_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.label_names = label_names
        self.blank_index = blank_index

        if self.feature_type == "mfcc":
            self.embedding = nn.Linear(input_size, 64)

        self.lstm = nn.LSTM(input_size=64, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, batch_features):
        if self.feature_type == "mfcc":
            batch_features = self.embedding(batch_features)

        output, _ = self.lstm(batch_features)
        output = self.fc(output)
        return output

    def decode(self, output):
        predictions = []
        for log_probs in output.transpose(0, 1):  # Iterate over batch
            word = []
            for token_log_probs in log_probs:  # Iterate over sequence
                token = torch.argmax(token_log_probs).item()
                if token != self.blank_index:  # Skip blank tokens
                    word.append(self.label_names[token])
            predictions.append(''.join(word))
        return predictions
