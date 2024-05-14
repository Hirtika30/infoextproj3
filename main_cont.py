import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import torch.nn.functional as F

from dataset import AsrDataset
from model import LSTM_ASR


def collate_fn(batch):
    word_spellings, features = zip(*batch)

    padded_word_spellings = pad_sequence([w.clone().detach() for w in word_spellings], batch_first=True)
    padded_features = pad_sequence([f.clone().detach() for f in features], batch_first=True)

    unpadded_word_spelling_lengths = [len(w) for w in word_spellings]
    unpadded_feature_lengths = [len(f) for f in features]

    return padded_word_spellings, padded_features, unpadded_word_spelling_lengths, unpadded_feature_lengths


def train(train_dataloader, model, ctc_loss, optimizer, num_epochs):
    model.train()
    loss_values = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        for word_spellings, features, word_spelling_lengths, feature_lengths in train_dataloader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = ctc_loss(outputs.transpose(0, 1), word_spellings, feature_lengths, word_spelling_lengths)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        loss_values.append(avg_epoch_loss)
        print(f"Training Loss: {avg_epoch_loss}")

    return loss_values


def decode(test_dataloader, model):
    model.eval()
    predictions = []
    confidences = []

    with torch.no_grad():
        for batch in test_dataloader:
            _, features, _, _ = batch  # Extracting features from the batch tuple
            outputs = model(features)
            log_probs = F.log_softmax(outputs, dim=-1)
            max_probs, pred_indices = torch.max(log_probs, dim=-1)

            for i in range(log_probs.size(0)):
                pred_word = ''.join([model.label_names[idx] for idx in pred_indices[i] if idx != model.blank_index])
                confidence = torch.exp(max_probs[i]).mean().item()
                predictions.append(pred_word)
                confidences.append(confidence)
                print(f"Decoded prediction: {pred_word}, Confidence: {confidence}")  # Add logging

    return predictions, confidences


def compute_accuracy(predictions, test_set):
    if not predictions or not test_set:
        print("No predictions or test data available.")
        return
    correct = sum(1 for pred, target in zip(predictions, test_set) if pred == target)
    total = len(predictions)
    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"Accuracy: {accuracy}%")


def main():
    training_set = AsrDataset(
        scr_file='/Users/hirtikamirghani/Downloads/data (1)-1 2/clsp.trnscr',
        wav_scp='/Users/hirtikamirghani/Downloads/data (1)-1 2/clsp.trnwav',
        wav_dir='/Users/hirtikamirghani/Downloads/data (1)-1 2/waveforms',
        mfcc=True
    )

    test_set = AsrDataset(
        wav_scp='/Users/hirtikamirghani/Downloads/data (1)-1 2/clsp.devwav',
        wav_dir='/Users/hirtikamirghani/Downloads/data (1)-1 2/waveforms',
        mfcc=True
    )

    print(f"Number of scripts loaded: {len(training_set.script)}")
    print(f"Number of features loaded: {len(training_set)}")
    print(f"Number of scripts loaded: {len(test_set.script)}")
    print(f"Number of features loaded: {len(test_set)}")

    if len(training_set) == 0 or len(test_set) == 0:
        print("Datasets are empty. Please check the file paths and content.")
        return

    train_dataloader = DataLoader(training_set, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_set, batch_size=32, shuffle=False, collate_fn=collate_fn)

    label_names = training_set.label_names
    blank_index = label_names.index('<blank>')

    model = LSTM_ASR(feature_type='mfcc', input_size=40, hidden_size=256, num_layers=2,
                     output_size=28, label_names=label_names, blank_index=blank_index)

    ctc_loss = nn.CTCLoss(blank=blank_index, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    num_epochs = 10
    loss_values = train(train_dataloader, model, ctc_loss, optimizer, num_epochs)

    # Plot the training loss
    plt.plot(range(1, num_epochs + 1), loss_values, marker='o', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss vs. Epoch')
    plt.legend()
    plt.grid(True)
    plt.show()

    predictions, confidences = decode(test_dataloader, model)

    # Save the predictions to a file
    with open('test_predictions_mfcc.txt', 'w') as f:
        for pred, conf in zip(predictions, confidences):
            f.write(f"{pred}\t{conf}\n")

    test_set_script = [data[0] for data in test_set.script]
    compute_accuracy(predictions, test_set_script)


if __name__ == "__main__":
    main()
