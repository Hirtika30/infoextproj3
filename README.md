# infoextproj3

# Isolated-Word Speech Recognizer Using CTC

## Project Overview

This project aims to build an isolated-word speech recognizer for a vocabulary of 48 words using the Connectionist Temporal Classification (CTC) objective function. The recognizer processes speech represented via quantized spectral features and Mel-frequency cepstral coefficients (MFCCs).

## Project Structure

├── dataset.py
├── model.py
├── main.py
├── README.md
├── requirements.txt
└── data/
├── clsp.lblnames
├── clsp.trnscr
├── clsp.trnlbls
├── clsp.trnwav
├── clsp.endpts
├── clsp.devwav
└── clsp.devlbls


## Data Description

- **clsp.lblnames**: Contains 256 two-character-long label names, one per line.
- **clsp.trnscr**: Script read by speakers, with 798 lines of data.
- **clsp.trnlbls**: Processed speech labels corresponding to the utterances in the script.
- **clsp.trnwav**: Names of the waveform files corresponding to each utterance in the script.
- **clsp.endpts**: Endpoint information indicating the start and end of each utterance.
- **clsp.devwav**: Names of the waveform files for the test set.
- **clsp.devlbls**: Labels for the test set utterances.

## Implementation Details

### Dataset Class (`dataset.py`)

This module handles loading the data from the provided files and preparing it for training. It supports both discrete and continuous feature representations.

- **Initialization**: The `AsrDataset` class initializes the dataset by loading label names, scripts, and features from the specified files. It also handles the mapping of letters to indices for input sequences.
- **Data Loading**: The class checks if the files exist and reads them line by line, stripping any extra whitespace and ignoring title lines if present.
- **`__len__`**: Returns the length of the features list.
- **`__getitem__`**: Retrieves the spelling of the word and the corresponding features for the given index, converting them into tensors.
- **`compute_mfcc`**: Computes MFCC features for the waveform files listed in the provided `wav_scp` file.

### Model Class (`model.py`)

This module defines the neural network model for speech recognition using LSTM layers.

- **LSTM_ASR Class**: Defines an LSTM-based model with embedding layers for discrete features.
  - **Initialization**: Sets up the network architecture, including embedding, LSTM, and fully connected layers.
  - **Forward Pass**: Processes the input features through the LSTM and fully connected layers to generate the output.
  - **Decode**: Decodes the output of the network into predicted word sequences by taking the most likely tokens at each time step.

### Training and Testing Script (`main.py`)

This module manages the training and testing process, including data loading, model initialization, and training loop. It also includes functions for decoding and computing accuracy.

- **`collate_fn`**: Pads sequences to the same length for batch processing.
- **`train`**: Trains the model using the CTC loss function. It iterates through the data loader, performs forward and backward passes, and updates the model parameters.
- **`decode`**: Evaluates the model on the test set and generates predictions.
- **`compute_accuracy`**: Calculates the accuracy of the model by comparing predictions with the ground truth labels.
- **`main`**: Main function to load data, initialize the model, start the training process, and evaluate the model on the test set.

## Training Results

### Training Loss

The training loss over 10 epochs is shown in the graph below:

![Training Loss vs. Epoch](Screenshot%202024-05-13%20at%2010.06.43%20PM.png)

### Loaded Data

- 256 label names from `clsp.lblnames`
- 798 scripts from `clsp.trnscr`
- 798 features from `clsp.trnlbls`
- 393 features from `clsp.devlbls`

### Training Output

Training Loss: 8.855269782543182
Training Loss: 4.745695620775223
Training Loss: 3.3873727416992185
Training Loss: 2.8302918243408204
Training Loss: 2.5141483449935915
Training Loss: 2.3518917465209963
Training Loss: 2.3360790157318116
Training Loss: 2.3240373134613037
Training Loss: 2.3259535789489747
Training Loss: 2.3178306865692138

### Test Output

The file test_predictions.txt contains the test output i.e the confidence of each utterance.

## How to Run

1. **Clone the Repository**:
   ```sh
   git clone <repository-url>
   cd <repository-directory>
   
2. **Place Data Files**:
3. 
Ensure all data files (clsp.lblnames, clsp.trnscr, clsp.trnlbls, clsp.trnwav, clsp.endpts, clsp.devwav, clsp.devlbls) are in the data/ directory.

4. **Install Dependencies**:
5. 
   pip install -r requirements.txt
   
7. **Run the Training Script**:

   python main.py

8. Requirements

Here is the requirements.txt file for the project:

torch==1.10.0
torchaudio==0.10.0
librosa==0.8.1
numpy==1.21.2
matplotlib==3.4.3
