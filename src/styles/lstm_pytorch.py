import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch import FloatTensor
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from styles.neural_network import training_data


root = os.path.realpath(os.path.join(
    os.path.dirname(__file__), os.path.pardir, os.path.pardir))
training_dir = os.path.join(root, 'test', 'training')
save_file = os.path.join(training_dir, 'lstm_pytorch.pth')


def get_vocabulary(vocab_size, window_size):
    vocabulary = {}
    for i in range(vocab_size):
        s = bin(i)[2:].zfill(window_size)
        vocabulary[s] = i
    return vocabulary


def window_to_bin_str(w):
    return ''.join(map(str, map(int, w.tolist()[0])))


def encode_features(seq, to_ix):
    idxs = [to_ix[window_to_bin_str(w)] for w in seq]
    return Variable(torch.LongTensor(idxs))


def prepare_labels(labels):
    return np.array(np.argmax(labels, axis=1).flatten().tolist()[0])


class LSTMGOL(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_windows,
                 output_dim, vocab_size, window_size, grid_w, grid_h):
        super(LSTMGOL, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.window_size = window_size
        self.num_windows = num_windows
        self.window_embeddings = nn.Embedding(num_windows, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.h_to_output = nn.Linear(hidden_dim, output_dim)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (Variable(torch.zeros(1, 1, self.hidden_dim)),
                Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, windows):
        wl = windows.shape[0]
        embeds = self.window_embeddings(windows)
        lstm_out, self.hidden = self.lstm(embeds.view(wl, 1, -1), self.hidden)
        target_space = self.h_to_output(lstm_out.view(wl, -1))
        output = F.sigmoid(target_space)
        return output


def test(width, height, model, vocabulary):
    threshold = 0.999
    print('\nTesting model with {} threshold'.format(threshold))
    features, labels = training_data(width, height)
    labels = np.array(labels.flatten().tolist()[0])
    features = encode_features(features, vocabulary)
    preds = model(features).data.ge(threshold).numpy().flatten()
    f1 = f1_score(labels, preds)
    accuracy = accuracy_score(labels, preds)
    print('Accuracy: {}'.format(accuracy))
    print('F1 Score: {}'.format(f1))


def train():
    grid_w, grid_h, window_size = 10, 10, 9
    embedding_dim, hidden_dim, output_dim = 5, 5, 2
    vocab_size = 512

    epochs = 50
    lr = 0.3

    np.random.seed(42)
    features, labels = training_data(grid_w, grid_h)
    vocabulary = get_vocabulary(vocab_size, window_size)

    model = LSTMGOL(embedding_dim, hidden_dim, features.shape[0],
                    output_dim, vocab_size, window_size, grid_w, grid_h)
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for e in range(1, epochs + 1):
        model.zero_grad()
        model.hidden = model.init_hidden()

        w_feats = encode_features(features, vocabulary)
        w_labels = Variable(FloatTensor(labels), requires_grad=False)
        predictions = model(w_feats)

        loss = loss_fn(predictions, w_labels)
        loss.backward()
        optimizer.step()

        if e % 10 == 0:
            print('Epoch {}/{} - Loss: {}'.format(e, epochs, loss.data[0]))

    torch.save(model, save_file)
    print('Saved model in {}'.format(save_file))

    test(grid_w, grid_h, model, vocabulary)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        if os.path.isfile(save_file):
            model = torch.load(save_file)
            test(model.grid_w, model.grid_h, model,
                 get_vocabulary(model.vocab_size, model.window_size))
        else:
            print('Could not find save file for this model')
    else:
        train()
