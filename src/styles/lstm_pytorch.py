import os
import sys

import numpy
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch import FloatTensor
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matrices
from parameters import load_parameters
from styles.neural_network import training_data


parameters, output_files = load_parameters(
    default_parameters={
        'epochs': 50,
        'width': 10,
        'height': 10,
        'window_size': 9,
        'embedding_dim': 5,
        'hidden_dim': 5,
        'output_dim': 2,
        'vocab_size': 512,
        'learning_rate': 0.3,
        'random_seed': 42,
    },
    default_output_directory='test/training',
    output_filenames={
        'model_parameters': 'lstm_pytorch_parameters.pth',
        'model': 'lstm_pytorch_model.pth',
    },
)
model_parameters_file = output_files['model_parameters']
model_file = output_files['model']


class Style:
    @staticmethod
    def populate_args(parser):
        parser.add_argument('--model-parameters-file',
                            default=model_parameters_file)
        parser.add_argument('--model-file',
                            default=model_file)

    def __init__(self, args):
        model_parameters = torch.load(args.model_parameters_file)
        self.model = Model(**model_parameters)
        self.model.load_state_dict(torch.load(args.model_file))
        self.vocabulary = get_vocabulary(
                model_parameters['vocab_size'],
                model_parameters['window_size'])

    def next(self, grid):
        reshaped = numpy.matrix(matrices.windows(grid).reshape(grid.size, 9))
        with open('/tmp/foo', 'w') as f:
            print(type(grid), file=f)
            print(type(reshaped), file=f)
        features = encode_features(reshaped, self.vocabulary)
        predictions = self.model(features).detach().numpy()
        y = numpy.argmax(predictions, axis=1)
        return y.reshape(grid.shape)


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
    return numpy.array(numpy.argmax(labels, axis=1).flatten().tolist()[0])


class Model(nn.Module):
    def __init__(self, width, height, num_windows, window_size,
                 embedding_dim, hidden_dim, output_dim, vocab_size):
        super().__init__()
        self.init_parameters = {
            'width': width,
            'height': height,
            'num_windows': num_windows,
            'window_size': window_size,
            'embedding_dim': embedding_dim,
            'hidden_dim': hidden_dim,
            'output_dim': output_dim,
            'vocab_size': vocab_size,
        }
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.width = width
        self.height = height
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
    labels = numpy.array(labels.flatten().tolist()[0])
    features = encode_features(features, vocabulary)
    preds = model(features).data.ge(threshold).numpy().flatten()
    f1 = f1_score(labels, preds)
    accuracy = accuracy_score(labels, preds)
    print('Accuracy: {}'.format(accuracy))
    print('F1 Score: {}'.format(f1))


def train(epochs,
          width, height, window_size,
          embedding_dim, hidden_dim, output_dim,
          vocab_size, learning_rate, random_seed):
    numpy.random.seed(random_seed)
    features, labels = training_data(width, height)
    vocabulary = get_vocabulary(vocab_size, window_size)

    model = Model(width, height, features.shape[0], window_size,
                  embedding_dim, hidden_dim, output_dim, vocab_size)
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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

    torch.save(model.init_parameters, model_parameters_file)
    torch.save(model.state_dict(), model_file)
    print('Saved model in {}'.format(model_file))

    test(width, height, model, vocabulary)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        if os.path.isfile(model_file):
            model_parameters = torch.load(model_parameters_file)
            model = Model(**model_parameters)
            model.load_state_dict(torch.load(model_file))
            test(model.width, model.height, model,
                 get_vocabulary(model.vocab_size, model.window_size))
        else:
            print('Could not find save file for this model')
    else:
        train(**parameters)
