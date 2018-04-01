import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, average_precision_score
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
save_file = os.path.join(training_dir, 'nn_pytorch_model.pth')


class Life(nn.Module):
    def __init__(self, features, hidden_layers, output_size, grid_w, grid_h):
        super(Life, self).__init__()
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.linear1 = nn.Linear(features, hidden_layers)
        self.linear2 = nn.Linear(hidden_layers, output_size)

    def forward(self, x):
        h = F.sigmoid(self.linear1(x))
        out = F.sigmoid(self.linear2(h))
        return out


def test(width, height, model):
    threshold = 0.999
    print('\nTesting model with {} threshold'.format(threshold))
    x, labels = training_data(width, height)
    labels = np.array(labels.flatten().tolist()[0])
    features = Variable(FloatTensor(x))
    preds = model(features).data
    bin_preds = preds.ge(threshold).numpy().flatten()
    f1 = f1_score(labels, bin_preds)
    accuracy = accuracy_score(labels, bin_preds)
    avg_prec = average_precision_score(labels, preds.numpy().flatten())
    print('Accuracy: {}'.format(accuracy))
    print('F1 Score: {}'.format(f1))
    print('Avg Precision: {}'.format(avg_prec))


def train():
    epochs = 3000
    width = 10
    height = 10
    hidden_layers = 5
    lr = 0.05

    np.random.seed(42)
    x, y = training_data(width, height)

    criterion = nn.BCELoss()
    model = Life(x.shape[1], hidden_layers, y.shape[1], width, height)
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-7)

    for i in range(1, epochs + 1):
        features = Variable(FloatTensor(x))
        labels = Variable(FloatTensor(y))
        optimizer.zero_grad()
        predictions = model(features)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        if i % 250 == 0:
            print('Epoch {}/{} - Loss: {}'.format(i, epochs, loss.data[0]))

    torch.save(model, save_file)
    print('Saved model in {}'.format(save_file))

    test(width, height, model)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        if os.path.isfile(save_file):
            model = torch.load(save_file)
            test(model.grid_w, model.grid_h, model)
        else:
            print('Could not find save file for this model')
    else:
        train()
