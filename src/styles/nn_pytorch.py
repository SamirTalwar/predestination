import os
import sys

import numpy
from sklearn.metrics import accuracy_score, f1_score, average_precision_score
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
        "epochs": 3000,
        "width": 10,
        "height": 10,
        "hidden_layers": 5,
        "learning_rate": 0.05,
        "random_seed": 42,
    },
    default_output_directory="test/training",
    output_filenames={
        "model_parameters": "nn_pytorch_parameters.pth",
        "model": "nn_pytorch_model.pth",
    },
)
model_parameters_file = output_files["model_parameters"]
model_file = output_files["model"]


class Style:
    NAME = "nn-pytorch"

    @staticmethod
    def populate_args(parser):
        parser.add_argument("--model-parameters-file", default=model_parameters_file)
        parser.add_argument("--model-file", default=model_file)

    def __init__(self, args):
        model_parameters = torch.load(args.model_parameters_file)
        self.model = Model(**model_parameters)
        self.model.load_state_dict(torch.load(args.model_file))

    def next(self, grid):
        reshaped = matrices.windows(grid).reshape(grid.size, 9)
        features = Variable(FloatTensor(reshaped))
        predictions = self.model(features).detach().numpy()
        y = numpy.argmax(predictions, axis=1)
        return y.reshape(grid.shape)


class Model(nn.Module):
    def __init__(self, features, hidden_layers, output_size, grid_w, grid_h):
        super().__init__()
        self.init_parameters = {
            "features": features,
            "hidden_layers": hidden_layers,
            "output_size": output_size,
            "grid_w": grid_w,
            "grid_h": grid_h,
        }
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
    print("\nTesting model with {} threshold".format(threshold))
    x, labels = training_data(width, height)
    labels = numpy.array(labels.flatten().tolist()[0])
    features = Variable(FloatTensor(x))
    preds = model(features).data
    bin_preds = preds.ge(threshold).numpy().flatten()
    f1 = f1_score(labels, bin_preds)
    accuracy = accuracy_score(labels, bin_preds)
    avg_prec = average_precision_score(labels, preds.numpy().flatten())
    print("Accuracy: {}".format(accuracy))
    print("F1 Score: {}".format(f1))
    print("Avg Precision: {}".format(avg_prec))


def train(epochs, width, height, hidden_layers, learning_rate, random_seed):
    numpy.random.seed(random_seed)
    x, y = training_data(width, height)

    criterion = nn.BCELoss()
    model = Model(x.shape[1], hidden_layers, y.shape[1], width, height)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=1e-7)

    for i in range(1, epochs + 1):
        features = Variable(FloatTensor(x))
        labels = Variable(FloatTensor(y))
        optimizer.zero_grad()
        predictions = model(features)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        if i % 250 == 0:
            print("Epoch {}/{} - Loss: {}".format(i, epochs, loss.data[0]))

    torch.save(model.init_parameters, model_parameters_file)
    torch.save(model.state_dict(), model_file)
    print("Saved model in {}".format(model_file))

    test(width, height, model)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        if os.path.isfile(model_file):
            model_parameters = torch.load(model_parameters_file)
            model = Model(**model_parameters)
            model.load_state_dict(torch.load(model_file))
            test(model.grid_w, model.grid_h, model)
        else:
            print("Could not find save file for this model")
    else:
        train(**parameters)
