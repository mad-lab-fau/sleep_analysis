import random

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable


class Model(nn.Module):
    def __init__(
        self, num_classes, input_size, hidden_size, num_layers, use_gpu, dataset_name="dataset_name", modality="acc"
    ):
        super(Model, self).__init__()
        torch.manual_seed(seed=42)
        torch.cuda.manual_seed(seed=42)
        torch.cuda.manual_seed_all(seed=42)
        random.seed(42)
        np.random.seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.num_classes = num_classes  # number of classes
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.modality = modality
        self.dataset_name = dataset_name

        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True
        )  # lstm
        if use_gpu:
            self.lstm = self.lstm.cuda()
            self.fc_1 = nn.Linear(hidden_size, 128).cuda()  # fully connected 1
            self.fc = nn.Linear(128, num_classes).cuda()  # fully connected last layer
            self.dropout = nn.Dropout(0.2).cuda()
            self.relu = nn.ReLU().cuda()
            self.softmax = nn.Softmax(dim=1).cuda()

        else:
            self.fc_1 = nn.Linear(hidden_size, 128)  # fully connected 1
            self.fc = nn.Linear(128, num_classes)  # fully connected last layer
            self.dropout = nn.Dropout(0.2)
            self.relu = nn.ReLU()
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        use_gpu = torch.cuda.is_available()

        # If the tensor is 2D, we need to expand it to 3D
        if len(x.shape) == 2:
            x = x.reshape(x.shape[0], x.shape[1], 1)

        if use_gpu:
            h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()  # hidden state
            c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()  # internal state
        else:
            h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # hidden state
            c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # internal state

        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0))  # lstm with input, hidden, and internal state
        hn = hn[-1:]
        hn = hn.view(-1, self.hidden_size)  # reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out)  # first Dense
        out = self.dropout(out)  # dropout
        out = self.relu(out)  # relu
        out = self.fc(out)  # Final Output
        return out
