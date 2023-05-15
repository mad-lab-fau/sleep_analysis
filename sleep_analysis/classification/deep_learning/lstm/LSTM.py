import datetime
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss

from sleep_analysis.classification.deep_learning.dl_scoring import dl_score, tensor_to_performance
from sleep_analysis.classification.deep_learning.lstm.model import Model
from sleep_analysis.classification.deep_learning.utils import get_num_classes
from sleep_analysis.datasets.mesadataset import *

# from torch.utils.tensorboard import SummaryWriter


class LSTM:
    def __init__(
        self,
        num_epochs,
        learning_rate,
        input_size,
        hidden_size,
        num_layers,
        seq_len,
        batch_size,
        modality,
        dataset_name,
        classification_type="binary",
    ):
        torch.manual_seed(seed=42)
        torch.cuda.manual_seed(seed=42)
        torch.cuda.manual_seed_all(seed=42)
        random.seed(42)
        np.random.seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # parameters of LSTM
        self.num_epochs = num_epochs  # number of epochs
        self.learning_rate = learning_rate  # set learning rate
        self.input_size = input_size  # number of features
        self.hidden_size = hidden_size  # number of features in hidden state
        self.num_layers = num_layers  # number of stacked lstm layers
        self.seq_len = seq_len  # sequence length of input sequence

        self.use_gpu = torch.cuda.is_available()
        self.batch_size = batch_size
        self.modality = modality
        self.dataset_name = dataset_name
        self.classification_type = classification_type

        if self.use_gpu:
            self.device = "cuda"
        else:
            self.device = "cpu"

    def train(self, x_train, y_train, x_val, y_val):

        # get number of classes
        num_classes = get_num_classes(self.classification_type)

        # load batched data
        x_batch_train_list, y_batch_train_list = self.batch_loader(x_train, y_train)

        # load empty model of our lstm class that needs to be trained
        lstm = self._load_empty_model(use_gpu=self.use_gpu, num_classes=num_classes)

        if self.classification_type == "binary":
            criterion = BCEWithLogitsLoss()  # binary cross-entropy loss for classification
        else:
            criterion = CrossEntropyLoss()  # multi-class loss

        optimizer = torch.optim.Adam(lstm.parameters(), lr=self.learning_rate)

        # initialize variables to determine and save best model in val-procedure
        min_val_loss = 9999999
        last_loss = 9999999
        max_val_performance = 0.0

        for epoch in range(self.num_epochs):
            train_losses = []
            lstm.train()

            # iterate over all batches of training data
            for x_batch_train, y_batch_train in zip(x_batch_train_list, y_batch_train_list):

                if self.use_gpu:
                    x_batch_train = x_batch_train.to(device=self.device)
                    y_batch_train = y_batch_train.to(device=self.device)

                outputs = lstm.forward(x_batch_train)  # forward pass
                optimizer.zero_grad()  # calculate the gradient, manually setting to 0

                # obtain the loss function
                if self.classification_type == "binary":
                    loss = criterion(outputs, y_batch_train)
                else:
                    loss = criterion(outputs, torch.squeeze(y_batch_train).long())

                loss.backward()  # calculates the loss of the loss function
                optimizer.step()  # improve from loss, i.e. backprop
                train_losses.append(loss.item())

            # change this line dependent on how often validation loss should be calculated
            if epoch % 1 == 0:

                # print train loss first
                print("-------------------------")
                print(datetime.datetime.now())
                print("-------------------------")
                print("Epoch: %d, train loss: %1.5f" % (epoch, np.mean(train_losses)))

                val_losses = []
                val_performances = []
                lstm.eval()
                # load validation data in batches
                x_batch_val_list, y_batch_val_list = self.batch_loader(x_val, y_val)

                # iterate over all batches of validation data
                for x_batch_val, y_batch_val in zip(x_batch_val_list, y_batch_val_list):

                    # move data to gpu if available
                    x_batch_val = x_batch_val.to(self.device)
                    y_batch_val = y_batch_val.to(self.device)

                    y_pred = lstm.forward(x_batch_val)

                    # calculate loss of batch-wise prediction
                    if self.classification_type == "binary":
                        val_loss = criterion(y_pred, y_batch_val)
                    else:
                        val_loss = criterion(y_pred, torch.squeeze(y_batch_val).long())

                    # append batch-wise loss to corresponding list
                    val_losses.append(val_loss.item())

                    # calculate accuracy of prediction
                    class_performance = tensor_to_performance(y_batch_val, y_pred, self.classification_type)

                    # append accuracy of batch-wise prediction to corresponding list
                    val_performances.append(class_performance["mcc"])

                # calculate mean loss/accuracy over all batches
                mean_val_loss = np.mean(val_losses)
                mean_performance = np.mean(val_performances)

                print("validation_loss: " + str(mean_val_loss))
                print("validation_performance: " + str(mean_performance))
                print("-------------------------")

                # define early stopping criteria
                if (
                    ((mean_val_loss - min_val_loss) > 0.05 or mean_val_loss == min_val_loss)
                    and epoch > 10
                    or (mean_performance <= 0.0 and epoch > 10)
                    or mean_val_loss == last_loss
                ):
                    print("break because of bad loss development")
                    break

                # save model if validation loss is better than previous best validation loss
                if min_val_loss > mean_val_loss:
                    min_val_loss = mean_val_loss
                    max_val_performance = mean_performance

                    print("*************************")
                    print("new best validation_loss: " + str(np.mean(val_losses)))
                    print("validation performance: " + str(np.mean(mean_performance)))
                    print("*************************")

                    torch.save(
                        lstm.state_dict(),
                        Path(__file__)
                        .parents[4]
                        .joinpath(
                            "exports/pickle_pipelines/lstm_"
                            + "_".join(self.modality)
                            + "_"
                            + self.dataset_name
                            + "_"
                            + self.classification_type
                        ),
                    )

                last_loss = mean_val_loss

        return max_val_performance

    def test(self, x_test, y_test):

        score_dict = {}
        pred_dict = {}

        num_classes = get_num_classes(self.classification_type)

        if self.use_gpu:
            # load "empty" lstm model --> same parameters as in training
            lstm = self._load_empty_model(use_gpu=True, num_classes=num_classes)

            # load best model obtained from training
            self._load_best_model(lstm)
            lstm = lstm.cuda()

        else:
            lstm = self._load_empty_model(use_gpu=False, num_classes=num_classes)

            # load best model obtained from training
            self._load_best_model(lstm)

        lstm.eval()
        # iterate over test data
        for i, (x_batch_test, y_batch_test) in enumerate(zip(x_test, y_test)):
            # obtain subject_id
            subj_idx = x_batch_test[1]
            # move data to gpu if available
            x_batch_test = x_batch_test[0].to(device=self.device)

            # apply model to test data and move to cpu and convert to numpy array
            y_pred = lstm.forward(x_batch_test).to(device="cpu")
            y_pred = y_pred.detach().numpy()

            # move ground truth data to cpu and convert to numpy array
            y_batch_test = y_batch_test[0].cpu()
            y_batch_test = pd.DataFrame(y_batch_test.detach().numpy(), columns=["sleep_stage"])

            # determine prediction based on classification type
            if self.classification_type == "binary":
                y_pred[y_pred >= 0.5] = 1
                y_pred[y_pred < 0.5] = 0
            else:
                y_pred = np.argmax(y_pred, axis=1)

            # save predictions in dictionary
            pred_dict[subj_idx] = y_pred

            # calculate classification performance for each subject
            subj_score = dl_score(
                y_pred, y_batch_test, classification_type=self.classification_type, subject_id=subj_idx
            )
            score_dict[subj_idx] = subj_score

        score_mean = pd.DataFrame(score_dict).agg(["mean"], axis=1).T
        subject_results = pd.DataFrame(score_dict)

        return subject_results, score_mean, pred_dict

    def batch_loader(self, x_train, y_train):
        return list(x_train.split(self.batch_size)), list(y_train.split(self.batch_size))

    def _load_best_model(self, model):

        model.load_state_dict(
            torch.load(
                Path(__file__)
                .parents[4]
                .joinpath(
                    "exports/pickle_pipelines/lstm_"
                    + "_".join(self.modality)
                    + "_"
                    + self.dataset_name
                    + "_"
                    + self.classification_type
                )
            )
        )

        model.eval()

    def _load_empty_model(self, use_gpu, num_classes):
        return Model(
            num_classes=num_classes,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            use_gpu=use_gpu,
            dataset_name=self.dataset_name,
            modality=self.modality,
        )
