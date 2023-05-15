import argparse
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss

from sleep_analysis.classification.deep_learning.dl_scoring import dl_score, tensor_to_performance
from sleep_analysis.classification.deep_learning.tcnn.tcn_lib import TemporalConvNet


class TcnMain:
    def __init__(
        self,
        num_inputs,
        output_size,
        num_chanels,
        kernel_size,
        dropout,
        learning_rate,
        batch_size,
        modality,
        dataset_name,
        classification_type,
    ):
        torch.manual_seed(seed=42)
        self.model = TemporalConvNet(
            num_inputs=num_inputs,
            output_size=output_size,
            num_channels=num_chanels,
            dropout=dropout,
            kernel_size=kernel_size,
            modality=modality,
            dataset_name=dataset_name,
        )
        self.classification_type = classification_type
        self.lr = learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.modality = modality
        self.cuda = False

        # set loss function based on classification type
        if self.classification_type == "binary":
            self.criterion = BCEWithLogitsLoss()  # binary cross-entropy loss for sleep/wake classification
        else:
            self.criterion = CrossEntropyLoss()  # multi-class loss for multiclass sleep stage classification

        # If cuda is available, use it!
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.device = "cuda"
        else:
            self.device = "cpu"

    def train(self, x_train, y_train, x_val, y_val, num_epochs):

        # get batches from train data
        x_batch_train_list, y_batch_train_list = self.batch_loader(x_train, y_train)

        # initialize variables to determine and save best model in val-procedure
        min_val_loss = 999999999
        max_val_performance = 0.0

        # start training
        for epoch in range(num_epochs):

            train_losses = []
            # set model to training mode
            self.model.train()

            # iterate over the batches
            for x_batch_train, y_batch_train in zip(x_batch_train_list, y_batch_train_list):

                if torch.cuda.is_available():
                    x_batch_train = x_batch_train.cuda()
                    y_batch_train = y_batch_train.cuda()

                output = self.model(x_batch_train)
                self.optimizer.zero_grad()

                if self.classification_type == "binary":
                    loss = self.criterion(output, y_batch_train)
                else:
                    loss = self.criterion(output, torch.squeeze(y_batch_train).long())

                loss.backward()

                self.optimizer.step()
                train_losses.append(loss.item())

            # change this line dependent on how often validation loss should be calculated
            if epoch % 1 == 0:
                print("-------------------------")
                print(datetime.datetime.now())
                print("-------------------------")
                print("Epoch: %d, loss: %1.5f" % (epoch, np.mean(train_losses)))
                val_losses = []
                val_performances = []
                self.model.eval()

                # get batches from validation data
                x_batch_val_list, y_batch_val_list = self.batch_loader(x_val, y_val)

                # iterate over the batches
                for x_batch_val, y_batch_val in zip(x_batch_val_list, y_batch_val_list):

                    if torch.cuda.is_available():
                        x_batch_val = x_batch_val.to(device=self.device)
                        y_batch_val = y_batch_val.to(device=self.device)

                    y_pred = self.model(x_batch_val)

                    # calculate loss of batch-wise prediction
                    if self.classification_type == "binary":
                        val_loss = self.criterion(y_pred, y_batch_val)
                    else:
                        val_loss = self.criterion(y_pred, torch.squeeze(y_batch_val).long())

                    # append batch-wise loss to corresponding list
                    val_losses.append(val_loss.item())

                    # calculate performance of validation batch
                    class_performance = tensor_to_performance(
                        y_batch_val, y_pred, classification_type=self.classification_type
                    )

                    # append performance of batch-wise prediction to corresponding list
                    val_performances.append(class_performance["mcc"])

                mean_val_loss = np.mean(val_losses)
                mean_performance = np.mean(val_performances)

                print("validation_loss: " + str(mean_val_loss))
                print("validation_performance: " + str(mean_performance))
                print("-------------------------")

                # break if validation loss does not decrease anymore
                if (
                    ((mean_val_loss - min_val_loss) > 0.03 or mean_val_loss == min_val_loss)
                    and epoch > 10
                    or (mean_performance == 0.0 and epoch > 10)
                    # or mean_val_loss == last_loss
                ):
                    print("break because of bad loss development")
                    break

                # save model if validation loss is at a new minimum
                if min_val_loss > np.mean(val_losses):
                    min_val_loss = np.mean(val_losses)
                    max_val_performance = mean_performance

                    print("*************************")
                    print("new best validation_loss: " + str(np.mean(val_losses)))
                    print("validation performance " + str(np.mean(mean_performance)))
                    print("*************************")

                    # save model
                    torch.save(
                        self.model.state_dict(),
                        Path(__file__)
                        .parents[4]
                        .joinpath(
                            "exports/pickle_pipelines/tcn_"
                            + "_".join(self.modality)
                            + "_"
                            + self.dataset_name
                            + "_"
                            + self.classification_type
                        ),
                    )

        return max_val_performance

    def test(self, x_test, y_test):
        score_dict = {}

        # if cuda is available, use it! and set model to evaluation mode
        if self.cuda:

            # load best model from training
            self.model.load_state_dict(
                torch.load(
                    Path(__file__)
                    .parents[4]
                    .joinpath(
                        "exports/pickle_pipelines/tcn_"
                        + "_".join(self.modality)
                        + "_"
                        + self.dataset_name
                        + "_"
                        + self.classification_type
                    )
                )
            )
            self.model.eval()

            self.model = self.model.cuda()

            # iterate over test batches
            for i, (x_batch_test, y_true) in enumerate(zip(x_test, y_test)):

                # subj_id is stored in list element 1
                subj_idx = x_batch_test[1]
                # batch data is stored in list element 0
                x_batch_test = x_batch_test[0].cuda()

                # get predictions
                y_pred = self.model(x_batch_test).cpu()
                y_pred = y_pred.detach().numpy()

                # get true labels
                y_true = y_true[0].cpu()
                y_true = pd.DataFrame(y_true.detach().numpy(), columns=["sleep_stage"])

                # convert predictions to labels
                y_pred = self.pred_to_label(y_pred)

                # compute performance
                subj_score = dl_score(y_pred, y_true, classification_type=self.classification_type, subject_id=subj_idx)

                # store performance in dictionary
                score_dict[subj_idx] = subj_score

        # if cuda is not available, use cpu
        else:

            # load best model from training and set model to evaluation mode
            self.model.load_state_dict(
                torch.load(
                    Path(__file__)
                    .parents[4]
                    .joinpath(
                        "exports/pickle_pipelines/tcn_"
                        + "_".join(self.modality)
                        + "_"
                        + self.dataset_name
                        + "_"
                        + self.classification_type
                    ),
                )
            )
            self.model.eval()

            # iterate over test batches
            for i, (x_batch_test, y_true) in enumerate(zip(x_test, y_test)):

                # subj_id is stored in list element 1
                subj_idx = x_batch_test[1]
                # batch data is stored in list element 0
                y_pred = self.model(x_batch_test[0])

                # get predictions
                y_pred = y_pred.detach().numpy()
                # get true labels
                y_true = y_true[0]
                y_true = pd.DataFrame(y_true.detach().numpy(), columns=["sleep_stage"])

                # convert predictions to labels
                y_pred = self.pred_to_label(y_pred)

                # compute performance
                subj_score = dl_score(y_pred, y_true, classification_type=self.classification_type, subject_id=subj_idx)

                # store performance in dictionary
                score_dict[subj_idx] = subj_score

        # create dataframe from score_dict
        subject_results = pd.DataFrame(score_dict)

        # drop 'confusion matrix' from score_dict
        score_dict = {k: v for k, v in score_dict.items() if k != "confusion_matrix"}

        # compute mean performance to directly print it
        score_mean = pd.DataFrame(score_dict).agg(["mean"], axis=1).T
        print(score_mean.T)

        return subject_results, score_mean

    def batch_loader(self, x_train, y_train):
        return list(x_train.split(self.batch_size)), list(y_train.split(self.batch_size))

    def pred_to_label(self, y_pred):
        if self.classification_type == "binary":
            y_pred = pd.DataFrame(y_pred, columns=["sleep_stage"])
            y_pred["sleep_stage"] = y_pred["sleep_stage"].apply(lambda x: 1 if x > 0.5 else 0)
        else:
            y_pred = np.argmax(y_pred, axis=1)
        return y_pred
