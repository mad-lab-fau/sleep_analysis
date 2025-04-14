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

import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedFocalLoss(nn.Module):
    """
    Focal Loss with Class Weights for Multi-Class Classification.

    This loss function addresses class imbalance by down-weighting easy examples and focusing on
    hard-to-classify samples. Additionally, it incorporates class weights to balance label distributions.

    Attributes:
    - class_weights (torch.Tensor): Tensor of shape (num_classes,) containing weights for each class.
    - gamma (float): Focusing parameter to adjust loss weight based on prediction confidence (default=2.0).
    - reduction (str): Specifies the reduction method, either "mean" (default) or "sum".
    """

    def __init__(self, class_weights, gamma=2.0, reduction="mean"):
        """
        Initializes the WeightedFocalLoss function.

        Parameters:
        - class_weights (torch.Tensor): Class weights to handle imbalanced data.
        - gamma (float, optional): Focusing parameter (default=2.0).
        - reduction (str, optional): Reduction mode, either "mean" or "sum" (default="mean").
        """
        super(WeightedFocalLoss, self).__init__()

        # Ensure class weights do not contain NaN, Inf, or zero values
        self.class_weights = torch.nan_to_num(class_weights, nan=1.0, posinf=1.0, neginf=1.0)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Computes the Focal Loss with Class Weights.

        Parameters:
        - inputs (torch.Tensor): Model logits (before softmax), shape [batch_size, num_classes].
        - targets (torch.Tensor): Ground truth labels, shape [batch_size].

        Returns:
        - torch.Tensor: Computed focal loss.
        """

        # Compute log probabilities safely, avoiding log(0) which produces -inf
        log_probs = F.log_softmax(inputs, dim=-1).clamp(min=-100)

        # Compute softmax probabilities safely, avoiding exp overflow
        probs = torch.exp(log_probs).clamp(min=1e-8, max=1.0)

        # Select log probabilities and probabilities corresponding to the target labels
        log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        probs = probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

        # Compute the focal weight, reducing the impact of easy-to-classify examples
        focal_weight = (1 - probs) ** self.gamma

        # Apply class weighting
        alpha_weight = self.class_weights[targets]  # Extract class weight per sample
        focal_weight = focal_weight * alpha_weight  # Multiply with class weights

        # Compute final loss
        loss = -focal_weight * log_probs

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class LSTM:
    def __init__(
        self,
        num_epochs,
        learning_rate,
        input_size,
        hidden_size,
        num_layers,
        seq_len,
        dropout,
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
        self.dropout=dropout

        self.use_gpu = torch.cuda.is_available()
        self.batch_size = batch_size
        self.modality = modality
        self.dataset_name = dataset_name
        self.classification_type = classification_type

        if self.use_gpu:
            self.device = "cuda"
        else:
            self.device = "cpu"

    def train(self, x_train, y_train, x_val, y_val, retrain = False):
        # get number of classes
        num_classes = get_num_classes(self.classification_type)

        # load batched data
        x_batch_train_list, y_batch_train_list = self.batch_loader(x_train, y_train)

        # calculate class weights
        class_labels, class_sample_count = np.unique(y_train.detach().numpy(), return_counts=True)
        class_sample_count = class_sample_count[np.argsort(class_labels)]

        print(f"class_labels_count: {class_labels}")
        print(f"class_sample_count: {class_sample_count}")
        print(f"len(y_train): {len(y_train.detach().numpy())}")

        weight_classes = class_sample_count / len(y_train.detach().numpy())
        class_weights = torch.tensor(weight_classes).float().to(self.device)

        class_weights = 1 - class_weights

        print(f"class_weights: {class_weights}")

        # load empty model of our lstm class that needs to be trained
        lstm = self._load_empty_model(use_gpu=self.use_gpu, num_classes=num_classes, dropout=self.dropout, use_attention=False)

        if retrain:
            print("Finetune MESA model with Radar dataset")
            # load best model obtained from training
            self._load_best_mesa_model(lstm)
            lstm = lstm.cuda()

        if self.classification_type == "binary":
            print("set binary cross-entropy loss for binary classification", flush=True)
            criterion = torch.nn.BCEWithLogitsLoss()
        else:
            print("set focal loss for multi-class classification", flush=True)
            # If N1 & REM still underperform, tune gamma (try values like 1.5, 2.5).
            criterion = WeightedFocalLoss(class_weights=class_weights, gamma=2.0, reduction="mean")

        print("learning rate", self.learning_rate)
        print("batch size", self.batch_size)
        print("num layer", self.num_layers)
        print("hidden size", self.hidden_size)


        optimizer = torch.optim.Adam(lstm.parameters(), lr=self.learning_rate, weight_decay=1e-5)


        # initialize variables to determine and save best model in val-procedure
        min_val_loss = 9999999
        max_val_performance = 0.0

        patience_threshold = 5
        min_epochs = 5
        patience_counter = 0


        for epoch in range(self.num_epochs):
            train_losses = []

            # iterate over all batches of training data
            for x_batch_train, y_batch_train in zip(x_batch_train_list, y_batch_train_list):
                if self.use_gpu:
                    x_batch_train = x_batch_train.to(device=self.device)
                    y_batch_train = y_batch_train.to(device=self.device)

               # for name, param in lstm.named_parameters():
               #     if torch.isnan(param).any():
               #         print(f"[WARNING] NaN detected in {name}, reinitializing weights.")
               #         nn.init.uniform_(param, a=-0.1, b=0.1)  # Reinitialize to prevent training crash
                lstm.train()
                outputs = lstm.forward(x_batch_train)  # forward pass
                outputs = outputs.clamp(min=-10, max=10)


                optimizer.zero_grad()  # calculate the gradient, manually setting to 0

                if torch.isnan(outputs).any():
                    print("[ERROR] NaN detected in model output! Stopping training.")
                    exit()
                if torch.isnan(y_batch_train).any():
                    print("[ERROR] NaN detected in target labels! Stopping training.")
                    exit()

                # obtain the loss function
                if self.classification_type == "binary":
                    loss = criterion(outputs, y_batch_train)
                else:
                    #loss = criterion(outputs, torch.squeeze(y_batch_train).long())
                    loss = torch.nan_to_num(criterion(outputs, torch.squeeze(y_batch_train).long()), nan=0.0, posinf=1.0, neginf=-1.0)

                if torch.isnan(loss).any():
                    print("[ERROR] NaN detected in loss! Stopping training.")
                    exit()

                loss.backward()  # calculates the loss of the loss function

                #  Step 1: Detect and Reset NaN Gradients Before Optimizer Step
                for name, param in lstm.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any():
                            print(f"[WARNING] NaN detected in {name} gradients, resetting to zero.")
                            param.grad = torch.zeros_like(param.grad)  # Prevent NaN propagation

                #  Step 2: Add Small Gradient Noise to Attention Layer
                for name, param in lstm.named_parameters():
                    if "attention.attention_weights" in name and param.grad is not None:
                        noise = torch.randn_like(param.grad) * 1e-3  # Tiny noise to prevent zero-variance
                        param.grad += noise

                #  Step 3: Apply Even Stronger Gradient Clipping
                torch.nn.utils.clip_grad_norm_(lstm.parameters(), max_norm=0.5)  # Reduce gradient explosion

                #  Step 4: Use SGD with Momentum for the Attention Layer Only
                attention_params = [param for name, param in lstm.named_parameters() if "attention.attention_weights" in name]
                #attention_optimizer = torch.optim.SGD(attention_params, lr=0.001, momentum=0.9)

                # Update all parameters with the main optimizer
                optimizer.step()

                #  Step 4 (continued): Use SGD optimizer only for the attention layer
                #attention_optimizer.step()


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

                print(f"Validation Loss: {mean_val_loss:.5f}")
                print(f"Validation MCC: {mean_performance:.5f}")
                print("-------------------------")

                #  Overfitting Check: Stop if training loss is much lower than validation loss
                train_loss = np.mean(train_losses)
                if (train_loss - mean_val_loss) > 0.3:
                    print("[WARNING] Possible Overfitting Detected: Large gap between train and validation loss.")
                    patience_counter += 1

                #  Improved Early Stopping
                if mean_val_loss < min_val_loss:
                    min_val_loss = mean_val_loss
                    max_val_performance = mean_performance
                    patience_counter = 0  # Reset patience if improvement is found

                    print("*************************")
                    print(f"New Best Validation Loss: {mean_val_loss:.5f}")
                    print(f"Validation Performance (MCC): {mean_performance:.5f}")
                    print("*************************", flush=True)

                    #  Save best model
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

                else:
                    patience_counter += 1
                    print(f"[INFO] No Improvement. Patience: {patience_counter}/{patience_threshold}", flush=True)

                #  Define new stopping criteria
                if (
                        patience_counter >= patience_threshold  # Stop if patience runs out
                        or (epoch > min_epochs and mean_performance <= 0.0)  # Stop if MCC is bad
                ):
                    print("[STOPPING] Training stopped due to no improvement or bad MCC.", flush=True)
                    break

        return max_val_performance

    def test(self, x_test, y_test, retrain=False):
        score_dict = {}
        pred_dict = {}

        num_classes = get_num_classes(self.classification_type)

        if self.use_gpu:
            # load "empty" lstm model --> same parameters as in training
            lstm = self._load_empty_model(use_gpu=False, num_classes=num_classes, dropout=self.dropout, use_attention=False)

            # load best model obtained from training
            self._load_best_model(lstm)
            lstm = lstm.cuda()

        else:
            lstm = self._load_empty_model(use_gpu=False, num_classes=num_classes, dropout=self.dropout, use_attention=False)

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

            # safe sleep stage predictions with subject id to csv file for subsequent analysis
            if retrain:
                pd.DataFrame(y_pred).to_csv(Path(__file__)
                    .parents[4]
                    .joinpath(
                        "exports/results_per_subject/lstm", self.dataset_name, self.classification_type, "retrain").joinpath(str(subj_idx) +  ".csv"))
            else:
                pd.DataFrame(y_pred).to_csv(Path(__file__)
                    .parents[4]
                    .joinpath(
                        "exports/results_per_subject/lstm", self.dataset_name, self.classification_type ,"radar_only").joinpath(str(subj_idx) + ".csv"))

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

    def _load_best_mesa_model(self, model):
        model.load_state_dict(
            torch.load(
                Path(__file__)
                .parents[4]
                .joinpath(
                    "exports/pickle_pipelines/lstm_"
                    + "_".join(self.modality)
                    + "_"
                    + "MESA_Sleep"
                    + "_"
                    + self.classification_type
                )
            )
        )

    def _load_empty_model(self, use_gpu, num_classes, dropout, use_attention=False):
        return Model(
            num_classes=num_classes,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            use_gpu=use_gpu,
            dropout=dropout,
            dataset_name=self.dataset_name,
            modality=self.modality,
            use_attention=use_attention
        )
