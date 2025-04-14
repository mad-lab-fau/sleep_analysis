import random
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

# Debugging Function
def debug_tensor(tensor, name):
    """Prints debug info if NaN or Inf values are detected in a tensor."""
    if torch.isnan(tensor).any():
        print(f"[DEBUG] NaN detected in {name}")
    if torch.isinf(tensor).any():
        print(f"[DEBUG] Inf detected in {name}")

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention_weights = nn.Linear(hidden_size, 1, bias=False)
        self.norm = nn.LayerNorm(hidden_size)  # Helps prevent exploding/vanishing gradients
        self.dropout = nn.Dropout(0.1)  # Prevents overfitting to a small number of features

    def forward(self, lstm_output):
        lstm_output = self.norm(lstm_output)  # Normalize before attention
        attn_scores = self.attention_weights(lstm_output).squeeze(-1)  # Shape: (batch_size, seq_length)
        attn_scores = attn_scores - attn_scores.max(dim=1, keepdim=True)[0]  # Stability trick
        attn_scores = attn_scores.clamp(min=-10, max=10)  # Clip extreme values
        attn_weights = torch.softmax(attn_scores, dim=1)  # Apply softmax
        attn_weights = self.dropout(attn_weights)  # Apply dropout to stabilize training
        attended_output = torch.sum(attn_weights.unsqueeze(-1) * lstm_output, dim=1)  # Weighted sum
        return attended_output

# Main LSTM Model
class Model(nn.Module):
    def __init__(
        self, num_classes, input_size, hidden_size, num_layers, dropout, use_gpu, use_attention=True,
        dataset_name="dataset_name", modality="acc"
    ):
        super(Model, self).__init__()
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        random.seed(42)
        np.random.seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.modality = modality
        self.dataset_name = dataset_name
        self.use_attention = use_attention
        self.use_gpu = use_gpu

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        if self.use_attention:
            self.attention = Attention(hidden_size)  # Only create attention if enabled

        # Fully connected layers
        self.fc_1 = nn.Linear(hidden_size, 128)
        self.fc = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(self.dropout)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        # ✅ Move model to GPU if enabled
        if self.use_gpu:
            self.to("cuda")

        # Initialize Weights
        self.init_weights()

    def init_weights(self):
        """Initialize weights properly to prevent NaNs"""
        for name, param in self.named_parameters():
            if "weight_ih" in name:  # Input-to-hidden weights
                nn.init.uniform_(param, a=-0.1, b=0.1)  # Uniform range prevents NaNs
            elif "weight_hh" in name:  # Hidden-to-hidden weights
                nn.init.orthogonal_(param)  # Helps with stability
            elif "bias" in name:
                nn.init.constant_(param, 0)  # Biases set to zero

    def forward(self, x):

        #print("GPU is enabled") if self.use_gpu else print("GPU is disabled")

        x = x.cuda() if self.use_gpu else x

        device = x.device  # Get device of input tensor

        # ✅ Ensure model parameters are on the same device
        self.to(device)

        if torch.isnan(x).any():
            print("[DEBUG] NaN detected in input batch. Skipping this batch.")
            return torch.zeros(x.shape[0], self.num_classes, device=x.device)

        # Ensure input has correct dimensions
        if len(x.shape) == 2:
            x = x.reshape(x.shape[0], x.shape[1], 1)

        mean_x = x.mean(dim=(0, 1), keepdim=True)
        std_x = x.std(dim=(0, 1), keepdim=True) + 1e-8  # Avoid division by zero

        if torch.isnan(mean_x).any() or torch.isnan(std_x).any():
            print("[DEBUG] Skipping normalization due to NaN in batch statistics")
        else:
            x = (x - mean_x) / std_x

        debug_tensor(x, "Normalized Input x")

        # Initialize hidden and cell state
        if self.use_gpu:
            h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=device).cuda()
            c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=device).cuda()
        else:
            h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=device)
            c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=device)

        # Add small noise to hidden states to prevent instability
        h_0 += torch.randn_like(h_0) * 1e-3
        c_0 += torch.randn_like(c_0) * 1e-3

        lstm_out, _ = self.lstm(x, (h_0, c_0))  # LSTM output

        debug_tensor(lstm_out, "LSTM Output")

        if self.use_attention:
            attn_out = self.attention(lstm_out)  # Use attention if enabled
            debug_tensor(attn_out, "Attention Output")
        else:
            attn_out = lstm_out.mean(dim=1)  # Use mean pooling instead of attention
            debug_tensor(attn_out, "Mean Pooled Output (Attention Disabled)")

        out = self.relu(attn_out)
        out = self.fc_1(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc(out)  # Final classification layer

        debug_tensor(out, "Final Model Output")

        return out
