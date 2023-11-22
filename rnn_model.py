import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.gru = nn.GRU(input_size, 
                          hidden_size, 
                          num_layers, 
                          dropout=dropout, 
                          batch_first=True)
        self.readout = nn.Linear(hidden_size, 2)

    def forward(self, x):
        # pass through GRU
        x, _ = self.gru(x)
        # turn last layers outputs into behavior predictions
        x = self.readout(x)
        return x