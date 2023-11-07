import torch
import torch.nn as nn



class RNNMODEL(nn.Module):
    def __init__(self, input_size = None, hidden_size=None, num_layers=None, readout_size=2, dropout = 0.1, gru_kwargs={}, readout_kwargs={}):
        super(RNNMODEL,self).__init__()
        params = [input_size, hidden_size, num_layers]
        if None in params:
            raise Exception
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.readout_size = readout_size
        self.gru_kwargs = gru_kwargs
        self.readout_kwargs = readout_kwargs
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=dropout,
            **gru_kwargs
        )
        self.readout = nn.Linear(in_features=self.hidden_size, out_features=self.readout_size)


    def forward(self, x):

        gru_out, _ = self.gru(x)
        readout_out = self.readout(gru_out)
        return readout_out


    def load_model(self, filepath):
        self.load_state_dict(torch.load(filepath))
        self.eval()  # Set the model to evaluation mode if needed


