import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, encoded_space_dim, input_dim):
        super().__init__()

        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 5, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 5, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 5, stride=2, padding=0),
            nn.ReLU(True)
        )

        self.flatten = nn.Flatten(start_dim=1)

        self.encoder_lin = nn.Sequential(
            nn.Linear(30 * 8 * 32, input_dim),
            nn.ReLU(True),
            nn.Linear(input_dim, encoded_space_dim)
        )
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x


class Decoder(nn.Module):
    def __init__(self, encoded_space_dim, input_dim):
        super().__init__()

        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, input_dim),
            nn.ReLU(True),
            nn.Linear(input_dim, 30 * 8 * 32),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 8, 30))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 5, stride=2, padding=0, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 5, stride=2, padding=1, output_padding=1)
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x
    
class DAE(nn.Module):
    def __init__(self, encoded_space_dim, input_dim):
        super().__init__()
        self.encoder = Encoder(encoded_space_dim, input_dim)
        self.decoder = Decoder(encoded_space_dim, input_dim)       
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x): 
        return self.decoder(self.encoder(x))

def build_model(args):
    model = DAE(args.encoded_space_dim, args.input_dim)
    criterion = nn.MSELoss()
    return model, criterion
