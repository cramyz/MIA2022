import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, n_params: int, n_bottleneck: int):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_params, 200),
            nn.Tanh(),
            nn.Linear(200, 80),
            nn.Tanh(),
            nn.Linear(80, 70),
            nn.Tanh(),
            nn.Linear(70, 50),
            nn.Tanh(),
            nn.Linear(50, 29),
            nn.Tanh(),
            nn.Linear(29, n_bottleneck)
            )
        
        self.decoder = nn.Sequential(
            nn.Linear(n_bottleneck, 29),
            nn.Tanh(),
            nn.Linear(29, 50),
            nn.Tanh(),
            nn.Linear(50, 70),
            nn.Tanh(),
            nn.Linear(70, 80),
            nn.Tanh(),
            nn.Linear(80, 200),
            nn.Tanh(),
            nn.Linear(200, n_params),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def generar_modelo(n_features: int, hyper_parametros, device):
    model = Autoencoder(n_features, n_features//hyper_parametros["divisor"]).double()
    model.to(device)

    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=hyper_parametros["learning_rate"], 
        weight_decay=hyper_parametros["weight_decay"]
    )

    return model, criterion, optimizer