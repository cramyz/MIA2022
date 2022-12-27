import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, n_params: int):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_params, int(n_params*2.5)),
            #nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Linear(int(n_params*2.5), max(2, int(n_params*3.0))),
            #nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Linear(int(n_params*3.0), max(2, int(n_params*3.5))),
            #nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Linear(int(n_params*3.5), max(2, int(n_params*4.0))),
            #nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Linear(int(n_params*4.0), max(2, int(n_params*5.0))),
            #nn.Dropout(0.1)     
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(max(2, int(n_params*5.0)), int(n_params*4.0), bias=False),
            #nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Linear(max(2, int(n_params*4.0)), int(n_params*3.5), bias=False),
            #nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Linear(max(2, int(n_params*3.5)), int(n_params*3.0), bias=False),
            #nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Linear(max(2, int(n_params*3.0)), int(n_params*2.5), bias=False),
            #nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Linear(int(n_params*2.5), n_params, bias=False),
            #nn.Dropout(0.1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def generar_modelo(n_features: int, hyper_parametros, device):
    model = Autoencoder(n_features).double()
    model.to(device)

    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=hyper_parametros["learning_rate"], 
        weight_decay=hyper_parametros["weight_decay"]
    )

    return model, criterion, optimizer