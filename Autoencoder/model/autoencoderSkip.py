import torch
import torch.nn as nn

class AutoencoderSkip(nn.Module):
    def __init__(self, n_params: int, has_bias=False):
        super(AutoencoderSkip, self).__init__()

        #Encoders
        self.encoder1 = nn.Sequential(
            nn.Linear(n_params, int(n_params*2.5)),
            #nn.Dropout(0.01),
            nn.LeakyReLU()
        )
        self.encoder2 =nn.Sequential(
            nn.Linear(int(n_params*2.5), int(n_params*3.0)),
            #nn.Dropout(0.01),
            nn.LeakyReLU()
        )
        self.encoder3 =nn.Sequential(
            nn.Linear(int(n_params*3.0), int(n_params*3.5)),
            #nn.Dropout(0.01),
            nn.LeakyReLU()
        )
        self.encoder4 =nn.Sequential(
            nn.Linear(int(n_params*3.5), int(n_params*4.0)),
            #nn.Dropout(0.01),
            nn.LeakyReLU()
        )
        self.encoder5 =nn.Sequential(
            nn.Linear(int(n_params*4.0), int(n_params*5.0)),
            #nn.Dropout(0.1)     
        )
        

        #Decoders
        self.decoder1 = nn.Sequential(
            nn.Linear(int(n_params*5.0), int(n_params*4.0), bias=has_bias),
            #nn.Dropout(0.01),
            #nn.LeakyReLU()
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(int(n_params*4.0), int(n_params*3.5), bias=has_bias),
            #nn.Dropout(0.01),
            nn.LeakyReLU()
        )
        
        self.decoder3 = nn.Sequential(
            nn.Linear(int(n_params*3.5), int(n_params*3.0), bias=has_bias),
            #nn.Dropout(0.01),
            nn.LeakyReLU()
        )
        self.decoder4 = nn.Sequential(
            nn.Linear(int(n_params*3.0), int(n_params*2.5), bias=has_bias),
            #nn.Dropout(0.01),
            nn.LeakyReLU()
        )
        self.decoder5 = nn.Sequential(
            nn.Linear(int(n_params*2.5), n_params, bias=has_bias),
            #nn.Dropout(0.01),
            #nn.LeakyReLU()
        )

    def forward(self, x, *args):
        #Encoder
        x = self.encoder1(x)
        x2 = self.encoder2(x) #1-skip
        x = self.encoder3(x2)
        x4 = self.encoder4(x) #2-skip
        x = self.encoder5(x4)
        #Decoder
        x = self.decoder1(x)
        x = self.decoder2(x + x4) #2-skip
        x = nn.LeakyReLU()(x)
        x = self.decoder3(x)
        x = self.decoder4(x + x2) #1-skip
        x = nn.LeakyReLU()(x)
        x = self.decoder5(x)
        return x

def generar_modelo(n_features: int, hyper_parametros, device):
    model = AutoencoderSkip(n_features).double()
    model.to(device)

    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=hyper_parametros["learning_rate"], 
        weight_decay=hyper_parametros["weight_decay"]
    )

    return model, criterion, optimizer