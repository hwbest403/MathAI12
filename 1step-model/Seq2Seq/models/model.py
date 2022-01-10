import torch.nn as nn


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, predict):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.predict = predict

    def forward(self, src, trg):
        encoder_outputs = self.encoder(src)
        output = self.decoder(trg, encoder_outputs, src=src, predict=self.predict)
        return output
