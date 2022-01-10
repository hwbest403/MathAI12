from Seq2Seq.encoders.TransformerEncoder import TransformerEncoder
from Seq2Seq.encoders.LSTMEncoder import LSTMEncoder

Encoders_list = {
    "LSTM": LSTMEncoder,
    "Transformer": TransformerEncoder
}

__all__ = ["TransformerEncoder", "LSTMEncoder", "Encoders_list"]
