from Seq2Seq.decoders.TransformerDecoder import TransformerDecoder
from Seq2Seq.decoders.LSTMDecoder import LSTMDecoder

Decoders_list = {
    "LSTM": LSTMDecoder,
    "Transformer": TransformerDecoder
}

__all__ = ["TransformerDecoder", "LSTMDecoder", "Decoders_list"]
