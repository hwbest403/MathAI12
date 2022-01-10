from Seq2Seq.encoders import Encoders_list
from Seq2Seq.decoders import Decoders_list
from Seq2Seq.models.model import Seq2Seq
from torch.nn.init import kaiming_normal_, xavier_normal_


def build_encoder(model_type, model_args, vocabs, device):
    return Encoders_list[model_type].from_args(model_args, vocabs, device)


def build_decoder(model_type, model_args, vocabs, device):
    return Decoders_list[model_type].from_args(model_args, vocabs, device)


def build_model(model_args, model_type, vocabs, device, predict=False):
    encoder = build_encoder(model_type, model_args, vocabs, device)
    decoder = build_decoder(model_type, model_args, vocabs, device)
    model = Seq2Seq(encoder=encoder, decoder=decoder, predict=predict)

    if model_args.param_init_he:
        for p in model.parameters():
            if p.dim() > 1:
                kaiming_normal_(p)

    if model_args.param_init_xavier:
        for p in model.parameters():
            if p.dim() > 1:
                xavier_normal_(p)
    return model
