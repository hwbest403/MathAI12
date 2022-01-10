from attrdict import AttrDict
import json
import argparse
import os

from Seq2Seq.modules.utils import set_seeds, load_device, init_logger
from Seq2Seq.build_model import build_model
from Seq2Seq.data_loader import build_dataset
from Seq2Seq.Train import Trainer


def main(opt):
    model_args_path = os.path.join('Seq2Seq/config', opt.model_type + '.json')
    train_args_path = os.path.join('Seq2Seq/config', 'train_config.json')

    with open(model_args_path, 'r', encoding='utf-8') as f:
        model_args = AttrDict(json.load(f))
    with open(train_args_path, 'r', encoding='utf-8') as f:
        train_args = AttrDict(json.load(f))

    init_logger()
    set_seeds(0)
    device = load_device(train_args)

    (train_iter, val_iter, _), vocabs = build_dataset(opt.model_type, train_args, device, model_args.vectors)
    model = build_model(model_args, opt.model_type, vocabs, device)
    model = model.to(device)
    trainer = Trainer(train_args, opt, model, vocabs)
    trainer.train(train_iter, val_iter)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='Transformer', required=True, type=str, help='LSTM or Transformer')
    opt = parser.parse_args()
    main(opt)
