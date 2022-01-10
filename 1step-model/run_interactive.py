import json
import os
import torch
import re
import logging
from attrdict import AttrDict
import pickle
import argparse

from Seq2Seq.modules.utils import set_seeds, load_device
from Seq2Seq.build_model import build_model

from run_test import translate_LSTM_beam_search, translate_Transformer_beam_search

logger = logging.getLogger(__name__)


def preprocessing(sent):
    '''
    input 문장을 학습 데이터와 같은 포맷으로 바꿔주기 위한 메서드.
    데이터에 맞게 수정할 것.
    아래 코드는 N2W 데이터를 위한 코드
    '''
    src = ''
    for i in range(len(sent)):
        if re.search(r'\s{1,}', sent[i]):
            src += ' _ '
        else:
            src += sent[i] + ' '
    src = src.rstrip()
    return src


def token2idx(src, vocab, device):
    '''
    적절한 토큰 단위로 preprocessing 한 input 데이터를 
    모델에 넣기 위하여 idx로 변환하는 메서드. 
    '''
    tokens = src.split()
    tokens = ['<sos>'] + tokens + ['<eos>']
    src_indexes = [vocab.stoi[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    return src_tensor


def interactive(opt):
    model_args_path = os.path.join('Seq2Seq/config', opt.model_type + '.json')
    test_args_path = os.path.join('Seq2Seq/config', 'test_config.json')

    with open(model_args_path, 'r', encoding='utf-8') as f:
        model_args = AttrDict(json.load(f))
    with open(test_args_path, 'r', encoding='utf-8') as f:
        test_args = AttrDict(json.load(f))

    set_seeds(0)
    device = load_device(test_args)
    logger.info('model loading...')

    vocabs = pickle.load(open(os.path.join(test_args.save_model_dir, f'vocabs.pkl'), 'rb'))
    model = build_model(model_args, opt.model_type, vocabs, device).to(device)

    checkpoint = torch.load(f'{test_args.save_model_dir}/model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    while True:
        sent = input('문장을 입력하세요: ')
        sent = preprocessing(sent)
        src_tensor = token2idx(sent, vocabs['src'], device)
        if opt.model_type == 'LSTM':
            src_tensor = torch.reshape(src_tensor, (-1, 1))
            pred = translate_LSTM_beam_search(
                src_tensor, vocabs, model, device, test_args.beam_size, test_args.max_decode_len)
        elif opt.model_type == 'Transformer':
            pred = translate_Transformer_beam_search(
                src_tensor, vocabs, model, device, test_args.beam_size, test_args.max_decode_len)
        pred = pred[0][0]
        pred = ' '.join(pred)
        print(pred)
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='LSTM', required=True, type=str, help='LSTM or Transformer')
    opt = parser.parse_args()
    interactive(opt)
