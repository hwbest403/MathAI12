import json
import os
import torch
import logging
from attrdict import AttrDict
import pickle
import sys

from Seq2Seq.modules.utils import set_seeds, load_device
from Seq2Seq.build_model import build_model

from run_test import translate_Transformer_beam_search
from utils import load_json, save_to_json
from data_utils import *

from utils import CustomTokenizer

try:
    print(sys.argv[1])
    # data_path = 'data/problemsheet_sample.json'
    data_path = 'data/test_sheet.json'
except:
    data_path = '/home/agc2021/dataset/problemsheet.json'

logger = logging.getLogger(__name__)


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


def interactive():
    model_type = 'Transformer'
    model_args_path = os.path.join('Seq2Seq/config', model_type + '.json')
    test_args_path = os.path.join('Seq2Seq/config', 'test_config.json')

    with open(model_args_path, 'r', encoding='utf-8') as f:
        model_args = AttrDict(json.load(f))
    with open(test_args_path, 'r', encoding='utf-8') as f:
        test_args = AttrDict(json.load(f))

    set_seeds(0)
    device = load_device(test_args)
    logger.info('model loading...')

    vocabs = pickle.load(open(os.path.join(test_args.save_model_dir, f'vocabs.pkl'), 'rb'))
    model = build_model(model_args, model_type, vocabs, device).to(device)

    checkpoint = torch.load(f'{test_args.save_model_dir}/model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    data_dict = load_json(data_path)

    tokenizer = CustomTokenizer()

    for key, elem in data_dict.items():
        question = elem['question']

        output_text, data = tokenizer.tokenize(question)
        output_text = tokenizer.tokenize_for_train(output_text)

        src_tensor = token2idx(output_text, vocabs['src'], device)
        pred = translate_Transformer_beam_search(
            src_tensor, vocabs, model, device, test_args.beam_size, test_args.max_decode_len)
        pred = pred[0][0]
        pred = ' '.join(pred)

        try:
            print(f'\npredicted : {pred}')
            binary_list = pred.split()

            arg_list = []
            for node in binary_list:
                if not node.startswith('func_') and node not in reserved_args:
                    arg_list.append(node)

            my_args = set(arg_list)
            print(f'my_args : {my_args}')
            print(f'data : {data}')

            for my_arg in my_args:
                if my_arg not in data:
                    print(f'finding... "{my_arg}"')
                    if my_arg == 'rep':
                        value = find_arg_rep(data)
                    else:
                        value = find_arg(my_arg, data)
                    print(f'got : {value}')
                    data[my_arg] = value

            for my_arg in my_args:
                if type(data[my_arg]) == list:
                    if 'rep' in data:
                        if type(data['rep']) != list:
                            data['rep'] = [data['rep']] * len(data[my_arg])

            print(f'data(converted) : {data}')

            output_codes = generate_code_from_binary(binary_list, var_dict=data)

            print('\n### Final Code ###')
            run_code = '\n'.join(output_codes)
            # print(run_code)

            print('\n### Run Code ###')
            exec_vars = {}
            exec(run_code, None, exec_vars)

            print('\n### Execution Result ###')
            print(exec_vars['final_result'])

            elem['equation'] = run_code
            elem['answer'] = f"{exec_vars['final_result']}"

        except:
            elem['equation'] = 'print()'
            elem['answer'] = ''


    # print(data_dict)
    save_to_json(data_dict, 'answersheet.json')


if __name__ == '__main__':
    interactive()
