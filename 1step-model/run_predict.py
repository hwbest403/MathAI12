import os
import torch
import logging
from attrdict import AttrDict

from Seq2Seq.modules.utils import set_seeds, load_device
from Seq2Seq.build_model import build_model

from run_test import translate_Transformer_beam_search
from utils import *
from data_utils import *

from utils import CustomTokenizer

data_path = 'data/mathko/test_1.json'

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

    data_list = load_json(data_path)

    tokenizer = CustomTokenizer()

    correct_count = 0
    answer_correct_count = 0

    for elem in data_list:
        question = elem['Question']
        equation = elem['EquationConv']
        answer = elem['Answer']

        output_text, data = tokenizer.tokenize(question)
        model_input = tokenizer.tokenize_for_train(output_text)

        src_tensor = token2idx(model_input, vocabs['src'], device)
        pred = translate_Transformer_beam_search(
            src_tensor, vocabs, model, device, test_args.beam_size, test_args.max_decode_len)
        pred = pred[0][0]
        predicted = ' '.join(pred)

        if predicted == equation:
            correct_count += 1
            continue

        print('\n========================================\n')
        print(f'Question : {question}')
        print(f'QuestionConv : {output_text}')
        print(f"\ntarget : {equation}")
        print(f"predicted : {predicted}")

        binary_list = predicted.split()

        arg_list = []
        for node in binary_list:
            if not node.startswith('func_') and node not in reserved_args:
                arg_list.append(node)

        my_args = set(arg_list)
        print(f'\nmy_args : {my_args}')
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

        try:
            output_codes = generate_code_from_binary(binary_list, var_dict=data)
        except:
            print('\nError : Can not generate Python code')
            continue

        try:
            print('\n### Execution Result ###')
            run_code = '\n'.join(output_codes)
            exec_vars = {}
            exec(run_code, None, exec_vars)

            if answer == exec_vars['final_result']:
                answer_correct_count += 1
                print('\nSuccess : The answer is same!')
            else:
                print(f'\nFail : The answer is "{answer}"')
        except:
            print('\nError : Can not run Python code')
            continue

    total = len(data_list)
    print('\n\n=== Final Result ===')
    print(f'total : {total}')
    print(f'equation correct : {correct_count}')
    print(f'answer correct : {correct_count + answer_correct_count}')
    print(f'equation accuracy : {correct_count / total:.3f}')
    print(f'answer accuracy : {(correct_count + answer_correct_count) / total:.3f}')

if __name__ == '__main__':
    interactive()
