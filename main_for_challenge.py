from Graph2Tree_kor.mawps.src.train_and_evaluate import *
from Graph2Tree_kor.mawps.src.models import *
from Graph2Tree_kor.mawps.src.expressions_transfer import *
from Graph2Tree_kor.mawps.src.pre_data import *
import torch.optim
import warnings
import logging
from ai_challenge_2step_data.utils import CustomTokenizer, save_to_json
from ai_challenge_2step_data.data_utils import generate_code_from_binary
import pickle


warnings.filterwarnings(action='ignore')

path = 'ai_challenge_2step_data/data/'
model_path = 'Graph2Tree_kor/mawps/model_traintest/'

logger = logging.getLogger(__name__)

def read_pickle(data_file):
    with open(model_path+data_file, 'rb') as f:
        data = pickle.load(f)
    return data

batch_size = 32
embedding_size = 128
hidden_size = 512
n_epochs = 80
learning_rate = (1e-3)
weight_decay = 1e-5
beam_size = 5
n_layers = 2
copy_nums = 10

def interactive():
    try:
        with open('/home/agc2021/dataset/problemsheet.json', 'r', encoding='utf-8-sig') as f:
            test_dict = json.load(f)
    except:
        with open(path + 'problemsheet.json', 'r', encoding='utf-8-sig') as f:
            test_dict = json.load(f)

    tokenizer = CustomTokenizer()
    test = []
    for key, elem in test_dict.items():
        question = elem['question']
        output_text, data = tokenizer.tokenize(question)
        output_text = tokenizer.tokenize_for_train_v3(output_text)
        elem['src'] = output_text
        elem['Numbers']=data
        elem['group_num'] = tokenizer.get_group_num(output_text)
        elem['src_split'] = elem['src'].split()
        elem['Question_Num'] = key
        test.append(elem)

    pairs_tested = transfer_num_test(test)

    (generate_nums, copy_nums) = read_pickle('generate_nums.p')
    input_lang = read_pickle('input_lang.p')
    output_lang = read_pickle('output_lang.p')

    test_pairs = prepare_data_for_test(pairs_tested, input_lang, output_lang, tree=True)

    encoder = EncoderSeq(input_size=input_lang.n_words, embedding_size=embedding_size, hidden_size=hidden_size,
                 n_layers=n_layers)
    predict = Prediction(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                 input_size=len(generate_nums))
    generate = GenerateNode(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                    embedding_size=embedding_size)
    merge = Merge(hidden_size=hidden_size, embedding_size=embedding_size)
    # the embedding layer is  only for generated number embeddings, operators, and paddings

    # 모델 초기화
    encoder.load_state_dict(torch.load(model_path+"encoder"))
    predict.load_state_dict(torch.load(model_path+"predict"))
    generate.load_state_dict(torch.load(model_path+"generate"))
    merge.load_state_dict(torch.load(model_path+"merge"))


    generate_num_ids = []
    for num in generate_nums:
        generate_num_ids.append(output_lang.word2index[num])

    if USE_CUDA:
        encoder.cuda()
        predict.cuda()
        generate.cuda()
        merge.cuda()
    answer = {}
    for test_batch in test_pairs:
        # print(test_batch)
        batch_graph = get_single_example_graph(test_batch[0], test_batch[1], test_batch[7], test_batch[4],
                                               test_batch[5])
        test_res = evaluate_tree(test_batch[0], test_batch[1], generate_num_ids, encoder, predict, generate,
                                 merge, output_lang, test_batch[5], batch_graph, beam_size=beam_size)

        prediction=out_expression_list(test_res, output_lang, test_batch[4])
        elem={}
        try:
            print("numbers: ", test_batch[8])
            print(f'\npredicted : {prediction}')
            binary_list = prediction
            output_codes = generate_code_from_binary(binary_list, var_dict=test_batch[8])

            print('\n### Final Code ###')
            run_code = '\n'.join(output_codes)
            # print(run_code)

            print('\n### Run Code ###')
            exec_vars = {}
            exec(run_code, None, exec_vars)

            print('\n### Execution Result ###')
            print(exec_vars['final_result'])
            elem['answer'] = f"{exec_vars['final_result']}"
            elem['equation'] = run_code

        except:
            elem['answer'] = ''
            elem['equation'] = 'print()'
        answer[test_batch[9]]=elem


    print("------------------------------------------------------")
    with open('./answersheet.json', 'w', encoding='utf-8') as fw:
        json.dump(answer, fw, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    interactive()
