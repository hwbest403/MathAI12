import json
import random
from utils import CustomTokenizer, save_to_json, load_json, basic_tokenizer
from data_utils import convert_func_to_binary

tokenizer = CustomTokenizer()

files = {
    '1': ['1.json'],
    '2': ['2.json'],
    '3': ['3.json', '3_ext.json'],
    '4': ['4.json', '4_ext.json'],
    '6': ['6.json', '6_ext.json'],
    '7': ['7.json'],
    '8': ['8.json', '8_ext.json']
}

random.seed(10)


def split_data(train_data):
    data_list = []

    count = round(len(train_data) * 0.8)

    for elem in train_data:
        output = {}

        question = elem['Question']
        question_conv = elem['QuestionConv']
        var_dict = elem['Numbers']
        equation = elem['Equation']
        answer = elem['Answer']

        # new_elem = {}
        elem['QuestionConv'] = basic_tokenizer(question_conv)
        print(elem['QuestionConv'])
        elem['EquationConv'] = convert_func_to_binary(equation)
        print(elem['EquationConv'])

        output_text, data = tokenizer.tokenize(question)
        print(output_text)
        print(data)
        #elem['src']  = tokenizer.tokenize_for_train(output_text)
        # elem['src']  = tokenizer.tokenize_for_train_v2(output_text)
        elem['src']  = tokenizer.tokenize_for_train_v3(output_text)
        elem['group_num']  = tokenizer.get_group_num(elem['src'])

        elem['trg'] = elem['EquationConv']

        print(elem)

        data_list.append(elem)

    #train_list = data_list[:count]
    #test_list = data_list[count:]

    return data_list


def save_data(data_list, type='train'):
    with open(f'{type}.json', 'w', encoding='utf-8') as f:
        for line in data_list:
            f.write(line)


if __name__ == '__main__':
    train_list = []
    test_list = []
    data = []
    for key, file_list in files.items():
        data_list = []
        for file in file_list:
            data_list += load_json(file)

        print(f'Chapter {key} : {len(data_list)}')
        random.shuffle(data_list)
        spl= split_data(data_list)
        data += spl
        #train_list += train
        #test_list += test

        # 챕터별로 테스트 데이터 저장
        #save_to_json(test, f'test_{key}.json')

    # train, test 저장
    #print(f'len(train_list) : {len(train_list)}')
    #print(f'len(test_list) : {len(test_list)}')

    #save_to_json(train_list, 'train.json')
    #save_to_json(test_list, 'test.json')
    save_to_json(data, 'all.json')

    #
    # test sheet 형식의 파일 만들기

    output = {}
    for i, test_elem in enumerate(test_list):
        elem = {'question': test_elem['Question']}
        output[i + 1] = elem

    save_to_json(output, 'test_sheet.json')
