from utils import *
from data_utils import *

file = 'saved_models/mathko/test_out_beam_3.json'
test_file = 'data/mathko/test.json'

tokenizer = CustomTokenizer(verbose=False)

predicted_data = load_json(file)
test_data = load_json(test_file)

assert len(predicted_data) == len(test_data)

outputs = []
correct_count = 0
answer_correct_count = 0

for pred_elem, test_elem in zip(predicted_data, test_data):

    if pred_elem['predicted'] == pred_elem['trg']:
        correct_count += 1
        continue

    answer = test_elem['Answer']
    input_text = test_elem['Question']
    output_text, data = tokenizer.tokenize(input_text)

    print('\n========================================\n')
    print(f'Question : {input_text}')
    print(f'QuestionConv : {output_text}')
    print(f"\ntarget : {pred_elem['trg']}")
    print(f"predicted : {pred_elem['predicted']}")


    predicted = pred_elem['predicted']
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

total = len(predicted_data)
print('\n\n=== Final Result ===')
print(f'total : {total}')
print(f'equation correct : {correct_count}')
print(f'answer correct : {correct_count + answer_correct_count}')
print(f'equation accuracy : {correct_count / total:.3f}')
print(f'answer accuracy : {(correct_count + answer_correct_count) / total:.3f}')
