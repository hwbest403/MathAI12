from utils import  load_json

data_dict = load_json('answersheet.json')

for key, elem in data_dict.items():
    question = elem['question']
    equation = elem['equation']

    print('\n### Run Code ###')
    exec_vars = {}
    exec(equation, None, exec_vars)

    # print('\n### Execution Result ###')
    # print(exec_vars['final_result'])