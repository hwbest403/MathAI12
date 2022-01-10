import json
import pickle
import re
from konlpy.tag import Komoran
import linecache
import sys


def print_exception(logger=None, exit=False):
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    line = line.strip()

    msg = 'EXCEPTION IN ({}, LINE: {}, CODE: {}): {} {}'.format(filename, lineno, line, exc_type, exc_obj)

    if logger:
        logger.error(msg)
    else:
        print(msg)

    if exit:
        sys.exit(1)

    return msg


def save_to_json(data, filename='data.json'):
    if filename[-4:] != 'json':
        filename += '.json'

    with open(f'{filename}', 'w', encoding='utf-8') as fw:
        json.dump(data, fw, indent=4, ensure_ascii=False)


def load_json(data_file):
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data


def save_vocab(vocab, path):
    output = open(path, 'wb')
    pickle.dump(vocab, output)
    output.close()


def load_vocab(path):
    import pickle

    with open(path, 'rb') as f:
        vocab = pickle.load(f)

    return vocab


def is_hangul(c):
    if ord('가') <= ord(c) <= ord('힣'):
        return True

    return False


def get_key(value, data_dict):
    for key, val in data_dict.items():
        if value == val:
            return key

    return None


def basic_tokenizer(text):
    text = text.replace(',', ' , ')
    text = text.replace('+', ' + ')
    text = text.replace('-', ' - ')
    text = text.replace('*', ' * ')
    text = text.replace('/', ' / ')
    text = ' '.join(text.split())

    prev_hangul = False
    prev_num = False

    output = []
    for char in text:
        if is_hangul(char):
            if not prev_hangul:
                output.append(' ')
            output.append(char)

            prev_hangul = True
            prev_num = False
        else:
            if char.isdigit():
                if prev_hangul:
                    output.append(' ')
                prev_num = True
            else:
                if prev_num and char != '.':
                    output.append(' ')
                    prev_num = False

                if not prev_num and char in ['.', '?']:
                    output.append(' ')

            output.append(char)
            prev_hangul = False

    result = ' '.join(''.join(output).split())

    return result


def convert_num(text):
    """
    숫자인 부분을 num 태그로 변환

    :param text:
    :return:
    """
    idx = 0

    words = text.split()
    data = {}
    outputs = []

    for word in words:
        try:
            value = float(word)

            if '.' not in word:
                value = int(word)

            key = f'num{idx}'
            data[key] = value
            outputs.append(key)

            idx += 1

        except ValueError:
            outputs.append(word)

    return ' '.join(outputs), data


class CustomTokenizer():
    def __init__(self, verbose=True):
        self.komoran = Komoran(userdic='userdict.txt')
        self.verbose = verbose

    def get_num_hanguls(self, text):
        """
        PoS Tagger를 이용하여 관형사 부분을 추출
        :param text:
        :return:
        """
        pos_results = self.komoran.pos(text)
        output = []
        for i, (key, val) in enumerate(pos_results):
            # print(key, val)

            if val == 'MM':
                next_val = pos_results[i + 1][1]
                if 'NN' in next_val:
                    output.append((pos_results[i][0], pos_results[i + 1][0]))

        return output

    def replace_hnagul_to_num(self, text, output):
        """
        추출된 관형사 목록을 이용하여 한글을 숫자로 변환
        :param text:
        :param output:
        :return:
        """
        convert_map = {
            '일의 자리': '1의 자리',
            '십의 자리': '10의 자리',
            '백의 자리': '100의 자리',
            '천의 자리': '1000의 자리',
            '만의 자리': '10000의 자리',
        }

        for key, val in convert_map.items():
            text = text.replace(key, val)

        num_hanguls = {
            '1': ['한'],
            '2': ['두', '둘'],
            '3': ['세', '셋'],
            '4': ['네', '넷'],
        }

        for i, elem in enumerate(output):
            num = None
            for key, hanguls in num_hanguls.items():
                if elem[0] in hanguls:
                    try:
                        if elem[0] == '한' and elem[1] == '답':  # 예외 : 처음 구하려고 한 답을 구해 보세요.
                            break
                        if elem[0] == '한' and output[i + 1][0] == '다른':  # 예외 : 필통 몇 개를 양팔 저울의 한 쪽에 올려놓고, 다른 쪽에는
                            break

                    except IndexError:
                        pass

                    num = key
                    break

            if num is None:
                continue

            old_text = f'{elem[0]} {elem[1]}'
            repl_text = f'{num} {elem[1]}'
            text = text.replace(old_text, repl_text)

        output_text = text

        return output_text

    def replace_place_name(self, text):
        places = {
            '(가)': 'pnm0',
            '(나)': 'pnm1',
            '(다)': 'pnm2',
            '(라)': 'pnm3',
            '(마)': 'pnm4',
            '(바)': 'pnm5',
            '(사)': 'pnm6',
            '(아)': 'pnm7',
            '(자)': 'pnm8',
            '(차)': 'pnm9'
        }

        for key, val in places.items():
            text = text.replace(key, val)

        return text

    def convert_color(self, text):
        colors = {
            '빨간색': ['빨간', '빨강', '빨간색'],
            '노란색': ['노란', '노랑', '노란색'],
            '흰색': ['흰', '흰색'],
            '검은색': ['검정', '검은', '검정색', '검은색'],
            '파란색': ['파랑', '파란', '파란색'],
            '초록색': ['초록', '초록색'],
            '남색': ['남색'],
            '보라색': ['보라', '보라색']
        }

        words = text.split()

        idx = 0
        data = {}

        for i, word in enumerate(words):
            for color, color_list in colors.items():
                if word in color_list:
                    key = f'col{idx}'
                    data[key] = word
                    words[i] = key

                    idx += 1

        return ' '.join(words), data

    def convert_shape(self, text):
        shapes = {
            '동그라미': ['동그라미', '동그란'],
            '네모': ['네모', '네모난'],
            '세모': ['세모', '세모난'],
            '타원': ['타원', '타원형']

        }

        words = text.split()

        idx = 0
        data = {}

        for i, word in enumerate(words):
            for color, color_list in shapes.items():
                if word in color_list:
                    key = f'sha{idx}'
                    data[key] = word
                    words[i] = key

                    idx += 1

        return ' '.join(words), data

    def convert_name(self, text):
        names = ['정국', '지민', '석진', '태형', '남준', '윤기', '호석', '민영', '유정', '은지', '유나']

        words = text.split()

        for i, word in enumerate(words):
            for name in names:
                if word.startswith(name):
                    if name != word:
                        words[i] = word.replace(name, f'{name} #')

        text = ' '.join(words)

        words = text.split()

        idx = 0
        data = {}


        for i, word in enumerate(words):
            if word in names:
                key = get_key(word, data)

                if key is None:
                    key = f'nae{idx}'
                    data[key] = word
                    idx += 1

                words[i] = key

        return ' '.join(words), data

    def convert_matched_value(self, text):
        """
        단순 매칭되는 값 변환
        :param text:
        :return:
        """
        data_dict = {
            'foo': ['사과', '복숭아', '오렌지', '참외', '바나나', '배', '감', '귤', '포도', '수박', '토마토', '무', '당근', '오이', '배추',
                 '사탕', '김밥', '빵', '라면', '과자', '음료수', '주스', '우유', '달걀','딸기','감귤','멜론','용과','두리안','리치','캔탈롭','파인애플','자두'],
            'sta': ['만년필', '연필', '색연필', '지우개', '공책', '도화지', '색종이', '풀', '테이프','볼펜', '샤프','싸인펜','삼색볼펜', '형광펜','펜','샤프펜슬','동화책','위인전'],
            'trn': ['비행기', '자동차', '트럭', '배', '자전거', '오토바이', '기차', '버스', '엘리베이터','캠핑카','승용차','버스'],
            'spt': ['배구공', '농구공', '축구공', '탁구공', '야구공', '줄넘기', '달리기', '수영', '시합','운동화','런닝화','축구화','풋살화','농구화'],
            'sbj': ['국어','영어','수학','사회','과학','음악','미술','체육','가정'],
            'plc': ['서점', '마트', '문구점', '집', '학교', '수영장'],
            'bld': ['페인트', '벽', '천장', '문', '울타리'],
            'ani': ['오리', '닭', '토끼', '물고기', '고래', '거위', '달팽이', '개구리', '강아지', '고양이', '비둘기', '병아리','돼지' ]
        }

        idx_dict = {}
        words = text.split()

        data = {}

        for i, word in enumerate(words):
            for key_code, values in data_dict.items():
                for value in values:
                    if len(value) == 1:
                        if word == value:
                            if i > 0 and i < len(words) - 1:
                                if words[i - 1] == ',' or words[i + 1] == ',':
                                    key = get_key(value, data)

                                    if key is None:
                                        if key_code not in idx_dict:
                                            idx_dict[key_code] = 0

                                        key = f'{key_code}{idx_dict[key_code]}'

                                        idx_dict[key_code] += 1

                                        words[i] = key

                                    data[key] = value
                            elif i == 0:
                                key = get_key(value, data)

                                if key is None:
                                    if key_code not in idx_dict:
                                        idx_dict[key_code] = 0

                                    key = f'{key_code}{idx_dict[key_code]}'

                                    idx_dict[key_code] += 1

                                    words[i] = key

                                data[key] = value


                    elif word.startswith(value):
                        key = get_key(value, data)

                        if key is None:
                            if key_code not in idx_dict:
                                idx_dict[key_code] = 0

                            key = f'{key_code}{idx_dict[key_code]}'

                            idx_dict[key_code] += 1

                        if word != value:
                            words[i] = word.replace(value, f'{key} #')
                        else:
                            words[i] = key

                        data[key] = value

        return ' '.join(words), data

    def convert_unk_opr(self, text):
        """
        미지수 및 연산기호 변환
        :param text:
        :return:
        """
        unks = ['A', 'B']
        oprs = ['+', '-', '*', '/']

        words = text.split()

        idx = 0
        idx2 = 0
        data = {}

        for i, word in enumerate(words):
            if word in unks:
                key = get_key(word, data)

                if key is None:
                    key = f'unk{idx}'
                    data[key] = word
                    idx += 1

                words[i] = key

            if word in oprs:
                key = get_key(word, data)

                if key is None:
                    key = f'opr{idx2}'
                    data[key] = word
                    idx2 += 1

                words[i] = key

        return ' '.join(words), data

    def convert_ord(self, text):
        """
        순서를 표시하는 단어 변환
        :param text:
        :return:
        """
        words = text.split()

        idx = 0
        data = {}

        for i, word in enumerate(words):
            if word.startswith('번째') or word.startswith('째'):
                try:
                    int(words[i - 1])  # "몇 번째"를 거르기 위함
                except ValueError:
                    continue

                key = f'ord{idx}'
                data[key] = int(words[i - 1])
                words[i - 1] = key

                idx += 1

        return ' '.join(words), data

    def convert_seq(self, text):
        """
        수열을 변환
        :param text:
        :return:
        """
        # pattern = r'\d+[ ]*[,]+[,\d ]+\d+'
        pattern = r'[AB\d]+[ ]*[,]+[AB,\d ]+[AB\d]+'

        results = re.findall(pattern, text)

        idx = 0
        data = {}

        for result in results:
            key = f'seq{idx}'

            output = []
            for x in result.split(' , '):
                try:
                    output.append(int(x))
                except ValueError:
                    output.append(x)

            data[key] = output
            text = text.replace(result, key)

            idx += 1

        return text, data

    def convert_dig(self, text):
        """
        자릿수 표시 변환
        :param text:
        :return:
        """
        words = text.split()

        idx = 0
        data = {}

        for i, word in enumerate(words):
            if word.startswith('자리'):
                try:
                    int(words[i - 1])  # "몇 번째"를 거르기 위함
                except ValueError:
                    continue

                key = f'dig{idx}'
                data[key] = int(words[i - 1])
                words[i - 1] = key

                idx += 1

        return ' '.join(words), data

    def convert_rep(self, text):
        """
        ~씩 과 같이 반복을 의미하는 단어 변환
        :param text:
        :return:
        """
        words = text.split()

        idx = 0
        data = {}

        for i, word in enumerate(words):
            if word.startswith('개씩') or word.startswith('장씩') or word.startswith('명씩'):
                try:
                    int(words[i - 1])  # "몇 번째"를 거르기 위함
                except ValueError:
                    continue

                key = f'rep{idx}'
                data[key] = int(words[i - 1])
                words[i - 1] = key

                idx += 1

        return ' '.join(words), data

    def tokenize(self, text):
        print(f'\noriginal : {text}') if self.verbose else None

        # 관형사를 한글로 변환
        output = self.get_num_hanguls(text)
        text = self.replace_hnagul_to_num(text, output)

        text = self.replace_place_name(text)

        text = basic_tokenizer(text)
        print(f'basic tokenizer : {text}') if self.verbose else None

        text, data_seq = self.convert_seq(text)  # 수열을 seq로 변환
        text, data_dig = self.convert_dig(text)  # 자릿수 표시를 dig로 변환
        text, data_rep = self.convert_rep(text)  # 반복의미를 rep로 변환
        text, data_ord = self.convert_ord(text)  # 순서표시를 ord로 변환
        text, data_col = self.convert_color(text)
        text, data_sha = self.convert_shape(text)
        text, data_nae = self.convert_name(text)
        text, data_matched = self.convert_matched_value(text)  # 단순매칭 변환
        text, data_unk = self.convert_unk_opr(text)  # 미지수(unk) 및 연산기호(opr) 변환
        text, data_num = convert_num(text)  # 숫자를 num 태그로 변환
        print(f'convert num : {text}') if self.verbose else None

        output_text = convert_text_final(text)  # 불필요한 문자 제거
        print(f'convert final : {output_text}') if self.verbose else None

        data = {}
        data.update(data_seq)
        data.update(data_dig)
        data.update(data_rep)
        data.update(data_ord)
        data.update(data_col)
        data.update(data_sha)
        data.update(data_nae)
        data.update(data_matched)
        data.update(data_unk)
        data.update(data_num)

        return output_text, data

    def tokenize_for_train(self, text):
        words = text.split()
        space_token = '_'

        output = []
        for word in words:
            result = re.findall(r'\d+', word)
            if len(result) == 0:
                for char in word:
                    output.append(char)
                output.append(space_token)
            else:
                output.append(word)

        if output[-1] == space_token:
            output = output[:-1]

        return ' '.join(output)

    def tokenize_for_train_v2(self, text):
        pos = self.komoran.pos(text)

        for key, tag in pos:
            if tag.startswith('NN'):
                text = text.replace(key, f' {key} ')
            elif tag.startswith('VV'):
                if len(key) > 1:
                    text = text.replace(key, f' {key} ')

        return ' '.join(text.split())


def convert_text_final(text):
    """
    불필요한 문자 제거
    :param text:
    :return:
    """
    output_text = text.replace('#', '')
    output_text = output_text.replace('.', '')
    output_text = output_text.replace(',', '')
    output_text = output_text.replace('?', '')
    output_text = ' '.join(output_text.split())

    return output_text
