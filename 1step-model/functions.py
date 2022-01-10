def func_final(input_value):
    output = None
    if type(input_value) == int or type(input_value) == float:
        input_int = int(input_value)

        if input_int != input_value:
            output = round(input_value, 2)
        else:
            output = input_int
    else:
        output = input_value
    result = output
    return result


def func_add(arg0, arg1):
    # Function : add
    result = arg0 + arg1
    return result


def func_minus(arg0, arg1):
    # Function : minus
    result = arg0 - arg1
    return result


def func_multiply(arg0, arg1):
    # Function : multiply
    result = arg0 * arg1
    return result


def func_divide(arg0, arg1):
    # Function : divide
    result = arg0 / arg1
    return result


def func_findbasicseq(rep_seq_list, rep_num_list):
    seq_list = list()
    if rep_num_list is None:
        cut_num = len(rep_seq_list) // 2
        for i in range(2, cut_num + 1):
            seq_num = i
            seq_a = rep_seq_list[:seq_num]
            seq_b = rep_seq_list[seq_num:seq_num + seq_num]
            if seq_a == seq_b:
                seq_list = seq_a
                break
            seq_list = rep_seq_list
    else:
        for i in range(len(rep_num_list)):
            dis_rep_seq = rep_seq_list[i]
            dis_rep_num = rep_num_list[i]
            for j in range(dis_rep_num):
                seq_list.append(rep_seq_list[i])

    result = seq_list
    return result


def func_findord(basic_seq_list, order):
    result = basic_seq_list[order % len(basic_seq_list) - 1]
    return result


def func_combinations(elems, num):
    from itertools import combinations
    result = len(list(combinations(elems, num)))
    return result


#
# 6
#
def func_transform_list(number0, number1):
    result = [number0, number1]
    return result


def func_inverseoperator(number_list, operator0):
    opers = {"+": (lambda x, y: x + y), "-": (lambda x, y: x - y), "*": (lambda x, y: x * y), "/": (lambda x, y: x / y)}
    fault = {'+': '-', '-': '+', '*': '/', '/': '*'}
    result = opers[fault[operator0]](number_list[1], number_list[0])
    return result


def func_rightoperator(number_list, operator0):
    opers = {"+": (lambda x,y: x+y), "-": (lambda x,y: x-y), "*": (lambda x,y: x*y),"/": (lambda x,y: x/y),"%": (lambda x,y: x%y)}
    answer = opers[operator0](number_list[0], number_list[1])
    result = answer
    return result


#
# 1
#
def func_sum(arg0, _):
    # arg0 (datatype: list()): list of numbers to be summed
    # return: sum of given numbers
    result = sum(arg0)
    return result


def func_odd(arg0, _):
    # arg0 (datatype: list()): list of numbers to find odd numbers
    # return: list of odd numbers from given list
    result = [i for i in arg0 if i % 2 == 1]
    return result


def func_range(arg0, arg1):
    # arg0 (datatype: int): start of list
    # arg1 (datatype: int): end of list (to be included)
    # return: list of numbers from start to end
    result = range(arg0, arg1 + 1)
    return result


def func_circlearea(arg0, arg1):
    # arg0 (datatype: int/float): radius of circle
    # arg1 (datatype: float): pi
    # return: area of circle
    from math import pi
    if arg1 == None:
        output = pi * arg0 * arg0
    else:
        output = arg1 * arg0 * arg0
    result = round(output, 2)
    return result


def func_radius(arg0, _):
    # arg0: diameter of circle
    # return: radius of circle, given diameter
    result = arg0 / 2
    return result


def func_m2cm(arg0, arg1):
    # arg0: m
    # arg1: cm (if given, else None)
    # return: _m _cm converted into _cm
    if arg1 == None:
        result = arg0 * 100
    else:
        result = arg0 * 100 + arg1
    return result


def func_diameter(arg0, _):
    # arg0: radius
    # return: diameter
    result = arg0 * 2
    return result


def func_cm2m(arg0, _):
    # arg0: cm
    # return: _m
    result = arg0 / 100
    return result


def func_perobj(arg0, arg1):
    # 사과 5개가 2250원
    # arg0: whole value (2250)
    # arg1: object count (5)
    # return value per count (2250/5)
    result = arg0 / arg1
    return result


def func_das2jaru(arg0, _):
    # arg0: n다스
    # return: 12n자루
    result = arg0 * 12
    return result


def func_kg2g(arg0, arg1):
    # arg0: kg
    # arg1: g (if given)
    # return: _kg _g to _g
    if arg1 == None:
        result = arg0 * 1000
    else:
        result = arg0 * 1000 + arg1
    return result


def func_even(arg0, _):
    # arg0: list of numbers
    # return: list of odd numbers from given list
    result = [i for i in arg0 if i % 2 == 0]
    return result


def func_multiple(arg0, arg1):
    # arg0: list of numbers
    # arg1: we need to find multiple of arg1
    # return: list of numbers that are multiple of arg1
    output = []
    for i in arg0:
        if i % arg1 == 0:
            output.append(i)
    # result = [i for i in arg0 if i % arg1 == 0]
    result = output
    return result


def func_between(arg0, arg1):
    # arg0: left num
    # arg1: right num
    # reutrn: center num
    result = range(arg0, arg1)[1]
    return result


def func_one(arg0, arg1):
    result = 1
    return result


def func_listindex(arg0,arg1):
    # arg0: list of numbers
    # arg1: n 번째 숫자
    result = arg0[arg1-1]
    return result


def func_listindexright(arg0,arg1):
    # arg0: list of numbers
    # arg1: 오른쪽에서 n번째 숫자
    result = arg0[-1*arg1]
    return result

def func_same(arg0,_):
    # returns the exact same number
    result = arg0
    return result


#
# 3
#
def func_product(elems, number):
    import itertools
    result = len(list(itertools.product(elems, repeat=number)))
    return result


def func_permutation(elems, number):
    import itertools
    result = len(list(itertools.permutations(elems, number)))
    return result


def func_combination(elems, number):
    import itertools
    result = len(list(itertools.combinations(elems, number)))
    return result


def func_combinations_with_replacement(elems, number):
    import itertools
    result = len(list(itertools.combinations_with_replacement(elems, number)))
    return result


def func_product_list(elems, number):
    import itertools
    output = []
    for a in itertools.product(elems, repeat=number):
        if a[0] != 0:
            output.append(int(''.join(map(str, a))))

    result = output
    # result = [int(''.join(map(str, a))) for a in itertools.product(elems, repeat=number) if a[0] != 0]
    return result


def func_permutation_list(elems, number):
    import itertools
    result = [int(''.join(map(str, a))) for a in itertools.permutations(elems, number) if a[0] != 0]
    return result


def func_len(elems, none):
    result = len(elems)
    return result


def func_find_order(elems, number):
    if len(number[1]) > 1:  # num[1]: 큰 or 작은
        flag = False
    else:
        flag = True
    result = sorted(elems, reverse=flag)[number[0] - 1]
    return result


def func_find_odd(elems, none):
    result = len([x for x in elems if x % 2 == 1])
    return result


def func_find_even(elems, none):
    result = len([x for x in elems if x % 2 == 0])
    return result


def func_find_range(elems, number):
    output = []
    for x in elems:
        if x <= number[1] and x >= number[0]:
            output.append(x)

    result = len(output)
    # result = len([x for x in elems if x <= number[1] and x >= number[0]])
    return result


def func_find_mul(elems, number):
    output = []
    for x in elems:
        if x % number == 0:
            output.append(x)

    result = len(output)
    # result = len([x for x in elems if x % number == 0])
    return result


def func_range_list(number, none):
    result = [i for i in range(number)]
    return result


def func_makelist(arg0, arg1):
    result = [arg0, arg1]
    return result

#
# 4
#

def func_OrderedList(num, dig):
    if dig == 'desc':
        sort_seq = sorted(num,reverse=True)
    elif dig == 'asc':
        sort_seq = sorted(num)
    result = sort_seq
    return result


def func_Find6Numbers(type_list, equ_list):
    result='error'
    equ_text = ''.join(equ_list) #리스트를 텍스트로 바꾼다
    count=0
    for i in range(type_list[0], type_list[1]+1):
        answer = eval(equ_text, {'A': i}) #텍스트 안에 변수를 하나씩 넣어서 계산한다
        if answer == True: # 계산결과 로직이 정상이면 카운트 업
            count+=1
    result = count
    return result


def func_FindNumber(num, dir0):
    if dir0 == 'first':
        answer = num[0]
    elif dir0 == 'last':
        answer = num[-1]
    elif dir0 == 'middle':
        mid_number = len(num)
        if mid_number%2 == 0 :
            mid_index = int(mid_number/2)
            answer = num[mid_index-1]
        else :
            mid_index = int((mid_number)/2)+1
            answer = num[mid_index-1]
    else:
        idx = dir0 - 1
        answer = num[idx]
    result = answer
    return result


def func_MakeSeq(dig0, none):
    seq_1 = range(10**dig0)
    Answer = list()
    for i in range(len(seq_1)):
        if seq_1[i] >= 10**(dig0-1):
            Answer.append(seq_1[i])
    result = Answer
    return result


def func_FindDivisor(seq_list, num_list):
    Answer = list()
    epoch_num = len(num_list)
    for i in range(epoch_num):
        mid_Answer = list()
        for j in range(len(seq_list)):
            if seq_list[j]%num_list[i] == 0:
                mid_Answer.append(1)
            else:
                mid_Answer.append(0)
        Answer.append(mid_Answer)
    result = Answer
    return result


def func_MergeList(seq_list_1, seq_list_2):
    answer = list()
    answer.append(seq_list_1)
    answer.append(seq_list_2)

    result = answer
    return result


def func_Cal_CommonDivisor_1(Mer_seq_list, dir0):
    Answer_seq = list()
    seq_list_num = len(Mer_seq_list)

    for i in range(seq_list_num - 1):
        # mid_Answer_seq = list()
        if i == 0:
            seq_list_1 = Mer_seq_list[0]
            seq_list_2 = Mer_seq_list[1]
            if dir0 == '또는':
                for i in range(len(seq_list_1)):
                    Answer_seq.append(seq_list_1[i] | seq_list_2[i])
            else:
                for i in range(len(seq_list_1)):
                    Answer_seq.append(seq_list_1[i] & seq_list_2[i])
        else:
            seq_list_1 = Answer_seq
            seq_list_2 = Mer_seq_list[i + 1]
            if dir0 == '또는':
                for i in range(len(seq_list_1)):
                    Answer_seq[i] = seq_list_1[i] | seq_list_2[i]
            else:
                for i in range(len(seq_list_1)):
                    Answer_seq[i] = seq_list_1[i] & seq_list_2[i]
    result = Answer_seq
    return result


def func_Count(seq_list,none):
    Answer = seq_list.count(1)
    result = Answer
    return result


def func_movepoint(dir0, dig0):
    answer=list()
    if dir0 == 'right':
        answer.append(dir0)
        answer.append(10**dig0)
    elif dir0 == 'left':
        answer.append(dir0)
        answer.append(1/(10**dig0))
    else:
        answer.append(dir0)
        answer.append(0)
    result = answer
    return result


def func_OriginNumber(dir_list, num0):
    answer = 0
    if dir_list[0] == 'right':
        answer = round(num0 / (dir_list[1] - 1), 2)
    elif dir_list[0] == 'left':
        answer = round(num0 / (1 - dir_list[1]), 2)
    else:
        answer = 0

    result = answer
    return result


def func_FindFirstnumber(num, dir0):
    if dir0 == 'largest':
        sort_seq = sorted(num, reverse =True)
        answer = sort_seq[0]
    elif dir0 == 'smallest':
        sort_seq = sorted(num)
        answer = sort_seq[0]
    else:
        sort_seq = sorted(num, reverse =True)
        answer = sort_seq[int(dir0) - 1]
    result = answer
    return result


def func_removeelement(num, dig):
    if dig == 'largest':
        sort_seq = sorted(num,reverse=True)
    else:
        sort_seq = sorted(num)
    rev_num = sort_seq[0]
    answer = sort_seq.remove(rev_num)
    answer = sort_seq
    result = answer
    return result


def func_SplitNumber(number, none):
    answer = []
    while(number!=0):
        answer.append(number%10)
        number = number//10
    answer.reverse()
    result = answer
    return result


def func_FinddigitNumber(number_0, number_1):
    digit_index = len(number_1)
    answer = number_0[-1*digit_index]
    result = answer
    return result


def func_diff_Numbers(dir_list, num0):
    answer= 0
    if dir_list[0] == 'right':
        answer = round((num0*dir_list[1])-num0,2)
    elif dir_list[0] == 'left':
        answer = round(num0-(num0*dir_list[1]),2)
    else:
        answer = 0
    result = answer
    return result

#
# 7
#

def func_difference(arg0,arg1):
    if arg0 >= arg1:
      result = arg0 - arg1
    else:
      result = arg1 - arg0
    return result

def func_findindex(rep_seq_list, text):
    if text==None:
        answer = len(rep_seq_list) + 1
    else:
        answer = int(rep_seq_list.index(text))
    result = answer
    return result

"""def func_findindex(rep_seq_list, text):
    if(text == 'A'):
        answer = int(rep_seq_list.index('A'))
    elif(text == 'B'):
        answer = int(rep_seq_list.index('B'))
    result = answer
    return result"""


def func_operator(opr, number):
    if (opr == '-'):
        if (number[0] > number[1]):
            answer = number[0] - number[1]
        else:
            answer = number[1] - number[0]
    elif (opr == '+'):
        answer = number[0] + number[1]
    elif (opr == '*'):
        answer = number[0] * number[1]
    elif (opr == '/'):
        if (number[0] > number[1]):
            answer = number[0] / number[1]
        else:
            answer = number[1] / number[0]

    result = answer
    return result


def func_findrule(rep_seq_list, rep_num_list):
    seq_z = []
    end = 0
    for_end = 0
    answer = []

    if (end == 0):
        for_end = 0
        for i in range(len(rep_seq_list) - 1):
            if (type(rep_seq_list[i]) == int and type(rep_seq_list[i + 1]) == int):
                seq_z.append(int(rep_seq_list[i + 1]) - int(rep_seq_list[i]))

        for i in range(len(seq_z) - 1):
            if (for_end == 0):
                if (seq_z[i + 1] == seq_z[i]):
                    abc = True
                    end = 1
                else:
                    abc = False
                    for_end = 1
                    end = 0

    if (end == 0):
        if rep_num_list is None:
            seq_z = []
            seq_a = []
            seq_b = []
            for_end = 0
            if (type(rep_seq_list[i]) == int and type(rep_seq_list[i + 1]) == int):
                cut_num = len(rep_seq_list) // 2
                for i in range(2, cut_num + 1):
                    seq_num = i
                    seq_a = rep_seq_list[:seq_num]
                    seq_b = rep_seq_list[seq_num:seq_num + seq_num]
                    if seq_a == seq_b:
                        seq_z = seq_a
                        end = 2
                        abc = True
                        break
                    seq_z = rep_seq_list

    if (end == 0):
        for_end = 0
        seq_z = []
        answer = []
        for i in range(len(rep_seq_list) - 1):
            if (type(rep_seq_list[i]) == int and type(rep_seq_list[i + 1]) == int):
                if (int(rep_seq_list[i]) > int(rep_seq_list[i + 1])):
                    seq_z.append(int(rep_seq_list[i]) / int(rep_seq_list[i + 1]))

        for i in range(len(seq_z) - 1):
            if (for_end == 0):
                if (seq_z[i + 1] == seq_z[i]):
                    abc = True
                    end = 3
                else:
                    abc = False
                    for_end = 1
                    end = 0

    if (end == 0):
        for_end = 0
        seq_z = []
        answer = []
        for i in range(len(rep_seq_list) - 1):
            if (type(rep_seq_list[i]) == int and type(rep_seq_list[i + 1]) == int):
                if (int(rep_seq_list[i]) < int(rep_seq_list[i + 1])):
                    seq_z.append(int(rep_seq_list[i + 1]) / int(rep_seq_list[i]))

        for i in range(len(seq_z) - 1):
            if (for_end == 0):
                if (seq_z[i + 1] == seq_z[i]):
                    abc = True
                    end = 4
                else:
                    abc = False
                    for_end = 1
                    end = 0
    if (end == 0):
        for_end = 0
        seq_a = []
        seq_z = []
        answer = []
        for i in range(len(rep_seq_list) - 1):
            if (type(rep_seq_list[i]) == int and type(rep_seq_list[i + 1]) == int):
                seq_a.append(int(rep_seq_list[i + 1]) - int(rep_seq_list[i]))  # 4 9 ...

        for i in range(len(seq_a) - 1):
            seq_z.append(int(seq_a[i + 1]) - int(seq_a[i]))  # 5 9 11
        for i in range(len(seq_z) - 1):
            if (for_end == 0):
                if (seq_z[i + 1] == seq_z[i]):
                    abc = True
                    end = 5
                else:
                    abc = False
                    for_end = 1
                    end = 0
    if (end == 0):
        for_end = 0
        seq_a = []
        seq_b = []
        seq_z = []
        answer = []
        for i in range(len(rep_seq_list) - 1):
            if (type(rep_seq_list[i]) == int and type(rep_seq_list[i + 1]) == int):
                seq_a.append(int(rep_seq_list[i + 1]) - int(rep_seq_list[i]))  # 4 9 ...

        for i in range(len(seq_a) - 1):
            seq_b.append(int(seq_a[i + 1]) - int(seq_a[i]))  # 5 9 11
        for i in range(len(seq_b) - 1):
            seq_z.append(int(seq_b[i + 1]) - int(seq_b[i]))
        for i in range(len(seq_z) - 1):
            if (for_end == 0):
                if (seq_z[i + 1] == seq_z[i]):
                    abc = True
                    end = 6
                else:
                    abc = False
                    for_end = 1
                    end = 0
    if (abc == True):
        if (end == 2):
            answer.append(end)
            answer.append(seq_z)
        else:
            answer.append(end)
            answer.append(seq_z[0])
    result = answer
    return result


def func_seqval(rep_seq_list, basic_seq_list):
    not_a = 0
    for i in range(len(rep_seq_list) - 1):
        if (type(rep_seq_list[i]) == int and type(rep_seq_list[i + 1]) == int):
            not_a = 1
        else:
            not_a = 0
            break
    if (not_a == 0):
        if (basic_seq_list[1] == 0):
            if (basic_seq_list[0][0] == 1):
                output = rep_seq_list[basic_seq_list[1] + 1] - basic_seq_list[0][1]
                result = output

            elif (basic_seq_list[0][0] == 4):
                output = rep_seq_list[basic_seq_list[1] + 1] / (basic_seq_list[0][1])
                result = output

            elif (basic_seq_list[0][0] == 3):
                output = rep_seq_list[basic_seq_list[1] + 1] * (basic_seq_list[0][1])
                result = output

        else:
            if (basic_seq_list[0][0] == 1):
                output = rep_seq_list[basic_seq_list[1] - 1] + basic_seq_list[0][1]
                result = output

            elif (basic_seq_list[0][0] == 3):
                output = rep_seq_list[basic_seq_list[1] - 1] / basic_seq_list[0][1]
                result = output

            elif (basic_seq_list[0][0] == 4):
                output = rep_seq_list[basic_seq_list[1] - 1] * (basic_seq_list[0][1])
                result = output

            elif (basic_seq_list[0][0] == 5):
                two = rep_seq_list[1] - rep_seq_list[0]
                output = rep_seq_list[0]
                for i in range(basic_seq_list[1]):
                    if (i == 0):
                        output += two
                    else:
                        two += basic_seq_list[0][1]
                        output += two
                result = output
            elif (basic_seq_list[0][0] == 6):
                for i in range(1, basic_seq_list[1] + 2):
                    output = 0
                    for j in range(1, i + 1):
                        output = output + j * j
                result = output

            elif (basic_seq_list[0][0] == 2):
                output = basic_seq_list[0][1][basic_seq_list[1] % len(basic_seq_list[0][1])]
                result = output


    elif (not_a == 1):
        if (basic_seq_list[0][0] == 1):
            output = rep_seq_list[0] + basic_seq_list[0][1] * (basic_seq_list[1] - 1)
            result = output
        elif (basic_seq_list[0][0] == 3):
            output = rep_seq_list[0] / (basic_seq_list[0][1] ** (basic_seq_list[1] - 1))
            result = output
        elif (basic_seq_list[0][0] == 4):
            output = rep_seq_list[0] * (basic_seq_list[0][1] ** (basic_seq_list[1] - 1))
            result = output
        elif (basic_seq_list[0][0] == 5):
            two = rep_seq_list[1] - rep_seq_list[0]
            output = rep_seq_list[0]
            for z in range(basic_seq_list[1] - 1):
                if (z == 0):
                    output += two
                else:
                    two += basic_seq_list[0][1]
                    output += two
            result = output
        elif (basic_seq_list[0][0] == 6):
            for i in range(1, basic_seq_list[1] + 1):
                output = 0
                for j in range(1, i + 1):
                    output = output + j * j
            result = output
        elif (basic_seq_list[0][0] == 2):
            output = basic_seq_list[0][1][basic_seq_list[1] % len(basic_seq_list[0][1]) - 1]
            result = output
    return result

