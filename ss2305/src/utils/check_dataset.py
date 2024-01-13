# Path: ../data/IVUS/negative/CHIBAMI_483_pre/frame_483_840.png, Label: 0
import re

def count_num(text_path):
    case_count = {}
    with open(text_path, 'r') as file:
        for line in file:
            left = line.split(',')[0]
            dir = left.split('/')[-2]
            match = re.search(r'CHIBAMI_(\d+(\.\d+)?)_pre', dir)
            if match:
                case_num_str = match.group(1)
                # ケース番号が辞書に存在するか確認し、存在しない場合は新たにキーを作成
                if case_num_str not in case_count:
                    case_count[case_num_str] = 1
                else:
                    # 既に存在する場合はカウントを増やす
                    case_count[case_num_str] += 1

        for k, v in case_count.items():
            print(f'{k} : {v}')
        print(len(case_count))

def count_path(path_list):
    case_count = {}
    for line in path_list:
        left = line.split(',')[0]
        dir = left.split('/')[-2]
        match = re.search(r'CHIBAMI_(\d+(\.\d+)?)_pre', dir)
        if match:
            case_num_str = match.group(1)
            # ケース番号が辞書に存在するか確認し、存在しない場合は新たにキーを作成
            if case_num_str not in case_count:
                case_count[case_num_str] = 1
            else:
                # 既に存在する場合はカウントを増やす
                case_count[case_num_str] += 1

    for k, v in case_count.items():
        if v < 18:
            print(f'case number of {k} is {v}. ')
        elif v < 30 and v > 23:
            print(f'case number of {k} is {v}. ')
        elif v < 146 and v > 143:
            print(f'case number of {k} is {v}. ')
        elif v < 137 and v > 139:
            print(f'case number of {k} is {v}. ')

    print(len(case_count))



if __name__ == '__main__':
    count_num('./data.txt')


