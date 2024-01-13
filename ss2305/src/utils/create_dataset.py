import glob
import os
import sys
import re
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class DivideImage:
    def __init__(self, img_dir, excel_path):
        self.img_dir = img_dir
        self.excel_path = excel_path

    def extract_img(self):
        # Extract image files
        self.img_path_list = []
        for subdir, dirs, files in os.walk(self.img_dir):
            for file in files:
                if file.endswith('.png') and not file == '.DS_Store':
                    file_path = os.path.join(subdir, file)
                    self.img_path_list.append(file_path)
        return self.img_path_list

    # Get the Final_TIMI data corresponding to No from the xml file
    def extract_data_from_excel(self):
        # Load excel file
        df = pd.read_excel(self.excel_path)

        # key: 'No', value: 'Final_TIMI'
        data_dict = {}
        No_list = []

        # Iterate through the DataFrame
        for index, row in df.iterrows():
            # Check if 'No' is not Nan and if 'Final_TIMI' is present
            if pd.notna(row['No']) and pd.notna(row['Final_TIMI']):
                no = int(row['No'])
                data_dict[no] = row['Final_TIMI']
            if pd.notna(row['No']):
                No_list.append(row['No'])

        return data_dict

    # Extract 'No' from all images path
    def extract_no_from_path(self, img_path_list):
        # key : path, value : 'No'
        path_no_dic = {}
        for path in img_path_list:
            dir_name = path.split('/')[-2]
            no = re.findall(r'\d+', dir_name)
            path_no_dic[path] = int(no[0])

        return path_no_dic

    # Dividing image negative or positive
    def dived_image(self, data_dic, path_no_dic, save_dir):
        # Make directory to divide negative or positive if save_dir is not exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            if not os.path.exists(f'{save_dir}/negative'):
                os.makedirs(f'{save_dir}/negative')
            if not os.path.exists((f'{save_dir}/positive')):
                os.makedirs(f'{save_dir}/positive')

        # Saving path
        negative_path = os.path.join(save_dir, 'negative')
        positive_path = os.path.join(save_dir, 'positive')

        # Check if negative and positive are right
        check = {}

        for path, no in path_no_dic.items():

            move_dir_name = os.path.basename(os.path.dirname(path))  # ex) CHIBAMI_45_pre
            data_path = os.path.dirname(save_dir)
            non_defined_path = os.path.join(data_path, 'NonDefined')

            if not os.path.exists(non_defined_path):
                os.makedirs(non_defined_path)

            # Check if data_dic[no] is not exist
            if no not in data_dic:
                # Move the image to another file
                parent_dir_name = os.path.basename(self.img_dir)
                parent_dir_path = os.path.join(data_path, parent_dir_name)
                move_dir_path = os.path.join(parent_dir_path, move_dir_name)
                if not os.path.exists(os.path.join(non_defined_path, move_dir_name)):
                    shutil.move(move_dir_path, non_defined_path)
                else:
                    # Since the entire directory is moved, when the next 'path' is reached, pass through here
                    # print(f"'{path}' already exists.")
                    pass
                continue

            # Change file name
            num = self.extract_num(move_dir_name)
            new_file_name = self.rename(path, num)
            if data_dic[no] == 1 or data_dic[no] == 2:
                save_path = os.path.join(negative_path, new_file_name)
                shutil.copy2(path, save_path)
                check[no] = 'negative'
            elif data_dic[no] == 3:
                save_path = os.path.join(positive_path, new_file_name)
                shutil.copy2(path, save_path)
                check[no] = 'positive'
            else:
                print('Something went wrong')

        return check

    def extract_num(self, file_name):
        match = re.search(r'(\d+(\.\d+)?)', file_name)
        if match:
            return match.group(1)
        else:
            return None
    def rename(self, path, num):
        file_name = os.path.basename(path)
        not_extension = os.path.splitext(file_name)[0]
        extension = os.path.splitext(file_name)[1]
        new_file_name = f'{not_extension}_{num}{extension}'
        return new_file_name

    '''
    |- IVUS
    |- negative
        |- ex1.png
    |- positive
        |- ex2.png
    
    |- IVUS
        |- train
            |- negative
                |- ex1.png
            |- positive
                 |- ex2.png
         |- val
            |- negative
                |- ex1.png
            |- positive
                 |- ex2.png
         |- test
            |- negative
                |- ex1.png
            |- positive
                 |- ex2.png
    '''
    def split_dataset(self, root_dir, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
        if (train_ratio + val_ratio + test_ratio) != 1:
            print(f'train_ratio + val_ratio + test_ratio != 1')
            sys.exit(1)

        classes = ['negative', 'positive']

        for split in ['train', 'val', 'test']:
            for cls in classes:
                os.makedirs(os.path.join(root_dir, split, cls), exist_ok=True)

        # Function to split data and move files
        def split_and_move_files(class_dir, train_dir, val_dir, test_dir):
            files = os.listdir(class_dir)
            train_files, test_files = train_test_split(files, test_size=test_ratio)
            train_files, val_files = train_test_split(train_files, test_size=val_ratio/(train_ratio + val_ratio))

            # Helper function to move files
            def move_files(files, dest_dir):
                for file in files:
                    shutil.move(os.path.join(class_dir, file), os.path.join(dest_dir, file))

            # Move files to their respective directories
            move_files(train_files, train_dir)
            move_files(val_files, val_dir)
            move_files(test_files, test_dir)

        # Loop through each class directory and split files
        for cls in classes:
            class_dir = os.path.join(root_dir, 'IVUS', cls)
            train_dir = os.path.join(root_dir, 'train',  cls)
            val_dir = os.path.join(root_dir, 'val',  cls)
            test_dir = os.path.join(root_dir, 'test', cls)

            # Split and move the files
            split_and_move_files(class_dir, train_dir, val_dir, test_dir)

        print('Dataset split complete')

    # Confirm if the label is right
    # datadict : key : 'No', value : 'Final_TIMI'
    def confirm_label(self, negative_path, positive_path, data_dict):
        print('The following data is incorrectly labeled.')
        for path in [negative_path, positive_path]:
            for image_file in os.listdir(path):

                if image_file == '.DS_Store':
                    continue

                # Split '_'
                parts = image_file.split('_')
                # Extract number
                no = parts[-1].split('.')[0]
                no = int(no)

                # Check if the 'No' matches to 'Final_TIMI'
                # 'Final_TIMI' of files in negative is 1 or 2
                if path == negative_path:
                    if data_dict[no] == 3:
                        print(f'{image_file} is wrong.')

                # 'Final_TIMI' of files in negative is 3
                if path == positive_path:
                    if data_dict[no] != 3:
                        print(f'{image_file} is wrong.')

    def extract_case_directory(self):
        # Extract image files
        self.case_list = []
        for subdir, dirs, files in os.walk(self.img_dir):
            for dir in dirs:
                dir_path = os.path.join(subdir, dir)
                self.case_list.append(dir_path)
        return self.case_list

    def extract_no_from_dir_path(self, case_list):
        # key : path, value : 'No'
        path_no_dic = {}
        for path in case_list:
            dir_name = path.split('/')[-1]
            no = re.findall(r"CHIBAMI_(\d+)", dir_name)
            path_no_dic[path] = int(no[0])

        return path_no_dic

    def dived_case(self, data_dic, case_no_dic, save_dir):
        # Make directory to divide negative or positive if save_dir is not exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            if not os.path.exists(f'{save_dir}/negative'):
                os.makedirs(f'{save_dir}/negative')
            if not os.path.exists((f'{save_dir}/positive')):
                os.makedirs(f'{save_dir}/positive')

        # Saving path
        negative_path = os.path.join(save_dir, 'negative')
        positive_path = os.path.join(save_dir, 'positive')

        # Check if negative and positive are right
        check = {}

        for path, no in case_no_dic.items():

            move_dir_name = os.path.basename(os.path.dirname(path))  # ex) anonymized_list
            data_path = os.path.dirname(save_dir)
            non_defined_path = os.path.join(data_path, 'NonDefined')
            directory_name = path.split('/')[-1]

            if not os.path.exists(non_defined_path):
                os.makedirs(non_defined_path)

            # Check if data_dic[no] is not exist
            if no not in data_dic:
                # Move the case directory
                if not os.path.exists(os.path.join(non_defined_path, directory_name)):
                    shutil.move(path, non_defined_path)
                continue

            # Change file name
            image_files = glob.glob(os.path.join(path, "*.png"))
            for image_file in image_files:
                filename = image_file.split('/')[-1]
                num = self.extract_num(directory_name)
                new_file_name = self.rename_image(filename, num)
                os.rename(image_file, os.path.join(path, new_file_name))

            if data_dic[no] == 1 or data_dic[no] == 2:
                save_path = os.path.join(positive_path, directory_name)
                shutil.copytree(path, save_path)
                check[no] = 'positive'
            elif data_dic[no] == 3:
                save_path = os.path.join(negative_path, directory_name)
                shutil.copytree(path, save_path)
                check[no] = 'negative'
            else:
                print('Something went wrong')

        return check

    def rename_image(self, image, num):
        """
        original file name : frame_4020.png
        changed file name : frame_{case number}_4020.png
        """
        name, ext = image.split('.')
        parts = name.split('_')

        new_name = f"{parts[0]}_{num}_{parts[1]}.{ext}"

        return new_name

def extract_case_number(path):
    """ ファイルパスから症例番号を抽出する。小数点がある場合はそのまま、ない場合は整数型にする """
    match = re.search(r'CHIBAMI_(\d+(\.\d+)?)_pre', path)
    if match:
        case_num_str = match.group(1)
        # if "." in case_num_str:
        #     print(path)

        return (case_num_str)
    else:
        return None


if __name__ == '__main__':
    img_dir_path = '../../data/anonymized_list'
    # img_dir_path = '../../data/input_test'
    xml_path = '../../data/CHIBAMI_case_list.xlsx'

    cd = DivideImage(img_dir_path, xml_path)

    case_path_list = cd.extract_case_directory()

    # data_dict key : 'No', value : 'Final_TIMI'
    data_dic = cd.extract_data_from_excel()

    # Extract 'No' from case path
    case_No_dic = cd.extract_no_from_dir_path(case_path_list)

    # Diving image depend on 'Final_TIMI'
    save_path = '../../data/IVUS_all'
    # save_path = '../../data/output_test'
    check = cd.dived_case(data_dic, case_No_dic, save_path)
    print(check)
    print()

    # # Confirm if label is right
    # negative_path = '../../data/IVUS/negative'
    # positive_path = '../../data/IVUS/positive'
    # # negative_path = '../../data/output_test/negative'
    # # positive_path = '../../data/output_test/positive'
    # cd.confirm_label(negative_path, positive_path, data_dic)

    # Split dataset
    # cd.split_dataset('../../data')
