import os
import numpy as np


# Class that writes to file
class MetaImageWriter:
    def __init__(self, output_directory_path):
        self.output_directory_path = output_directory_path
        self.output_filename = None

        # Check if the directory exists
        try:
            if not os.path.exists(output_directory_path):
                raise ValueError(f'{output_directory_path} is not found')
        except ValueError as e:
            print(f'Error: {e}')

    # Receive it in dictionary format and write a raw file
    def write_dict_to_mhd(self, header_info_dic, save_file_name):
        # 'ElementDataFile' is added extension if it does not have it.
        if '.' not in header_info_dic['ElementDataFile']:
            header_info_dic['ElementDataFile'] += '.raw'

        # Add extension if 'save_file_name' do not have it
        if '.' in save_file_name:
            output_filename = '.'.join(save_file_name.split('.')[:-1]) + '.mhd'
        else:
            output_filename = save_file_name + '.mhd'

        output_file_path = os.path.join(self.output_directory_path, output_filename)
        with open(output_file_path, 'w') as mhd_file:
            for key, value in header_info_dic.items():
                mhd_file.write(f"{key} = {value}\n")

    # argument 'image_data' is a ndarray(z, y, x)
    def write_raw_file(self, image_data, save_file_name):
        # Add extension if 'save_file_name' do not have it
        if '.' in save_file_name:
            output_filename = '.'.join(save_file_name.split('.')[:-1]) + '.raw'
        else:
            output_filename = save_file_name + '.raw'
        output_file_path = os.path.join(self.output_directory_path, output_filename)

        image_data.tofile(output_file_path)
        # with open(output_file_path, 'wb') as raw_file:
        #     raw_file.write(image_data.tobytes())

    def save_as_metaimage(self, image_data, header_info_dic, save_file_name=None):
        # Get filename
        self.output_filename = header_info_dic['ElementDataFile']

        # First, save the raw file
        self.write_raw_file(image_data, save_file_name)

        # Next, save the mhd file
        self.write_dict_to_mhd(header_info_dic, save_file_name)

     # Get data type from mhd file
    def get_data_type(self, type):
        if type == 'MET_UCHAR':
            return np.unit8
        elif type == 'MET_USHORT':
            return np.uint16
        elif type == 'MET_SHORT':
            return np.int16

