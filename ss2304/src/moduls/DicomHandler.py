import argparse
import numpy as np
import os
import pydicom


class DICOMHandler:
    def __init__(self, dir_name_path: str, output_filename):
        try:
            if not os.path.exists(dir_name_path):
                raise ValueError(f'{dir_name_path} is not found')
            self.dir_name_path: str = dir_name_path
        except ValueError as e:
            print(f'Error: {e}')

        self.output_filename = output_filename
        self.dicom_files_path = []  # Absolute path
        self.sorted_list_dicom_files_path = None
        self.type_of_inspection = None  # CT, MR, PT ect

        # Get DICOM files path
        self.list_dicom_files(self.dir_name_path)

        # Sort Instance Number in ascending order
        self.sorted_list_dicom_files_path = self.sort_dicom_files_by_instance_number(self.dicom_files_path)

        # Information to write mhd file
        self.dicom_info_dict = {}

        # 3D volume
        self.volume = None

    # Check whether the input file is DICOM file
    def is_dicom_file(self, filename):
        try:
            # Load DICOM file
            dcm = pydicom.dcmread(filename)

            # DICOM files usually include "PatientName" and other specific tags, so
            # By check this, checking whether the file is DICOM
            return "PatientName" in dcm
        except:
            return False

    # Add DICOM files to list
    def list_dicom_files(self, dir_path):
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                if self.is_dicom_file(file_path):
                    self.dicom_files_path.append(file_path)

    # Sort Instance Number in ascending order
    def sort_dicom_files_by_instance_number(self, list_dicom_file_path):
        # Sort DICOM files
        sorted_dicom_files_path = sorted(self.dicom_files_path, key=lambda x: pydicom.dcmread(x).InstanceNumber)
        return sorted_dicom_files_path

    def get_sorted_list_dicom_files_path(self):
        return self.sorted_list_dicom_files_path

    # 8-bit unsigned integer: MET_UCHAR
    # 16-bit unsigned integer: MET_USHORT
    # 16-bit signed integer: MET_SHORT
    def dicom_to_metimage_datatype(self, dicom):
        # Get Bits Allocated and PixelRepresentation tags
        bits_allocated = dicom.BitsAllocated
        pixel_representation = dicom.PixelRepresentation

        if bits_allocated == 8 and pixel_representation == 0:
            return 'MET_UCHAR'
        elif bits_allocated == 16 and pixel_representation == 0:
            return 'MET_USHORT'
        elif bits_allocated == 16 and pixel_representation == 1:
            return 'MET_SHORT'
        else:
            return 'Unknown'

    # Using Euclidean distance
    def get_resolution(self, first_dcm, second_dcm):
        x_res, y_res = first_dcm.PixelSpacing   # resolution[mm/pixel]

        position1 = np.array(first_dcm.ImagePositionPatient)
        position2 = np.array(second_dcm.ImagePositionPatient)
        z_res = np.linalg.norm(position1 - position2)

        # Round to 2 decimal places and convert to float
        x_res = float(round(x_res, 5))
        y_res = float(round(y_res, 5))
        z_res = float(round(z_res, 5))

        return x_res, y_res, z_res

    # Get information from DICOM files
    def get_info_from_dicom_file(self, sorted_list_dicom_files_path):
        # Get information to write mhd file
        ObjectType = 'Image'
        NDims = 3 if len(self.sorted_list_dicom_files_path) > 1 else 2

        first_dicom = pydicom.dcmread(self.sorted_list_dicom_files_path[0])
        second_dicom = pydicom.dcmread(self.sorted_list_dicom_files_path[1])
        xy_dims = first_dicom.pixel_array.shape
        z_dim = len(self.sorted_list_dicom_files_path)
        DimSize = xy_dims + (z_dim,)
        ElementType = self.dicom_to_metimage_datatype(first_dicom)  # Data type
        ElementSpacing = self.get_resolution(first_dicom, second_dicom)  # Space-separated resolution
        ElementByteOrderMSB = False
        ElementDataFile = f'{self.output_filename}'

        self.dicom_info_dict = {
            'ObjectType': ObjectType,
            'NDims': NDims,
            "DimSize": f'{DimSize[0]} {DimSize[1]} {DimSize[2]}',
            'ElementType': ElementType,
            'ElementSpacing': f'{ElementSpacing[0]} {ElementSpacing[1]} {ElementSpacing[2]}',
            'ElementByteOrderMSB': ElementByteOrderMSB,
            'ElementDataFile': ElementDataFile,
            # Add the following to the mhd file
            # To prevent vertically flipped display in the Z direction
            # 'TransformMatrix': '10001000-1',
            # 'Offset': '000',
            # 'CenterOfRotation': '000',
            # 'AnatomicalOrientation': 'RAS',
        }

        return self.dicom_info_dict

    def stack_dicom_files(self):
        # Create an empty list to store 2D numpy arrays
        slices = []

        # Check data type
        first_dicom = self.sorted_list_dicom_files_path[0]
        first_dicom_data = pydicom.dcmread(first_dicom)
        data_type = self.dicom_to_metimage_datatype(first_dicom_data)
        np_type = self.get_data_type(data_type)
        print(np_type)

        # Loop through each sorted DICOM file path
        for dicom_file_path in self.sorted_list_dicom_files_path:
            dicom_data = pydicom.dcmread(dicom_file_path)

            # If the modality is CT, calculate the CT value using the following formula
            if dicom_data.Modality == 'CT':
                dicom_pixel_array_temp = dicom_data.pixel_array
                dicom_pixel_array_temp = dicom_pixel_array_temp * dicom_data.RescaleSlope + dicom_data.RescaleIntercept
                dicom_pixel_array_temp = dicom_pixel_array_temp.astype(np_type)
                slices.append(dicom_pixel_array_temp)
            else:
                slices.append(dicom_data.pixel_array)

        # Stack 2D slices to create a 3D volume
        self.volume = np.stack(slices)
        # print(self.volume)
        return self.volume

        # Get data type from mhd file

    def get_data_type(self, type):
        if type == 'MET_UCHAR':
            return np.unit8
        elif type == 'MET_USHORT':
            return np.uint16
        elif type == 'MET_SHORT':
            return np.int16

def get_from_commandline():
    parser = argparse.ArgumentParser(description='Enter input and output file name of DICOM file.')

    # The following arguments have the format (variable name, type, help string)
    parser.add_argument('dicom_dir_path', type=str, help='Input file name of DICOM')
    parser.add_argument('output_filename', type=str, help='Output file name of DICOM')

    # check if there are enough arguments from the command line
    try:
        args = parser.parse_args()
    except argparse.ArgumentError as e:
        print(f'Command line argument error: {e}')
        return

    return args


if __name__ == '__main__':
    # args = get_from_commandline()
    dcm = DICOMHandler('../../data', 'output')
    sorted_list_dicom_files_path = dcm.get_sorted_list_dicom_files_path()
    dicom_info_dict = dcm.get_info_from_dicom_file(sorted_list_dicom_files_path)
    volume = dcm.stack_dicom_files()
