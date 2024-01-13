from moduls.DicomHandler import DICOMHandler, get_from_commandline
from moduls.MetaImageWriter import MetaImageWriter


if __name__ == '__main__':
    args = get_from_commandline()
    dicom_dir_path, output_filename = args.dicom_dir_path, args.output_filename
    dcm = DICOMHandler(dicom_dir_path, output_filename)
    sorted_list_dicom_files_path = dcm.get_sorted_list_dicom_files_path()
    header_dicom_dict = dcm.get_info_from_dicom_file(sorted_list_dicom_files_path)
    volume = dcm.stack_dicom_files()

    # Argument is directory name
    writer = MetaImageWriter('../data/HeadCtSample_2022')
    writer.save_as_metaimage(image_data=volume, header_info_dic=header_dicom_dict, save_file_name=output_filename)

