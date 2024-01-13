import numpy as np

from moduls.CT_image_processing import CT3DImageProcesser
from moduls.MetaImageWriter import MetaImageWriter
from moduls.MetaImageRead import MetaDataRead


if __name__ == '__main__':
    CT_data_dir_path = '../data/ChestCT/ChestCT.mhd'
    ct_3d = CT3DImageProcesser(CT_data_dir_path)
    thresholded_3D_image = ct_3d.thresholding_3D_image(ct_3d.image_array, -100, 200)
    morphology_3D_image = ct_3d.morphological_3D_operations(thresholded_3D_image)
    labeled_3D_img = ct_3d.larger_connected_component3D(morphology_3D_image)

    # Read mhd file
    data_read = MetaDataRead('../data/ChestCT/ChestCT.mhd')
    mhd_dict = data_read.read_as_dict()
    # Changing the output file name and element type
    mhd_dict['ElementDataFile'] = 'ss2304-02.raw'

    # Write raw file
    writer = MetaImageWriter('../data/ChestCT')
    # Change 3D_img to property data type
    data_type = writer.get_data_type(mhd_dict['ElementType'])
    writer.save_as_metaimage(image_data=labeled_3D_img.astype(data_type), header_info_dic=mhd_dict, save_file_name='ss2304-02')