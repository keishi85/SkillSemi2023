import itk
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import filters


class CTImageProcesser:
    def __init__(self, mhd_file_path):
        # Read mhd and raw file
        self.image = itk.imread(mhd_file_path)

        # Get numpy array
        self.image_array = itk.array_from_image(self.image)

        self.ex_img = self.image_array[34]

    # Threshold processing
    def thresholding(self, img, lower_thresh, upper_thresh):
        return np.where((img > lower_thresh) & (img < upper_thresh), 1, 0)

    # Otsu threshold process
    def otsu_thresholding(self, img):
        # Otsu's thresholding using skimage's filters
        otsu_thresh = filters.threshold_otsu(img)

        # Apply thresholding
        binary_img = img > otsu_thresh
        return binary_img.astype(int)

    # Morphology processing and hole filling
    def morphological_operations(self, img):
        # Shrink with 10*10 structural elements
        struct_elem_erosion = np.ones((3, 3))

        # Dilation with 5*5 structural elements
        struct_elem_dilation = np.ones((3, 3))

        img = ndimage.binary_erosion(img, structure=struct_elem_erosion, iterations=5)
        img = ndimage.binary_dilation(img, structure=struct_elem_dilation, iterations=17)
        img = ndimage.binary_closing(img, structure=struct_elem_dilation, iterations=2)

        # Filling holes
        img = ndimage.binary_fill_holes(img)
        return img

    # Labeling
    def larger_connected_component(self, img, neighborhood=8):
        # 4 or 8 neighborhood
        struct_element_eight = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        struct_element_four = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]

        if neighborhood == 8:
            labeled, num_features = ndimage.label(img, structure=struct_element_eight)
        else:
            labeled, num_features = ndimage.label(img, structure=struct_element_four)

        # Calculate the size of each connected region
        sizes = ndimage.sum(img, labeled, range(num_features + 1))

        # Extract connected regions that is larger than 'threshold'
        threshold = 500
        mask_size = sizes > threshold
        return mask_size[labeled]

    # Display image
    def display_image(self, image_array=None):
        if image_array is None:
            image_array = self.image_array

        plt.imshow(image_array, cmap='gray')
        plt.colorbar()
        plt.show()

    # Display labeled image
    def display_labeled_image(self, labeled_img):
        plt.figure(figsize=(8, 8))
        plt.imshow(labeled_img, cmap='nipy_spectral')
        plt.colorbar()
        plt.show()


# 3D CT image processing
class CT3DImageProcesser(CTImageProcesser):
    def __init__(self, mhd_file_path):
        super().__init__(mhd_file_path)

    # Thresholding 3D image
    def thresholding_3D_image(self, img, lower_thresh, upper_thresh):
        # Create empty array to save threshed image
        self.thresholded_img = np.zeros_like(img)

        # Threshold all 2D images
        for z in range(img.shape[0]):
            two_dim_slice = img[z, :, :]
            thresholded_slice = self.thresholding(two_dim_slice, lower_thresh, upper_thresh)
            self.thresholded_img[z, :, :] = thresholded_slice
        return self.thresholded_img

    # Morphology processing and hole filling
    def morphological_3D_operations(self, img):
        # Create empty array to save threshed image
        self.morphology_img = np.zeros_like(img)

        for z in range(img.shape[0]):
            two_dim_slice = img[z, :, :]
            # Morphology processing
            morphology_slice = self.morphological_operations(two_dim_slice)

            # Labeling
            morphology_slice = self.larger_connected_component(morphology_slice)

            self.morphology_img[z, :, :] = morphology_slice
        return self.morphology_img

    # 3D labeling
    def larger_connected_component3D(self, img):
        labeled, num_features = ndimage.label(img)

        # Calculate the size of each connected region
        sizes = ndimage.sum(img, labeled, range(num_features + 1))
        max_area = np.argmax(sizes[1:]) + 1
        mask_img = labeled == max_area

        return mask_img



if __name__ == '__main__':
    CT_data_dir_path = '../../data/ChestCT/ChestCT.mhd'
    # ct_img_processer = CTImageProcesser(CT_data_dir_path)
    # ex_img = ct_img_processer.ex_img
    # ex_img_thresh = ct_img_processer.thresholding(ex_img, -200, 400)
    # ex_img_morphology = ct_img_processer.morphological_operations(ex_img_thresh)
    # ex_labeled_img = ct_img_processer.larger_connected_component(ex_img_morphology)

    # ct_img_processer.display_image(ex_labeled_img)at
    # ct_img_processer.display_labeled_image(ex_labeled_img)

    ct_3d = CT3DImageProcesser(CT_data_dir_path)
    thresholded_3D_image = ct_3d.thresholding_3D_image(ct_3d.image_array, -300, 400)
    morphology_3D_image = ct_3d.morphological_3D_operations(thresholded_3D_image)
    labeled_3D_img = ct_3d.larger_connected_component3D(morphology_3D_image)

    # Read mhd file
    data_read = MetaDataRead('../../data/ChestCT/ChestCT.mhd')
    mhd_dict = data_read.read_as_dict()
    mhd_dict['ElementDataFile'] = 'ss2304-02.raw'

    # Write raw file
    writer = MetaImageWriter('../../data/ChestCT')
    writer.save_as_metaimage(image_data=morphology_3D_image, header_info_dic=mhd_dict, save_file_name='ss2304-02')




