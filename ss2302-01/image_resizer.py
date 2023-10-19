from PIL import Image
import argparse
import numpy as np


class ImageProcess:
    # Get input image file name, output image file name, output image size of width and height,
    # cropping/padding base point (0~4) from command line
    def __init__(self, input_file, output_file):
        # arguments from command line
        self.input_filename: str = input_file
        self.output_filename: str = output_file
        self.output_pixel_size: tuple(int, int) = None
        self.base_point: int = None

        self.input_pixel_size: tuple(int, int) = None
        self.input_gray_img_array = None
        self.output_img_array = None
        self.output_gray_img = None

        # Get image file input from command line in grayscale
        try:
            input_img = Image.open(self.input_filename)
            self.input_gray_img = input_img.convert('L')

            # convert numpy array
            self.input_gray_img_array = np.array(self.input_gray_img)
        except FileNotFoundError:
            print(f'Error: Input file {self.input_filename} not found.')
        except Exception as e:
            print(f'An error occurred while processing the image: {str(e)}')

    # change image size depending on pixel_size
    # 0(default): center, 1: upper lift, 2: lower left, 3: lower right, 4: upper right
    def resizer(self, output_pixel_size: tuple, base_point=0):
        self.output_pixel_size = output_pixel_size
        self.base_point = base_point
        self.input_pixel_size = self.input_gray_img.size

        # decide padding or cropping
        if self.input_pixel_size[0] < self.output_pixel_size[0] or self.input_pixel_size[1] < self.output_pixel_size[1]:
            operation: str = 'padding'
        else:
            operation: str = 'cropping'
        print(operation)
        # Image initialization
        if operation == 'padding':
            self.output_img_array = np.zeros((self.output_pixel_size[0], self.output_pixel_size[1]), dtype=np.uint8)
        else:
            self.output_img_array = np.copy(self.input_gray_img)

        # base_point is center
        if self.base_point == 0:
            if operation == 'padding':
                top = (self.output_pixel_size[1] - self.input_pixel_size[1]) // 2
                left = (self.output_pixel_size[0] - self.input_pixel_size[0]) // 2
                self.output_img_array[
                    top:top + self.input_gray_img_array.shape[1],
                    left:left + self.input_gray_img_array.shape[0]
                ] = self.input_gray_img_array.copy()
            else:
                top = (self.input_pixel_size[1] - self.output_pixel_size[1]) // 2
                left = (self.input_pixel_size[0] - self.output_pixel_size[0]) // 2
                cropped_img = self.input_gray_img_array[
                              top:top + self.output_pixel_size[1],
                              left:left + self.output_pixel_size[0]
                              ]
                self.output_img_array = cropped_img.copy()

        # base point is upper left
        elif self.base_point == 1:
            if operation == 'padding':
                self.output_img_array[0:self.input_gray_img_array.shape[1], 0:self.input_gray_img_array.shape[0]] \
                    = self.input_gray_img_array.copy()
            else:
                cropped_img = self.input_gray_img_array[0:self.output_pixel_size[1], 0:self.output_pixel_size[0]]
                self.output_img_array = cropped_img.copy()

        # base point is lower left
        elif self.base_point == 2:
            if operation == 'padding':
                top = (self.output_pixel_size[1] - self.input_pixel_size[1])
                self.output_img_array[
                    top:top + self.input_gray_img_array.shape[1], 0:self.input_gray_img_array.shape[0]] \
                    = self.input_gray_img_array.copy()
            else:
                top = self.input_pixel_size[1] - self.output_pixel_size[1]
                cropped_img = self.input_gray_img_array[top:self.input_pixel_size[1], 0:self.output_pixel_size[0]]
                self.output_img_array = cropped_img.copy()

        # base point is lower right
        elif self.base_point == 3:
            if operation == 'padding':
                top = self.output_pixel_size[1] - self.input_pixel_size[1]
                right = self.output_pixel_size[0] - self.input_pixel_size[0]
                self.output_img_array[
                    top:top + self.input_gray_img_array.shape[1], right:right + self.input_pixel_size[0]] \
                    = self.input_gray_img_array.copy()
            else:
                top = self.input_pixel_size[1] - self.output_pixel_size[1]
                right = self.input_pixel_size[0] - self.output_pixel_size[0]
                cropped_img = self.input_gray_img_array[
                              top:top + self.output_pixel_size[1],
                              right:right + self.output_pixel_size[0]]
                self.output_img_array = cropped_img.copy()

        # base point is upper right
        elif self.base_point == 4:
            if operation == 'padding':
                left = self.output_pixel_size[1] - self.input_pixel_size[1]
                self.output_img_array[
                    0:self.input_pixel_size[1], left:left + self.input_pixel_size[0]] \
                    = self.input_gray_img_array
            else:
                left = self.input_pixel_size[0] - self.output_pixel_size[0]
                cropped_img = self.input_gray_img_array[
                              0:self.output_pixel_size[1], left:left + self.output_pixel_size[0]]
                self.output_img_array = cropped_img.copy()

        self.output_gray_img = Image.fromarray(self.output_img_array)
        self.output_gray_img.save(f'./saved_img/{self.output_filename}')
        self.output_gray_img.show()


def get_from_commandline():
    parser = argparse.ArgumentParser(description='Input some information')

    # The following arguments have the format (variable name, type, help string)
    parser.add_argument('input_filename', type=str, help='Input image name of file')
    parser.add_argument('output_filename', type=str, help='Output image name of file')
    parser.add_argument('pixel_size', type=int, nargs=2, metavar=('width', 'height'),
                        help='Output image pixel size')
    parser.add_argument('base_point', type=int, help='Base point for cropping or padding (0~4)')

    # check if there are enough arguments from the command line
    try:
        args = parser.parse_args()
        input_filename = args.input_filename
        output_filename = args.output_filename
        pixel_size = tuple(args.pixel_size) if args.pixel_size else None
        base_point = args.base_point
    except argparse.ArgumentError as e:
        print(f'Command line argument error: {e}')
        return

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_from_commandline()
    input_filename = args.input_filename
    output_filename = args.output_filename
    output_pixel_size = tuple(args.pixel_size)
    base_point = args.base_point
    print(input_filename, output_filename, output_pixel_size, base_point)

    img = ImageProcess(input_filename, output_filename)
    img.resizer(output_pixel_size, base_point)
