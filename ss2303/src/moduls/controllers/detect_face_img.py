import cv2
import csv
import numpy as np
import os


class ImageProcess:
    # Get image. Argument 0: Gray scale, 1: Color(default)
    def __init__(self, input_file, select_color=1):
        # arguments from command line
        self.input_filename: str = input_file
        self.filename = None
        self.output_filename: str = None

        self.input_img = None
        self.drawn_face_img = None
        self.blur_face_img = None
        self.swapped_img = None
        self.changed_color_img = None
        self.faces = None
        self.face_info = None   # Information for writing to csv file
        try:
            # Check file existence
            if not os.path.exists(self.input_filename):
                raise Exception(f'File not found {self.input_filename}')

            # Original the image
            if select_color == 1:
                self.input_img = cv2.imread(self.input_filename)

            # Gray Scale
            else:
                self.input_img = cv2.imread(self.input_filename, cv2.IMREAD_GRAYSCALE)

        except Exception as e:
            print(f'Error: {str(e)}')

        # Get filename
        base_name = os.path.basename(self.input_filename)
        self.filename = os.path.splitext(base_name)[0]

    # Perform face detection, save_path represents the save destination
    def detect_face(self):
        # Get current path
        current_path = os.path.abspath(__file__)
        current_directory = os.path.dirname(current_path)

        trained_model = os.path.join(current_directory, 'face_detection_yunet_2023mar.onnx')
        face_detector = cv2.FaceDetectorYN.create(trained_model, "", (0, 0))

        # Convert to 3 channels if the image is other than 3 channels
        channels = 1 if len(self.input_img.shape) == 2 else self.input_img.shape[2]
        if channels == 1:
            self.input_img = cv2.cvtColor(self.input_img, cv2.COLOR_GRAY2BGR)
        if channels == 4:
            self.input_img = cv2.cvtColor(self.input_img, cv2.COLOR_BGRA2BGR)

        # Specifying input size
        height, width, _ = self.input_img.shape
        face_detector.setInputSize((width, height))

        # Detect faces
        _, faces = face_detector.detect(self.input_img)
        self.faces = faces if faces is not None else []

        # Face is not detected if length of faces is 0
        if len(self.faces) == 0:
            print('No face detected in the input image.')

        # Save face
        self.save_data(kind='face')

    # Draw the bounding box and landmarks of the detected face
    def draw_img(self):
        self.drawn_face_img = self.input_img.copy()
        for face in self.faces:
            # bounding box
            box = list(map(int, face[:4]))
            color = (0, 0, 255)
            thickness = 2
            cv2.rectangle(self.drawn_face_img, box, color, thickness, cv2.LINE_AA)

            # Landmarks of right eye, left eye, nose, right corner of mouth, left corner of mouth
            landmarks = list(map(int, face[4:len(face)-1]))
            landmarks = np.array_split(landmarks, len(landmarks) / 2)
            for landmark in landmarks:
                radius = 5
                thickness = -1
                cv2.circle(self.drawn_face_img, landmark, radius, color, thickness, cv2.LINE_AA)

            # Confidence
            confidence = face[-1]
            confidence = "{:.2f}".format(confidence)
            position = (box[0], box[1] - 10)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.5
            thickness = 2
            cv2.putText(self.drawn_face_img, confidence, position, font, scale, color, thickness, cv2.LINE_AA)

    # Blur the face area
    def blur_face(self, strength=10):
        self.blur_face_img = self.input_img.copy()

        # Calculate ksize and sigmaX based on the strength value
        ksize = (2 * strength + 1, 2 * strength + 1)  # Ensure ksize is odd
        sigmaX = 0.3 * ((ksize[0] - 1) * 0.5 - 1) + 0.8  # A common formula to compute sigma based on ksize

        for face in self.faces:
            x, y, w, h = map(int, face[:4])

            # Cut out the face area
            face_region = self.blur_face_img[y:y+h, x:x+w]

            # Blur the face area
            blurred_face = cv2.GaussianBlur(face_region, ksize, sigmaX)  # (90, 90) is strength of blur
            self.blur_face_img[y:y+h, x:x+w] = blurred_face

    # Swap face with argument that is face image.
    # Select_color = 1 (default): Color, 0: Gray Scale
    def swap_face(self, face_img_path, select_color=1):
        # All detected face is replaced
        try:
            # Check existing the file
            if not os.path.exists(face_img_path):
                raise Exception(f'File not fount {face_img_path}')

            if select_color == 1:
                face_img = cv2.imread(face_img_path)
            else:
                face_img = cv2.imread(face_img_path, cv2.IMREAD_GRAYSCALE)
        except Exception as e:
            print(f'Error : {str(e)}')

        self.swapped_img = self.input_img.copy()
        for face in self.faces:
            x, y, w, h = map(int, face[:4])

            # Resize face image to match the size of detected face
            face_img = cv2.resize(face_img, (w, h))

            # Replace the detected face with the face image
            self.swapped_img[y:y+h, x:x+w] = face_img

    # Changing the saturation of the face area
    def change_color(self, brightness_val):
        self.changed_color_img = self.input_img.copy()
        for face in self.faces:
            # Get face
            x, y, w, h = map(int, face[:4])
            face = self.changed_color_img[y:y+h, x:x+w]

            # Change face color
            brightness_array = np.ones(face.shape, dtype=face.dtype) * brightness_val
            face = cv2.subtract(face, brightness_array)
            print(face.shape)

            # Replace face to input image
            self.changed_color_img[y:y+h, x:x+w] = face

    # Save each processed image
    def save_data(self, kind='original', save_path=None):
        # Specify "saved_img" as the save destination
        # Make directory "saved_img" if it is not exist
        if save_path is None or not os.path.exists(save_path):
            # Get current path
            current_path = os.path.abspath(__file__)
            parent_directory = current_path

            # Move up 4 directory levels
            for _ in range(4):
                parent_directory = os.path.dirname(parent_directory)

            save_path = f'{parent_directory}/image/saved_data'
            os.makedirs(save_path, exist_ok=True)

        # Save the drawn image
        if kind == 'drawn':
            cv2.imwrite(f'{save_path}/{self.filename}.png', self.drawn_face_img)
        elif kind == 'original':
            cv2.imwrite(f'{save_path}/{self.filename}_{kind}.png', self.input_img)
        elif kind == 'blur':
            cv2.imwrite(f'{save_path}/{self.filename}_{kind}.png', self.blur_face_img)
        elif kind == 'face':
            face_save_path = f'{parent_directory}/image/face_img'
            os.makedirs(f'{parent_directory}/image/face_img', exist_ok=True)
            # Save the face area to "face_img" directory
            count_face = len(self.faces)
            for face in self.faces:
                x, y, w, h = map(int, face[:4])
                face_img = self.input_img[y:y + h, x:x + w].copy()
                print(f'{parent_directory}/image/face_img/{self.filename}_face{count_face}.png')
                if count_face > 0:
                    count_face -= 1
                    cv2.imwrite(f'{parent_directory}/image/face_img/{self.filename}_face{count_face}.png', face_img)
                else:
                    cv2.imwrite(f'{parent_directory}/image/face_img/{self.filename}_face.png', face_img)
        elif kind == 'swap':
            cv2.imwrite(f'{save_path}/{self.filename}_swap.png', self.swapped_img)
        elif kind == 'color':
            cv2.imwrite(f'{save_path}/{self.filename}_changed_color.png', self.changed_color_img)
        elif kind == 'csv':
            # Write csv file
            with open(f'{save_path}/{self.filename}.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(self.faces)
        else:
            print('Error: argument that is "kind" is wrong.')


    # kind = original(default), drawn, blur, swap
    def show_img(self, kind="original"):
        if kind == 'drawn':
            cv2.imshow('Drawn image', self.drawn_face_img)
        elif kind == 'blur':
            cv2.imshow('Blur image', self.blur_face_img)
        elif kind == 'swap':
            cv2.imshow('Swapped image', self.swapped_img)
        elif kind == 'color':
            cv2.imshow('Changed face color', self.changed_color_img)
        else:
            cv2.imshow('Input image', self.input_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_drawn_img(self):
        return self.drawn_face_img

    def get_blur_img(self):
        return self.blur_face_img

    def get_swap_img(self):
        return self.swapped_img


if __name__ == '__main__':
    img_path = '../../../image/sample_img/sample.JPG'
    img = ImageProcess(img_path)
    # img = ImageProcess('./sample_img/sample.JPG', 1)
    img.detect_face()
    img.draw_img()
    img.blur_face(14)
    # # img.swap_face('../../image/face_img/sample_face.png')
    img.change_color(50)
    img.show_img(kind='color')
    img.save_data(kind='face')