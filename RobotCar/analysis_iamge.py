import cv2
import os
import numpy as np
from glob import glob

ORIGINAL_IMAGES_PATH = "./ORIGINAL_IMAGES"
# OUTPUT_IMAGES_PATH = "./edges_images"
OUTPUT_IMAGES_PATH = "./contours_images"
# OUTPUT_IMAGES_PATH = "./ex"

THRESHOLD = 104
THRESHOLD1 = 20
THRESHOLD2 = 200

def image_processing(dir_path, output_dir_path):
    for image_path in glob(f"{dir_path}/*.jpg"):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, THRESHOLD, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(binary, THRESHOLD1, THRESHOLD2) # 50:閾値1, 150:閾値2
        
        # cv2.imwrite(f"{output_dir_path}/{os.path.basename(image_path)}", edges)

        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 最大輪郭の検出
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)

            # 空の画像を作成して最大輪郭を描画
            # image_to_show = np.zeros_like(gray)
            cv2.drawContours(gray, [largest_contour], -1, (255, 255, 0), 2)
            cv2.imwrite(f"{output_dir_path}/{os.path.basename(image_path)}", gray)


if __name__ == "__main__":
    image_processing(ORIGINAL_IMAGES_PATH, OUTPUT_IMAGES_PATH)