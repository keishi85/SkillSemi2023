import numpy as np
import math
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms

def load_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def draw_markers_on_image(img_path, markers_data_json):
    image = Image.open(img_path)
    draw = ImageDraw.Draw(image)
    markers_data_json = load_json_file(markers_data_json)

    for marker in markers_data_json:
        # get coordinate
        center_x = marker['center']['x']
        center_y = marker['center']['y']
        from_x = marker['from']['x']
        from_y = marker['from']['y']
        to_x = marker['to']['x']
        to_y = marker['to']['y']
        deg = marker['deg']

        # Select color depend on 'type'
        c = marker['type']
        color = 'red' if c == 'C' else 'yellow' if c == 'B' else 'blue'

        # Calculate radius, bounding box and start angle
        radius1 = np.sqrt((to_y - center_y) ** 2 + (to_x - center_x) ** 2)
        radius2 = np.sqrt((from_y - center_y) ** 2 + (from_x - center_x) ** 2)
        radius = int((radius1 + radius2) / 2)
        print(f'first : {radius}, second : {radius2}')
        bbox = [center_x - radius, center_y - radius, center_x + radius, center_y + radius]
        angle_rad = math.atan2((from_y - center_y), (from_x - center_x))
        angle_deg = math.degrees(angle_rad)
        angle_deg = angle_deg % 360

        # Draw result of segmentation
        draw.arc(bbox, start=angle_deg, end=(angle_deg+deg), fill=color, width=2)

        # Draw points and lines for debugging
        radius = 5
        left_up_point = (center_x - radius, center_y - radius)
        right_down_point = (center_x + radius, center_y + radius)
        left_up_point2 = (to_x - radius, to_y - radius)
        right_down_point2 = (to_x + radius, to_y + radius)
        left_up_point3 = (from_x - radius, from_y - radius)
        right_down_point3 = (from_x + radius, from_y + radius)

        # 円（点）を描画
        draw.ellipse([left_up_point, right_down_point], fill=color)
        draw.ellipse([left_up_point2, right_down_point2], fill=color)
        draw.ellipse([left_up_point3, right_down_point3], fill=color)


        # draw.point((from_x, from_y), fill=color)
        # draw.point((to_x, to_y), fill=color)
        # draw.line((center_x, center_y, from_x, from_y), fill=color)
        # draw.line((center_x, center_y, to_x, to_y), fill=color)

    print(markers_data_json[0].keys())
    print(markers_data_json[0].values())
    print()
    image.show()



    # Todo: 画像データは(512, 512, 3)



if __name__ == '__main__':
    img_path = '../../data/anonymized_list/CHIBAMI_49_pre/frame_4020.png'
    json_path = '../../data/anonymized_list/CHIBAMI_49_pre/CHIBAMI_49_pre.json'
    draw_markers_on_image(img_path, json_path)


