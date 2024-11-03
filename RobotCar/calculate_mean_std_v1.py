from itertools import repeat
import os
from glob import glob
from multiprocessing.pool import ThreadPool
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

NUM_THREADS = os.cpu_count()


# mean std for pytorch normalization
def calc_channel_sum(img_path):  # 计算均值的辅助函数，统计单张图像颜色通道和，以及像素数量
    img = np.array(Image.open(img_path).convert('RGB')) / 255.0  # 准换为RGB的array形式
    h, w, _ = img.shape
    pixel_num = h * w
    channel_sum = img.sum(axis=(0, 1))  # 各颜色通道像素求和
    return channel_sum, pixel_num


def calc_channel_var(img_path, mean):  # 计算标准差的辅助函数
    img = np.array(Image.open(img_path).convert('RGB')) / 255.0
    channel_var = np.sum((img - mean) ** 2, axis=(0, 1))
    return channel_var


if __name__ == '__main__':
    # train_path = Path(r'./DATA/train')

    data_path = "./DATA"
    folders = [folder for folder in glob(os.path.join(data_path, "*")) if os.path.basename(folder) in ["train", "val"]]
    for folder in folders:
        train_path = Path(folder)
        # img_f = list(train_path.rglob('*.png'))
        img_f = list(train_path.rglob('*.jpg'))
        n = len(img_f)
        result = ThreadPool(NUM_THREADS).imap(calc_channel_sum, img_f)  # 多线程计算
        channel_sum = np.zeros(3)
        cnt = 0
        pbar = tqdm(enumerate(result), total=n)
        for i, x in pbar:
            channel_sum += x[0]
            cnt += x[1]
        mean = channel_sum / cnt
        print('\n')
        print(f"{os.path.basename(folder)}")
        print("R_mean is %f, G_mean is %f, B_mean is %f" % (mean[0], mean[1], mean[2]))

        result = ThreadPool(NUM_THREADS).imap(lambda x: calc_channel_var(*x), zip(img_f, repeat(mean)))
        channel_sum = np.zeros(3)
        pbar = tqdm(enumerate(result), total=n)
        for i, x in pbar:
            channel_sum += x
        var = np.sqrt(channel_sum / cnt)
        print('\n')
        print("R_var is %f, G_var is %f, B_var is %f" % (var[0], var[1], var[2])) 
        print("- * 50")