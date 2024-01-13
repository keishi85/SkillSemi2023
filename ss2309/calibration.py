from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# チェスボードを撮影した画像があるディレクトリ
img_dir = Path("Camera")

# 画像を読み込む。
samples = []
for path in img_dir.glob("*.jpg"):
    img = cv2.imread(str(path))
    samples.append(img)

cols = 9  # 列方向の交点数
rows = 6  # 行方向の交点数

fig = plt.figure(figsize=(10, 14), facecolor="w")

# チェスボードのマーカー検出を行う。
img_points = []
for i, img in enumerate(samples, 1):
    # 画像をグレースケールに変換する。
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # チェスボードの交点を検出する。
    ret, corners = cv2.findChessboardCorners(img, (cols, rows))

    if ret:  # すべての交点の検出に成功
        img_points.append(corners)
    else:
        print("Failed to detect chessboard corners.")

    # 検出結果を描画する。
    dst = cv2.drawChessboardCorners(img.copy(), (cols, rows), corners, ret)

    ax = fig.add_subplot(5, 3, i)
    ax.set_axis_off()
    ax.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))

# 検出した画像座標上の点に対応する3次元上の点を作成する。
world_points = np.zeros((rows * cols, 3), np.float32)
world_points[:, :2] = np.mgrid[:cols, :rows].T.reshape(-1, 2)

# 画像の枚数個複製する。
object_points = [world_points] * len(samples)

h, w, c = samples[0].shape
ret, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(object_points, img_points, (w, h), None, None)

print("reprojection error:\n", ret)
print("camera matrix:\n", camera_matrix)
print("distortion:\n", distortion)
print("rvecs:\n", rvecs[0].shape)
print("tvecs:\n", tvecs[0].shape)
