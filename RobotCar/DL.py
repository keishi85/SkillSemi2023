import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import cv2
import numpy as np

# モデルのロード（事前学習済み）
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()  # 評価モードに設定

# 画像のロードと前処理
def load_image(image_path):
    image = Image.open(image_path)
    image = F.to_tensor(image)
    return image

# 物体検出を行い、結果を表示
def detect_objects(image_path):
    image = load_image(image_path)
    predictions = model([image])

    # predictionsはリストの辞書で、キーには'boxes', 'labels', 'scores'が含まれる
    predictions = predictions[0]

    # 画像をNumPy配列に変換
    image_np = cv2.imread(image_path)
    for i in range(len(predictions['boxes'])):
        score = predictions['scores'][i].item()
        if score > 0.8:  # スコアが0.8以上のもののみを表示
            box = predictions['boxes'][i].detach().numpy().astype(np.int32)
            label = predictions['labels'][i].item()
            cv2.rectangle(image_np, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 3)
            cv2.putText(image_np, str(label), (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 検出結果を表示
    cv2.imshow('Detection', image_np)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 画像パスを指定
image_path = './ORIGINAL_IMAGES/image_240.jpg'
detect_objects(image_path)
