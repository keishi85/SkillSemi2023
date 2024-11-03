import argparse
import numpy as np
import os
from datetime import datetime
import tqdm

import torch
import torchvision.models as models
import torch.nn as nn
from torch.optim import Adam

from utils import get_model, get_data
from plot import plot_loss


def argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default="./DATA")
    parser.add_argument("--model_name", type=str, default="resnet18")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("custom_model_path", type=str, default=None, help="Path to the custom model to load.")

    return parser

def set_seed(seed):
    torch.backends.cudnn.deterministic = True   # 再現可能にする設定
    torch.backends.cudnn.benchmark = False      # ベンチマークモードをOFFにする(異なるアルゴリズムを試して最適なものを見つける)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def eval(val_loader, critirion, model, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for img, label in val_loader:
            img = img.to(device)
            label = label.to(device)

            output = model(img)
            loss = critirion(output, label.float())
            total_loss += loss.item()
    
    # 精度ではなく，平均損失を返す
    avg_loss = total_loss / len(val_loader)
    return avg_loss


def train(args):
    data_path = args.data_path
    model_name = args.model_name
    img_size = args.img_size
    batch_size = args.batch_size
    num_epochs = args.num_epochs

    time_stamp = datetime.now().strftime("%m%d%H%M")
    save_model_name = f"{model_name}_{time_stamp}.pth"
    save_model_path = os.path.join(os.path.join(os.path.dirname(data_path), "model"), save_model_name)
    print(f"save_model_path : {save_model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device : {device}")

    model = get_model(model_name)
    model.to(device)
    model.train()

    train_loader, val_loader = get_data(data_path, img_size, batch_size)

    optimizer = Adam(model.parameters(), lr=0.0001)

    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []
    min_val_loss = np.inf

    for epoch in range(num_epochs):
        train_loss = 0
        for img, label in tqdm.tqdm(train_loader):
            img = img.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            output = model(img)

            loss = criterion(output, label.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        #  学習セット，検証セットでの性能評価
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        avg_val_loss = eval(val_loader, criterion, model, device)
        val_losses.append(avg_val_loss)

        # 検証セットでの性能が良い場合，モデルを保存
        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_model_path)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # モデルの保存
    torch.save(model.state_dict(), save_model_path)

    # Losssのプロット
    plot_loss(train_losses, val_losses, os.path.join(data_path, "loss3.png"))


"""
実行 : python train.py --data_path ./DATA --img_size 224 --batch_size 32
"""
if __name__  == "__main__":
    args = argument_parser().parse_args()
    print(args)

    set_seed(args.seed)
    train(args)