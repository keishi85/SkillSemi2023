from datetime import datetime as dt
import os
import numpy as np
import random as rn
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split, KFold, GroupKFold
from torchvision import transforms, datasets, models
from torchvision.models import googlenet, GoogLeNet_Weights, resnet50, ResNet50_Weights

from utils import roc, class_weighting
from models import resnet18
from utils.plot_loss import plot_loss, plot_and_save_metrics, plot_learning_curves, plot_accuracy_over_epochs
from utils.check_dataset import count_path
from utils.create_dataset import extract_case_number

"""
IVUS
  |- positive
    |- 98
      |- ex.png
  |- negative
"""


def train(data_dir, output_path, model_name='', class_weight=False,
          learning_rate=0.0001, beta_1=0.99, batch_size=16, max_epoch=50,
          gpu_id="0", time_stamp=""):
    # Fix seed
    seed_num = 234567
    os.environ['PYTHONHASHSEED'] = '0'
    rn.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.backends.cudnn.deterministic = True

    if not os.path.isdir(data_dir):
        print(f"Error: {data_dir} is not found.")
        sys.exit(1)

    # Automatic creation of output folder
    if not os.path.isdir(output_path):
        print(f"Path of output data ({output_path}) is created automatically.")
        output_path += f'_{model_name}'
        os.makedirs(output_path)

    # Set ID of CUDA device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if time_stamp == "":
        time_stamp = dt.now().strftime('%Y%m%d%H%M')


    output_path = os.path.join(output_path, time_stamp)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    loss_log_file_name = f"{output_path}/loss_log_{time_stamp}_fold"
    model_file_name = f"{output_path}/model_best_{time_stamp}_fold"
    roc_curve_file_name = f"{output_path}/validation_roc_{time_stamp}_fold"

    # Determine value of resize depend on model
    resize_val = None
    if model_name == 'ResNet':
        resize_val = 224
    elif model_name == 'GoogleNet':
        resize_val = 224
    else:
        resize_val = 224

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(resize_val),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(180),
            transforms.ToTensor(),
            transforms.Normalize([0.09, 0.09, 0.09], [0.14, 0.14, 0.14]),
        ]),
        'val': transforms.Compose([
            transforms.Resize(resize_val),
            transforms.ToTensor(),
            transforms.Normalize([0.09, 0.09, 0.09], [0.14, 0.14, 0.14]),
        ]),
        'negative': transforms.Compose([
            transforms.CenterCrop(224),
        ])
    }

    # train, val, test path
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    # Create dataset using ImageFolder
    train_dataset = datasets.ImageFolder(root=train_dir, transform=data_transforms['train'])
    val_dataset = datasets.ImageFolder(root=val_dir, transform=data_transforms['val'])

    # Divide training set to training set and test set
    best_validation_accuracy_list = []

    loss_log_file_name = f"{output_path}/loss_log_{time_stamp}_fold.csv"
    model_file_name = f"{output_path}/model_{time_stamp}_fold.pth"
    roc_curve_file_name = f"{output_path}/validation_roc_{time_stamp}_fold.png"

    # Select model
    model = None
    if model_name == 'ResNet18':
        model = resnet18.ResNet18()
        print('model : ResNet18')
    elif model_name == 'ResNet50':
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 2)
        print('model : ResNet50')
    elif model_name == 'GoogleNet':
        model = googlenet(weights=GoogLeNet_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 2)
        print(f'model : GoogleNet')
    elif model_name == 'ResNet18':
        model = resnet18.ResNet18()
        print(f'model : ResNet18')
    else:
        model = models.vgg16(pretrained=True)
        # Replace the last layer (fully connected layer) of VGG16 and add a new output layer
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, 2)
        print(f'model : VGG16')

    model = model.to(device)

    training_losses = []
    validation_losses = []
    training_accuracies = []
    validation_accuracies = []

    # トレーニングデータセットのクラスごとのデータ数をカウント
    train_labels = [train_dataset.targets[i] for i in np.arange(len(train_dataset))]
    train_class_counts = defaultdict(int)
    for label in train_labels:
        train_class_counts[label] += 1
    print(f'train_class_counts : {train_class_counts}')

    # 検証データセットのクラスごとのデータ数をカウント
    val_labels = [val_dataset.targets[i] for i in np.arange(len(val_dataset))]
    val_class_counts = defaultdict(int)
    for label in val_labels:
        val_class_counts[label] += 1
    print(f'val_class_counts : {val_class_counts}')

    if class_weight:
        print('class weight : True')
    else:
        print('class weight : False')

    train_size = len(train_dataset)
    val_size = len(val_dataset)
    print(f'train size : {train_size}, validation size : {val_size}')

    # Create data loader
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 pin_memory=True)

    # Adam optimizer exponential moving average coefficient
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(beta_1, 0.999), weight_decay=0.0005)

    criterion = None
    if class_weight == True:
        class_weights = class_weighting.get_class_weights(dataset=train_dataset)
        weights = torch.tensor([class_weights[i] for i in range(len(class_weights))], dtype=torch.float)
        weighs = weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=weighs)

    else:
        criterion = nn.CrossEntropyLoss()

    best_validation_loss = float('inf')
    for epoch in range(max_epoch):

        training_loss = 0
        validation_loss = 0

        probabilities = np.zeros(val_size)
        abnormal_labels = np.zeros(val_size)

        # training
        model.train()
        training_acc = 0
        count = 0
        for batch_idx, (data, labels) in enumerate(train_dataloader):
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(data)

            loss = criterion(outputs, labels)
            training_loss += loss.item()

            loss.backward()
            optimizer.step()

            prediction = outputs.argmax(dim=1, keepdims=True)
            training_acc += prediction.eq(labels.data.view_as(prediction)).sum().item()

            count += 1

        avg_training_loss = training_loss / count

        # validation
        model.eval()
        validation_acc = 0
        count_val = 0

        with torch.no_grad():

            for batch_idx, (validation_data, validation_labels) in enumerate(val_dataloader):
                abnormal_labels[batch_idx] = validation_labels[0]
                data, labels = validation_data.to(device), validation_labels.to(device)

                outputs = model(data)

                probabilities[batch_idx] = outputs.cpu()[0, 1]

                loss = criterion(outputs, labels)
                validation_loss += loss.item()

                prediction = outputs.argmax(dim=1, keepdims=True)
                validation_acc += prediction.eq(labels.data.view_as(prediction)).sum().item()

                count_val += 1

        avg_validation_loss = validation_loss / count_val

        if best_validation_loss > avg_validation_loss:
            best_validation_loss = avg_validation_loss
            torch.save(model.state_dict(), model_file_name)
            best_validation_acc = roc.roc_analysis(probabilities, abnormal_labels, roc_curve_file_name)
            best_validation_accuracy_list.append(best_validation_acc)

            saved_str = '==> model saved'
        else:
            saved_str = ''

        print("epoch %d: training_loss:%.4f validation_loss:%.4f validation_accuracy=%.3f%s" %
              (epoch + 1, avg_training_loss, avg_validation_loss,
               100.0 * validation_acc / len(val_dataloader.dataset),
               saved_str))

        training_losses.append(avg_training_loss)
        validation_losses.append(avg_validation_loss)
        training_accuracies.append(training_acc / train_size)
        validation_accuracies.append(validation_acc / val_size)

        with open(loss_log_file_name, 'a') as fp:
            fp.write(f'{epoch + 1},{avg_training_loss},{avg_validation_loss}\n')

    plot_loss(loss_log_file_name, output_path, 1)
    # Plot
    plot_accuracy_over_epochs(training_accuracies, validation_accuracies, f'{output_path}/acc_graph.png')

    plot_learning_curves(
        training_losses, validation_losses,
        training_accuracies, validation_accuracies,
        f"{output_path}/learning_curves.png"
    )

    average_validation_acc = np.mean(best_validation_accuracy_list)
    print(f'Average validation accuracy: {average_validation_acc:.3f}')

    best_fold = np.argmax(best_validation_accuracy_list)
    best_model_path = f"{output_path}/model_{time_stamp}_best_fold{best_fold}.pth"
    # Rename the best model
    if os.path.exists(f"{output_path}/model_{time_stamp}_fold{best_fold}.pth"):
        os.rename(f"{output_path}/model_{time_stamp}_fold{best_fold}.pth", best_model_path)
    else:
        print(f"File not found: {model_file_name}")

    return average_validation_acc


def random_search(train_func):
    # Define range of hyper parameters
    param_distributions = {
        'learning_rate': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
        'beta_1': [0.8, 0.85, 0.9, 0.95, 0.99],
        'batch_size': [16, 32, 64, 128],
    }

    # Number of times
    n_iter = 10

    best_score = 0
    best_params = {}

    for i in range(n_iter):
        # ランダムにパラメータを選択
        params = {k: np.random.choice(v) for k, v in param_distributions.items()}
        print(f"Trial {i + 1}: {params}")

        # 訓練を行う
        score = train(data_dir, output_path, learning_rate=params['learning_rate'],
                      beta_1=params['beta_1'], batch_size=params['batch_size'], max_epoch=50)

        # スコアが改善された場合、パラメータを更新
        if score > best_score:
            best_score = score
            best_params = params

if __name__ == "__main__":
    data_dir = '../data/Data'
    output_path = '../data/result'
    lr = 0.001
    batch_size = 16
    max_epoch = 50

    # 'ResNet', 'GoogleNet', 'ResNet18', 'vgg16'
    ave_val_acc = train(data_dir, model_name='GoogleNet', output_path=output_path,
                        learning_rate=lr,
                        class_weight=True,
                        batch_size=batch_size,
                        max_epoch=max_epoch)
    # print(f"AUC value of validation data: {ave_val_acc:.3f}")