from datetime import datetime as dt
import os
import numpy as np
import random as rn
import sys

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from sklearn.model_selection import train_test_split, KFold
from torchvision import transforms, datasets, models

from utils import roc, class_weighting
from models import resnet18
from utils.plot_loss import plot_loss, plot_and_save_metrics, plot_learning_curves, plot_accuracy_over_epochs
from utils.Extract_min_frame import get_min_frame_dataset
from utils.evaluate import plot_matrix, process_stacking, test


def train(data_dir, output_path, model_name, model_path,
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

    output_path = '/'.join(model_path.split('/')[:-1])
    print(output_path)

    _fold = 1

    # Determine value of resize depend on model
    resize_val = None
    if model_name == 'ResNet':
        resize_val = 224
    elif model_name == 'GoogleNet':
        resize_val = 299
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

    # Create dataset using ImageFolder
    dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms['val'])

    # Divide training set to training set and test set
    best_validation_accuracy_list = []
    test_acc = 0
    test_acc_list = []
    sensitivity_list = []
    specificity_list = []

    loss_log_file_name = f"{output_path}/loss_log_{time_stamp}_fold{_fold}.csv"
    model_file_name = f"{output_path}/model_{time_stamp}_fold{_fold}.pth"
    roc_curve_file_name = f"{output_path}/validation_roc_{time_stamp}_fold{_fold}.png"

    # Select model
    model = None
    if model_name == 'ResNet':
        model = resnet18.ResNet18()
        print('model : ResNet18')
    elif model_name == 'GoogleNet':
        model = models.googlenet(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 2)
        print(f'model : GoogleNet')
    elif model_name == 'ResNet18':
        model = resnet18.ResNet18()
        print(f'model : ResNet18')
    else:
        model = models.vgg16(pretrained=False)
        # Replace the last layer (fully connected layer) of VGG16 and add a new output layer
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, 2)
        print(f'model : VGG16')

    model.load_state_dict(torch.load(model_path))
    model.eval()
    model = model.to(device)

    test_dataset = torch.utils.data.dataset.Subset(dataset, np.arange(len(dataset)))

    # Get minimum frame for each case
    min_frame_test_dataset = get_min_frame_dataset(dataset, test_dataset)

    test_size = len(min_frame_test_dataset)
    print(f'test size : {test_size}')

    # Create data loader
    test_dataloader = torch.utils.data.DataLoader(min_frame_test_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    pin_memory=True)

    # Adam optimizer exponential moving average coefficient
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(beta_1, 0.999), weight_decay=0.0005)

    criterion = nn.CrossEntropyLoss()

    predicted_list = []
    true_list = []
    score_list = []
    predict_prob = []

    # Temporary, disable automatic differentiation function
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(test_dataloader):
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            score = F.softmax(outputs, dim=1)

            _, predicted = outputs.max(1)
            true_list = np.append(true_list, labels.cpu().numpy().astype(int))
            predicted_list = np.append(predicted_list, predicted.cpu().numpy())
            score_list = np.append(score_list, score.cpu().numpy()[:, 1])
            for i in score.cpu().numpy().tolist():
                predict_prob.append(i)
    plot_matrix(true_list, predicted_list, 'confusion matrix', output_path, 1)

# Save test data
# metrics_list = [test_acc_list, sensitivity_list, specificity_list]
# metric_names = ["Accuracy", "Sensitivity", "Specificity"]
# plot_title = "Test Metrics Over Epochs"
# save_path = f"{output_path}/test_metrics.png"

# plot_and_save_metrics(metrics_list, metric_names, plot_title, save_path)
# #
# return average_validation_acc


if __name__ == "__main__":
    data_dir = '../data/test'
    output_path = '../data/result'
    model_path = '../data/result/202311290658/model_202311290658_best_fold5.pth'
    lr = 0.001
    batch_size = 16
    max_epoch = 500

    # 'ResNet', 'GoogleNet', 'ResNet18', 'vgg16'
    train(data_dir, model_name='GoogleNet',
                        output_path=output_path,
                        model_path=model_path,
                        learning_rate=lr,
                        batch_size=batch_size,
                        max_epoch=max_epoch)
    # print(f"AUC value of validation data: {ave_val_acc:.3f}")