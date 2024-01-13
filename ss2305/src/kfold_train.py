from datetime import datetime as dt
import os
import numpy as np
import random as rn
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, KFold
from torchvision import transforms, datasets, models

from utils.Dataset import CustomDataset
from utils import roc, class_weighting
from models import resnet18
from utils.plot_loss import plot_loss, plot_and_save_metrics, plot_learning_curves, plot_accuracy_over_epochs
from utils.Extract_min_frame import get_min_frame_dataset


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
    dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms['train'])

    # 5 fold cross validation

    kf = KFold(n_splits=5)
    seed = 0

    # Divide dataset to train_dataset and test_dataset
    # train_val_index and test_index are list
    train_val_index, test_index = train_test_split(np.arange(len(dataset)), test_size=0.2, random_state=seed)

    # Divide training set to training set and test set
    best_validation_accuracy_list = []
    test_acc = 0
    test_acc_list = []
    sensitivity_list = []
    specificity_list = []

    for _fold, (train_index, val_index) in enumerate(kf.split(train_val_index)):

        loss_log_file_name = f"{output_path}/loss_log_{time_stamp}_fold{_fold}.csv"
        model_file_name = f"{output_path}/model_{time_stamp}_fold{_fold}.pth"
        roc_curve_file_name = f"{output_path}/validation_roc_{time_stamp}_fold{_fold}.png"

        training_losses = []
        validation_losses = []
        training_accuracies = []
        validation_accuracies = []

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
            model = models.vgg16(pretrained=True)
            # Replace the last layer (fully connected layer) of VGG16 and add a new output layer
            num_features = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_features, 2)
            print(f'model : VGG16')

        model = model.to(device)

        # Convert training and validation set indices to actual indices
        train_index = train_val_index[train_index]
        val_index = train_val_index[val_index]

        # The train dataset rotates the 'negative' class image data by 90, 180, and 270 degrees,
        # increasing the number of images by a factor of 4.
        train_dataset = torch.utils.data.dataset.Subset(dataset, train_index)
        train_dataset = CustomDataset(root=data_dir,
                                      negative_transform=data_transforms['negative'],
                                      indices=train_dataset.indices,
                                      transform=data_transforms['train'],)

        val_dataset = torch.utils.data.dataset.Subset(dataset, val_index)
        test_dataset = torch.utils.data.dataset.Subset(dataset, test_index)

        # Get minimum frame for each case
        min_frame_test_dataset = get_min_frame_dataset(dataset, test_dataset)

        train_size = len(train_dataset)
        val_size = len(val_dataset)
        test_size = len(min_frame_test_dataset)
        print(f'train size : {train_size}, validation size : {val_size}, test size : {test_size}')

        # Create data loader
        train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=batch_size,
                                                       shuffle=True,
                                                       pin_memory=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                     batch_size=batch_size,
                                                     shuffle=True,
                                                     pin_memory=True)
        test_dataloader = torch.utils.data.DataLoader(min_frame_test_dataset,
                                                      batch_size=batch_size,
                                                      shuffle=False,
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
        best_validation_acc = 0.0
        train_acc_list = []
        val_acc_list = []
        train_acc = 0
        val_acc= 0
        for epoch in range(max_epoch):

            training_loss = 0
            validation_loss = 0

            probabilities = np.zeros(val_size)
            abnormal_labels = np.zeros(val_size)

            # training
            model.train()
            training_acc = 0
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
                train_acc = prediction.eq(labels.data.view_as(prediction)).sum().item()

            avg_training_loss = training_loss / (batch_idx + 1)

            # validation
            model.eval()
            validation_acc = 0

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
                    val_acc = prediction.eq(labels.data.view_as(prediction)).sum().item()

            avg_validation_loss = validation_loss / (batch_idx + 1)



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

        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)

        plot_loss(loss_log_file_name, output_path, _fold)
        # Plot
        plot_accuracy_over_epochs(training_accuracies, validation_accuracies, f'{output_path}/acc_graph{_fold}.png')

        plot_learning_curves(
            training_losses, validation_losses,
            training_accuracies, validation_accuracies,
            f"{output_path}/learning_curves{_fold}.png"
        )

        # test
        probabilities = np.zeros(test_size)
        abnormal_labels = np.zeros(test_size)

        # Temporary, disable automatic differentiation function
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(test_dataloader):
                abnormal_labels[batch_idx] = labels[0]  # Correct label
                data, labels = data.to(device), labels.to(device)

                outputs = model(data)
                probabilities[batch_idx] = outputs.cpu()[0, 1]  # prediction label

                pred = outputs.argmax(dim=1, keepdim=True)
                test_acc += pred.eq(labels.data.view_as(pred)).sum().item()
            auc, tpr, fpr, cutoff_threshold = roc.roc_analysis(probabilities, abnormal_labels, roc_curve_file_name)
            test_acc = test_acc / test_size
            specificity = 1.0 - tpr

            test_acc_list.append(test_acc)
            sensitivity_list.append(tpr)
            specificity_list.append(specificity)
            print(f"Accuracy : {test_acc}, Sensitivity: {tpr:.3f}, Specificity: {1.0 - tpr:.3f} ({cutoff_threshold})")

    average_validation_acc = np.mean(best_validation_accuracy_list)
    print(f'Average validation accuracy: {average_validation_acc:.3f}')

    best_fold = np.argmax(best_validation_accuracy_list)
    best_model_path = f"{output_path}/model_{time_stamp}_best_fold{best_fold}.pth"
    # Rename the best model
    if os.path.exists(f"{output_path}/model_{time_stamp}_fold{best_fold}.pth"):
        os.rename(f"{output_path}/model_{time_stamp}_fold{best_fold}.pth", best_model_path)
    else:
        print(f"File not found: {model_file_name}")

    # Save test data
    metrics_list = [test_acc_list, sensitivity_list, specificity_list]
    metric_names = ["Accuracy", "Sensitivity", "Specificity"]
    plot_title = "Test Metrics Over Epochs"
    save_path = f"{output_path}/test_metrics.png"

    plot_and_save_metrics(metrics_list, metric_names, plot_title, save_path)

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


# Todo : lossにおいてbatch_sizeで割るのを修正

if __name__ == "__main__":
    data_dir = '../data/IVUS'
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