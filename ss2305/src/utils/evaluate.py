'''Train CIFAR10 with PyTorch.'''
from ast import Pass
from sqlalchemy import true
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import os
import argparse
from PIL import Image
# from utils import progress_barroc_curve, auc
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sn  # 画图模块
import numpy as np
import random
# from utils import progress_bar
from sklearn.metrics import accuracy_score
import math
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score
from sklearn.metrics import matthews_corrcoef  # MCC
from sklearn.metrics import precision_recall_curve  # PRcurve
from imblearn.metrics import sensitivity_score, geometric_mean_score, specificity_score
from sklearn import svm
from sklearn.preprocessing import StandardScaler


class Custom_DataSet(torch.utils.data.Dataset):
    def __init__(self, img_list, transforms=None):
        self.img_list = img_list
        self.transform = transforms

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        image_path = self.img_list[index][0]
        image = Image.open(image_path)

        if self.transform is not None:
            image = self.transform(image)

        label = self.img_list[index][1]
        return image, label


def plot_matrix(y_true, y_pred, title_name, save_path, num):
    cm = confusion_matrix(y_true, y_pred)  # 混淆矩阵
    # annot = True 格上显示数字 ，fmt：显示数字的格式控制
    ax = sn.heatmap(cm, cmap="Blues", cbar=False, annot=True, fmt='g', xticklabels=[
        'Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    # xticklabels、yticklabels指定横纵轴标签
    ax.set_title(title_name)  # 标题
    ax.set_xlabel('Predict')  # x轴
    ax.set_ylabel('True')  # y轴
    save_path += f"/confusion_matrix{num}.png"
    plt.savefig(save_path)
    plt.close()


def roc_analysis(labels, likelihoods):
    # sort by likelihood (descending order)
    sorted_idx = np.argsort(-likelihoods)
    binary_array = labels[sorted_idx]
    sorted_likelihood = likelihoods[sorted_idx]

    # ROC analysis
    fpr = np.cumsum(1 - binary_array) / np.sum(1 - binary_array)
    tpr = np.cumsum(binary_array) / np.sum(binary_array)
    auc = np.sum(tpr * (1 - binary_array)) / np.sum(1 - binary_array)

    # Add (0,0) for plotting
    fpr_for_plot = np.insert(fpr, 0, 0.0)
    tpr_for_plot = np.insert(tpr, 0, 0.0)

    # Get cut-off point by Youden index
    cutoff_idx = np.argmax(tpr - fpr)
    max_accuracy = (np.sum(binary_array[0:cutoff_idx + 1] == 1)
                    + np.sum(binary_array[cutoff_idx + 1:] == 0)) / len(tpr)

    # Plot ROC curve
    plt.plot(fpr_for_plot, tpr_for_plot, 'darkorange',
             linewidth=2.0, clip_on=False)

    base_points = np.array([0, 1])
    plt.plot(base_points, base_points, 'k', linestyle='dotted')
    plt.plot(fpr[cutoff_idx], tpr[cutoff_idx], 'o', ms=10, markeredgewidth=2,
             markerfacecolor="None", markeredgecolor='k')

    plt.xlabel('False positive rate (FPR)')
    plt.ylabel('True positive rate (TPR)')

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.text(0.5, 0.3, 'AUC=%.3f' % (auc), size=15)

    plt.savefig("./evaluation_result/auc_curve.png")
    plt.close()

    return auc, tpr[cutoff_idx], fpr[cutoff_idx], sorted_likelihood[cutoff_idx]


def model_evaluate(net):
    net.eval()
    predicted_list = []
    true_list = []
    score_list = []
    predict_prob = []

    with torch.no_grad():
        for _, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            score = F.softmax(outputs, dim=1)

            _, predicted = outputs.max(1)

            true_list = np.append(true_list, targets.cpu().numpy().astype(int))
            predicted_list = np.append(predicted_list, predicted.cpu().numpy())
            score_list = np.append(score_list, score.cpu().numpy()[:, 1])
            for i in score.cpu().numpy().tolist():
                predict_prob.append(i)

    return true_list, predicted_list, predict_prob


def test(net):
    net.eval()
    predicted_list = []
    true_list = []
    score_list = []
    predict_prob = []

    with torch.no_grad():
        for _, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            score = F.softmax(outputs, dim=1)

            _, predicted = outputs.max(1)

            true_list = np.append(true_list, targets.cpu().numpy().astype(int))
            predicted_list = np.append(predicted_list, predicted.cpu().numpy())
            score_list = np.append(score_list, score.cpu().numpy()[:, 1])
            for i in score.cpu().numpy().tolist():
                predict_prob.append(i)

    return true_list, predicted_list, predict_prob


def train_test_percent(benign_case, malignant_case, benign_dict, malignant_dict, image_nums):
    benign_total = 0
    malignant_total = 0

    for case in benign_case:
        benign_total += len(benign_dict[case])

    for case in malignant_case:
        malignant_total += len(malignant_dict[case])

    return (0.5 * image_nums / benign_total), (0.5 * image_nums / malignant_total)


def get_all_case_data(data_path, aug_path, batch_aug_path, aug_list, malignant_case_dict, benign_case_dict):
    # 返回两个字典
    # 一个字典 是存储benign case dataset
    # 一个字典是存储 maligant case dataset
    classes_list = os.listdir(data_path)
    # print(batch_aug)
    # print(classes_list)
    # exit()
    benign_dict = {}
    malignant_dict = {}
    for key in ['train', 'test']:
        if key == 'train':
            train_malignant_list = malignant_case_dict[key]
            train_benign_list = benign_case_dict[key]
            # print("train_malignant_list", train_malignant_list)
            for case in train_malignant_list:
                class_name = 'malignant'
                # print(case)

                image_path_list = []
                case_path = os.path.join(  # aug path
                    os.path.join(aug_path, class_name), case)
                # print(case_path)
                for image_name in os.listdir(case_path):
                    image_label = []
                    image_path = os.path.join(case_path, image_name)
                    image_label.append(image_path)  # image
                    image_label.append(1)  # label
                    image_path_list.append(image_label)

                ########################################
                # batch_aug
                if batch_aug_path is not None:
                    for aug_name in aug_list[1]:
                        # print("aug_name:", aug_name)
                        case_aug_folder = batch_aug_path + "/" + case + "_" + aug_name
                        # print(case_aug_folder)
                        if not os.path.exists(case_aug_folder):
                            print("Batch aug path not exist!")
                        for image_name in os.listdir(case_aug_folder):
                            # print(image_name)
                            # exit()
                            image_label = []
                            image_path = os.path.join(
                                case_aug_folder, image_name)
                            image_label.append(image_path)  # image
                            image_label.append(1)  # label
                            image_path_list.append(image_label)

                malignant_dict[case] = image_path_list

            for case in train_benign_list:
                class_name = 'benign'
                # print(case)

                image_path_list = []
                case_path = os.path.join(
                    os.path.join(aug_path, class_name), case)
                # print(case_path)
                for image_name in os.listdir(case_path):
                    image_label = []
                    image_path = os.path.join(case_path, image_name)
                    image_label.append(image_path)  # image
                    image_label.append(0)  # label
                    image_path_list.append(image_label)
                # print("image_path len:", len(image_path_list))
                # print("image path:", image_path_list[:2])
                ########################################
                # batch_aug
                if batch_aug_path is not None:
                    for aug_name in aug_list[0]:
                        # print("aug_name:", aug_name)
                        case_aug_folder = batch_aug_path + "/" + case + "_" + aug_name
                        # print(case_aug_folder)
                        if not os.path.exists(case_aug_folder):
                            print("Batch aug path not exist!")
                        for image_name in os.listdir(case_aug_folder):
                            # print(image_name)
                            # exit()
                            image_label = []
                            image_path = os.path.join(
                                case_aug_folder, image_name)
                            image_label.append(image_path)  # image
                            image_label.append(0)  # label
                            image_path_list.append(image_label)
                        # print("image_path len:", len(image_path_list))
                        # print("image path:", image_path_list[-2:])
                        # exit()
                benign_dict[case] = image_path_list

        # if key == 'valid':
        #     valid_malignant_list = malignant_case_dict[key]
        #     valid_benign_list = benign_case_dict[key]
        #     for case in valid_malignant_list:
        #         class_name = 'malignant'
        #         # print(case)

        #         image_path_list = []
        #         case_path = os.path.join(
        #             os.path.join(data_path, class_name), case)
        #         # print(case_path)
        #         for image_name in os.listdir(case_path):
        #             image_label = []
        #             image_path = os.path.join(case_path, image_name)
        #             image_label.append(image_path)  # image
        #             image_label.append(1)      # label
        #             image_path_list.append(image_label)
        #         malignant_dict[case] = image_path_list

        #     for case in valid_benign_list:
        #         class_name = 'benign'
        #         # print(case)

        #         image_path_list = []
        #         case_path = os.path.join(
        #             os.path.join(data_path, class_name), case)
        #         # print(case_path)
        #         for image_name in os.listdir(case_path):
        #             image_label = []
        #             image_path = os.path.join(case_path, image_name)
        #             image_label.append(image_path)  # image
        #             image_label.append(0)      # label
        #             image_path_list.append(image_label)
        #         # print(image_path_list)
        #         # exit()
        #         benign_dict[case] = image_path_list

        if key == "test":
            test_malignant_list = malignant_case_dict[key]
            test_benign_list = benign_case_dict[key]
            for case in test_malignant_list:
                class_name = 'malignant'
                # print(case)

                image_path_list = []
                case_path = os.path.join(
                    os.path.join(data_path, class_name), case)
                # print(case_path)
                for image_name in os.listdir(case_path):
                    image_label = []
                    image_path = os.path.join(case_path, image_name)
                    image_label.append(image_path)  # image
                    image_label.append(1)  # label
                    image_path_list.append(image_label)
                malignant_dict[case] = image_path_list

            for case in test_benign_list:
                class_name = 'benign'

                image_path_list = []
                case_path = os.path.join(
                    os.path.join(data_path, class_name), case)
                for image_name in os.listdir(case_path):
                    image_label = []
                    image_path = os.path.join(case_path, image_name)
                    image_label.append(image_path)  # image
                    image_label.append(0)  # label
                    image_path_list.append(image_label)

                benign_dict[case] = image_path_list

    return benign_dict, malignant_dict


def get_dataset_from_list_dict(select_case_list, select_percent, benign_dict, malignant_dict):
    # 单个fold所选择的 benign case 和malignant case

    # print(select_percent) # [0.36610878661087864, 0.07640253219821]
    # exit()
    # 第一个是benign case 第二个列表是 malignant case
    data_list = []
    for case in select_case_list[0]:
        sample = random.sample(benign_dict[case], int(
            np.round(select_percent[0] * len(benign_dict[case]))))
        for sam in sample:
            data_list.append(sam)

    for case in select_case_list[1]:
        sample = random.sample(malignant_dict[case], int(
            np.round(select_percent[1] * len(malignant_dict[case]))))
        for sam in sample:
            data_list.append(sam)

    return data_list


def hard_voting(predict_list):
    return_list = []
    # predict_list[0], predict_list[1], predict_list[2], predict_list[3],
    # predict_list[4], predict_list[5], predict_list[6], predict_list[7]
    for i in zip(predict_list[0], predict_list[1], predict_list[2]):
        vote_class1 = 0
        vote_class2 = 0
        for re in i:
            if re == 0:
                vote_class1 += 1
            else:
                vote_class2 += 1
        if vote_class1 > vote_class2:
            return_list.append(0)
        if vote_class1 < vote_class2:
            return_list.append(1)
        # if vote_class1 == vote_class2:
        #     # print("Same Voting!")
        #     return_list.append(1)
    # print("return_list:", return_list)
    # print("len of return list", len(return_list))
    # exit()
    return return_list


def process_hard_voting(true_list, predict_list):
    print("Hard Voting!")
    # print("The Number of Model is  Odd !")  # 奇数个模型
    voting_result = hard_voting(predict_list)
    # print("true_list[0]", true_list[0])
    # print("voting result:", voting_result)
    accuracy = accuracy_score(true_list[0], voting_result)
    f1score = f1_score(true_list[0], voting_result, average='binary')
    sensitivity = sensitivity_score(
        true_list[0], voting_result, average='binary')
    specificity = specificity_score(
        true_list[0], voting_result, average='binary')
    fpr, tpr, thresholds = roc_curve(true_list[0], voting_result)
    auc_value = auc(fpr, tpr)
    CM = confusion_matrix(true_list[0], voting_result)
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    PPV = round(TP / (TP + FP), 4)
    NPV = round(TN / (TN + FN), 4)
    # PPV = precision positive predictive value
    # NPV negative predictive value
    # print("PPV:{:.4f}, NPV:{:.4f}".format(PPV, NPV))
    # print("Acc, F1-score, AUC, PPV, NPV, Sen, Spec: {:.2f}  {:.2f}  {:.2f}  {:.2f}  {:.2f}  {:.2f}  {:.2f}".format(
    #     accuracy, f1score, auc(fpr, tpr), PPV, NPV, sensitivity, specificity))

    return accuracy, f1score, auc_value, PPV, NPV, sensitivity, specificity


def soft_voting(predict_probs):
    outputs = np.array(predict_probs)
    # exit()
    # avg = np.average(outputs, axis=0, weights=[0.25, 0.50, 0.25]) # MobileNet-V2
    # avg = np.average(outputs, axis=0, weights=[0.20, 0.40, 0.40]) # densenet121
    # avg = np.average(outputs, axis=0, weights=[0.15, 0.80, 0.05]) # resnet18
    avg = np.average(outputs, axis=0)

    return avg.argmax(axis=1), avg


def soft_voting_weight(predict_probs, weights):
    outputs = np.array(predict_probs)
    # exit()
    # avg = np.average(outputs, axis=0, weights=[0.25, 0.50, 0.25]) # MobileNet-V2
    avg = np.average(outputs, axis=0, weights=weights)  # densenet121
    # avg = np.average(outputs, axis=0, weights=[0.15, 0.80, 0.05]) # resnet18
    # avg = np.average(outputs, axis=0)

    return avg.argmax(axis=1), avg


def process_soft_voting(true_list, predict_prob_list):
    print("Soft Voting!")
    soft_voting_result, predict_prob_avg = soft_voting(
        predict_prob_list)
    accuracy = accuracy_score(true_list[0], soft_voting_result)
    f1score = f1_score(true_list[0], soft_voting_result, average='binary')
    sensitivity = sensitivity_score(
        true_list[0], soft_voting_result, average='binary')
    specificity = specificity_score(
        true_list[0], soft_voting_result, average='binary')
    fpr, tpr, thresholds = roc_curve(true_list[0], soft_voting_result)
    auc_value = auc(fpr, tpr)
    CM = confusion_matrix(true_list[0], soft_voting_result)
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    PPV = round(TP / (TP + FP), 4)
    NPV = round(TN / (TN + FN), 4)

    return accuracy, f1score, auc_value, PPV, NPV, sensitivity, specificity


def process_soft_voting_weighted(true_list, predict_prob_list):
    print("Weighted Soft Voting!")
    weights = [0.30, 0.35, 0.35]
    soft_voting_result, predict_prob_avg = soft_voting_weight(
        predict_prob_list, weights)
    accuracy = accuracy_score(true_list[0], soft_voting_result)
    f1score = f1_score(true_list[0], soft_voting_result, average='binary')
    sensitivity = sensitivity_score(
        true_list[0], soft_voting_result, average='binary')
    specificity = specificity_score(
        true_list[0], soft_voting_result, average='binary')
    fpr, tpr, thresholds = roc_curve(true_list[0], np.average(
        score_list, axis=0, weights=weights))
    auc_value = auc(fpr, tpr)
    CM = confusion_matrix(true_list[0], soft_voting_result)
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    PPV = round(TP / (TP + FP), 4)
    NPV = round(TN / (TN + FN), 4)

    return accuracy, f1score, auc_value, PPV, NPV, sensitivity, specificity


def draw_roc(fpr, tpr):
    # sns.set_style('darkgrid', {'axes.facecolor': '0.9'})

    print('AUC: {}'.format(auc(fpr, tpr)))
    plt.figure(figsize=(10, 10))
    lw = 3
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.yticks([i / 10.0 for i in range(11)])
    plt.xticks([i / 10.0 for i in range(11)])
    plt.text(0.7, 0.3, 'AUC=%.3f' % (auc(fpr, tpr)), size=22)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig("./evaluation_result/auc_curve.png")
    plt.close()


def parse_info():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.001,
                        type=float, help='learning rate')
    parser.add_argument('--epochs', default=50, type=int,
                        help='Epoch for Training')
    parser.add_argument("-kf", "--k_folders", type=int, default=5,
                        help="the number of k folder cross validcation")
    parser.add_argument("-bs", "--batch_size", type=int, default=32,
                        help="batch_size for training")
    parser.add_argument("-dp", "--dataset_path", type=str, default="D:/ebus_dataset_new_version/ebus_data_940/",
                        help="The path of dataset")
    parser.add_argument("-ap", "--aug_path", type=str, default="D:/ebus_dataset_new_version/ebus_data_940_Enhance_Aug/",
                        help="The path of dataset")
    # parser.add_argument("-tb", "--tensorboard_scale", type=str, default="mobilenet_v2",
    #                     help="batch_size for training")
    args = parser.parse_args()

    return args


def benign_malignant_images_check(image_list, benign_list, malignant_list):
    benign_image_nums = 0
    malignant_image_nums = 0

    for image in image_list:
        case_name = os.path.split(image[0])[1].split(".")[0][:3]
        if case_name in benign_list:
            benign_image_nums += 1
        if case_name in malignant_list:
            malignant_image_nums += 1
    # print("benign_image_nums:", benign_image_nums)
    # print("malignant_image_nums:", malignant_image_nums)
    # exit()
    return benign_image_nums, malignant_image_nums


def get_case_list(data_path):
    # 返回两个字典
    # 一个字典 是存储benign case dataset
    # 一个字典是存储 maligant case dataset
    classes_list = os.listdir(data_path)
    benign_list = []
    malignant_list = []

    for class_name in classes_list:
        case_list = os.listdir(os.path.join(data_path, class_name))
        # print(case_list)
        # exit()
        if class_name == "benign":
            benign_list = case_list

        if class_name == "malignant":
            malignant_list = case_list

    return benign_list, malignant_list


def setup_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_dataset_from_list_dict_v3(benign_list, malignant_list, benign_dict, malignant_dict, trainset_numbers):
    train_image_list = []
    train_benign_num = train_malignant_num = trainset_numbers // 2
    # print("train_benign", train_benign_num)
    # print("train_malignant", train_malignant_num)
    total_benign_list = []
    total_malignant_list = []
    for case in benign_list:
        total_benign_list.append(case)
    for case in malignant_list:
        total_malignant_list.append(case)

    avg_benign = train_benign_num // len(total_benign_list)
    avg_malignant = train_malignant_num // len(total_malignant_list)
    # print("avg_benign", avg_benign)
    # print("avg_malignant", avg_malignant)
    benign_count_dict = {}
    malignant_count_dict = {}
    # beingn case
    benign_not_enough_list = []
    benign_count = 0
    for case in total_benign_list:
        if avg_benign >= len(benign_dict[case]):
            # print("Not enough case:",case)
            # need get all images
            benign_not_enough_list.append(case)
            for image in benign_dict[case]:
                train_image_list.append(image)
            benign_count += len(benign_dict[case])
            benign_count_dict[case] = len(benign_dict[case])
    # recalculate the ho many benign are needed
    benign_reget = train_benign_num - benign_count
    # print("reget benign:", benign_reget)
    if len(total_benign_list) - len(benign_not_enough_list) != 0:
        avg_reget = math.ceil(
            benign_reget / (len(total_benign_list) - len(benign_not_enough_list)))
        # print("reget benign image:", benign_reget)
        # print("avg reget image:", avg_reget)
        for case in benign_not_enough_list:
            total_benign_list.remove(case)

        for case in total_benign_list:
            # print("Enough case", case)
            sample = random.sample(benign_dict[case], avg_reget)
            benign_count_dict[case] = len(sample)
            for sam in sample:
                train_image_list.append(sam)

    ############################################################
    # for malignant case:
    malignant_not_enough_list = []
    malignant_count = 0
    for case in total_malignant_list:
        if avg_malignant >= len(malignant_dict[case]):
            print("Not enough case:", case)
            # need get all images
            # exit()
            malignant_not_enough_list.append(case)
            # train_image_list = np.append(
            #     train_image_list, malignant_dict[case])
            for image in malignant_dict[case]:
                train_image_list.append(image)
            malignant_count += len(malignant_dict[case])
            malignant_count_dict[case] = len(malignant_dict[case])
    # exit()
    # recalculate the ho many benign are needed
    malignant_reget = train_malignant_num - malignant_count
    if len(total_malignant_list) - len(malignant_not_enough_list) != 0:

        malignant_avg_reget = math.ceil(malignant_reget / (
                len(total_malignant_list) - len(malignant_not_enough_list)))
        # print("reget malignant image:", malignant_reget)
        # print("avg malignant image:", avg_malignant)
        for case in malignant_not_enough_list:
            total_malignant_list.remove(case)

        for case in total_malignant_list:
            # print("Enough case", case)
            sample = random.sample(malignant_dict[case], malignant_avg_reget)
            malignant_count_dict[case] = len(sample)
            for sam in sample:
                train_image_list.append(sam)

    # remove image
    remove_list = []
    remove_list = random.sample(
        train_image_list, (len(train_image_list) - trainset_numbers))
    for remo in remove_list:
        train_image_list.remove(remo)

    # print("Now train images:", len(train_image_list))
    # exit()
    return train_image_list


def get_dataset_from_list_dict_v4(benign_list, malignant_list, benign_dict, malignant_dict, trainset_numbers,
                                  benign_ration=None):
    train_image_list = []
    if benign_ration is None:
        train_benign_num = train_malignant_num = trainset_numbers * 0.5
    if benign_ration is not None:
        train_malignant_num = trainset_numbers * 0.5
        train_benign_num = benign_ration * trainset_numbers * 0.5
    # print("train_benign", train_benign_num)
    # print("train_malignant", train_malignant_num)
    total_benign_list = []
    total_malignant_list = []
    train_benign_list = []
    train_malignant_list = []
    for case in benign_list:
        total_benign_list.append(case)
    for case in malignant_list:
        total_malignant_list.append(case)

    avg_benign = round(train_benign_num / len(total_benign_list)) + 4
    avg_malignant = round(train_malignant_num / len(total_malignant_list)) + 1
    # print("avg_benign", avg_benign)
    # print("avg_malignant", avg_malignant)
    # exit()
    benign_count_dict = {}
    malignant_count_dict = {}
    # beingn case
    benign_not_enough_list = []
    benign_count = 0
    # print(len(train_image_list))
    for case in total_benign_list:
        if len(benign_dict[case]) < avg_benign:
            # print("Not enough case:",case)

            # need get all images
            benign_not_enough_list.append(case)
            for image in benign_dict[case]:
                train_benign_list.append(image)
            benign_count += len(benign_dict[case])
            benign_count_dict[case] = len(benign_dict[case])

        else:
            sample = random.sample(benign_dict[case], avg_benign)
            benign_count_dict[case] = len(sample)
            for sam in sample:
                train_benign_list.append(sam)
    # exit()
    # 删除多余的
    if len(train_benign_list) > train_benign_num:
        del_sample = random.sample(train_benign_list, int(
            len(train_benign_list) - train_benign_num))
        for del_sam in del_sample:
            train_benign_list.remove(del_sam)
    # print(len(train_benign_list))
    train_image_list.extend(train_benign_list)

    # exit()
    ############################################################
    # for malignant case:
    malignant_not_enough_list = []
    malignant_count = 0
    for case in total_malignant_list:
        if avg_malignant > len(malignant_dict[case]):
            # print("Not enough case:", case)
            # need get all images
            # exit()
            malignant_not_enough_list.append(case)
            # train_image_list = np.append(
            #     train_image_list, malignant_dict[case])
            for image in malignant_dict[case]:
                train_malignant_list.append(image)
            malignant_count += len(malignant_dict[case])
            malignant_count_dict[case] = len(malignant_dict[case])

        else:
            sample = random.sample(malignant_dict[case], avg_malignant)
            malignant_count_dict[case] = len(sample)
            for sam in sample:
                train_malignant_list.append(sam)
    # 删除多余的

    # exit()
    if len(train_malignant_list) > train_malignant_num:
        del_sample = random.sample(train_malignant_list, int(
            len(train_malignant_list) - train_malignant_num))
        for del_sam in del_sample:
            train_malignant_list.remove(del_sam)
    # print(len(train_malignant_list))
    train_image_list.extend(train_malignant_list)

    return train_image_list


def get_case_list(data_path):
    # 返回两个字典
    # 一个字典 是存储benign case dataset
    # 一个字典是存储 maligant case dataset
    classes_list = os.listdir(data_path)
    benign_list = []
    malignant_list = []

    for class_name in classes_list:
        case_list = os.listdir(os.path.join(data_path, class_name))
        # print(case_list)
        # exit()
        if class_name == "benign":
            benign_list = case_list

        if class_name == "malignant":
            malignant_list = case_list

    return benign_list, malignant_list


def gen_model(model_name):
    if model_name == "densenet121_fc":
        net = models.densenet121(
            weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        for parameter in net.parameters():
            parameter.required_grad = False

        net.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Dropout(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128),
            nn.Dropout(0.2),
            nn.BatchNorm1d(128),
            nn.Linear(128, 2)
        )
    if model_name == "mobilenet_v2_fc":
        net = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

        for parameter in net.parameters():
            parameter.required_grad = False

        net.classifier[1] = nn.Sequential(
            nn.Linear(1280, 256),
            nn.Dropout(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.Dropout(0.2),
            nn.BatchNorm1d(128),
            nn.Linear(128, 2)
        )
    if model_name == "densenet169_fc":
        net = models.densenet169(
            weights=models.DenseNet169_Weights.IMAGENET1K_V1)
        for parameter in net.parameters():
            parameter.required_grad = False

        net.classifier = nn.Sequential(
            nn.Linear(1664, 512),
            nn.Dropout(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128),
            nn.Dropout(0.2),
            nn.BatchNorm1d(128),
            nn.Linear(128, 2)
        )
    if model_name == "resnet18_fc":
        net = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1)
        for parameter in net.parameters():
            parameter.required_grad = False

        net.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.Dropout(0.2),
            nn.BatchNorm1d(128),
            nn.Linear(128, 2)
        )
    if model_name == "resnet34_fc":
        net = models.resnet34(
            weights=models.ResNet34_Weights.IMAGENET1K_V1)

        for parameter in net.parameters():
            parameter.required_grad = False

        net.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.Dropout(0.2),
            nn.BatchNorm1d(128),
            nn.Linear(128, 2)
        )
    if model_name == "resnet50_fc":
        net = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1)

        for parameter in net.parameters():
            parameter.required_grad = False
        net.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.Dropout(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128),
            nn.Dropout(0.2),
            nn.BatchNorm1d(128),
            nn.Linear(128, 2)
        )
    if model_name == "efficientnet_b0_fc":
        net = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

        for parameter in net.parameters():
            parameter.required_grad = False

        net.classifier[1] = nn.Sequential(
            nn.Linear(1280, 256),
            nn.Dropout(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.Dropout(0.2),
            nn.BatchNorm1d(128),
            nn.Linear(128, 2)
        )
    if model_name == "shuffenet_v2_fc":
        net = models.shufflenet_v2_x0_5(
            weights=models.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1)
        for parameter in net.parameters():
            parameter.required_grad = False
        net.fc = nn.Sequential(
            nn.Linear(1024, 256),
            nn.Dropout(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.Dropout(0.2),
            nn.BatchNorm1d(128),
            nn.Linear(128, 2)
        )

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    return net


def concate_result(all_predict_prob):
    con_result = np.concatenate(
        (all_predict_prob[0], all_predict_prob[1]), 1)

    return con_result


def process_stacking(train_concate, valid_labels_list, test_concate, true_list):
    scaler = StandardScaler()
    scaler.fit(train_concate)
    train_concate = scaler.transform(train_concate)

    clf = svm.SVC(kernel='rbf', probability=True,
                  max_iter=300, C=1000, gamma=0.001)

    clf.fit(train_concate, valid_labels_list[0])

    print('------Testing SVM------')
    # scaler2 = StandardScaler()
    # scaler2.fit(test_concate)
    test_concate = scaler.transform(test_concate)
    test_result = clf.predict(test_concate)

    accuracy = accuracy_score(true_list[0], test_result)
    f1score = f1_score(true_list[0], test_result, average='binary')
    sensitivity = sensitivity_score(
        true_list[0], test_result, average='binary')
    specificity = specificity_score(
        true_list[0], test_result, average='binary')
    fpr, tpr, thresholds = roc_curve(true_list[0], test_result)
    auc_value = auc(fpr, tpr)
    CM = confusion_matrix(true_list[0], test_result)
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    PPV = round(TP / (TP + FP), 4)
    NPV = round(TN / (TN + FN), 4)
    # print("Acc, F1-score, AUC, PPV, NPV, Sen, Spec: {:.2f}  {:.2f}  {:.2f}  {:.2f}  {:.2f}  {:.2f}  {:.2f}".format(
    #     accuracy, f1score, auc_value, PPV, NPV, sensitivity, specificity))
    # exit()
    return accuracy, f1score, auc_value, PPV, NPV, sensitivity, specificity


if __name__ == "__main__":

    setup_seed(222)

    # patience = 5
    args = parse_info()
    # patient_nums = 60.0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start_epoch = 1  # start from epoch 0 or last checkpoint epoch
    # early_stopping = EarlyStopping(patience=10, verbose=True)
    transform_train = transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomVerticalFlip(p=0.5),
        # transforms.RandomRotation(90),
        transforms.CenterCrop(680),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.116088, 0.115966, 0.116208),
                             (0.127014, 0.126976, 0.127053)),
    ])

    transform_test = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.CenterCrop(680),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.116088, 0.115966, 0.116208),
                             (0.127014, 0.126976, 0.127053)),
    ])
    # Data
    print('==> Preparing data..')

    trainset_numbers = 400
    validset_numbers = 100
    testset_numbers = 100

    benign_list, malignant_list = get_case_list(
        args.dataset_path)

    print('==> Building model..')

    acc_list = []
    f1_score_list = []
    auc_list = []
    # h_acc, h_f1_score_value, h_auc_value, h_ppv, h_npv, h_sen, h_spec
    h_acc_list, h_f1_score_value_list, h_auc_value_list, h_ppv_list, h_npv_list, h_sen_list, h_spec_list = [
    ], [], [], [], [], [], []
    s_acc_list, s_f1_score_value_list, s_auc_value_list, s_ppv_list, s_npv_list, s_sen_list, s_spec_list = [
    ], [], [], [], [], [], []
    w_acc_list, w_f1_score_value_list, w_auc_value_list, w_ppv_list, w_npv_list, w_sen_list, w_spec_list = [
    ], [], [], [], [], [], []
    for fold in range(1, 6):

        folder_name = "fold" + str(fold)
        print("fold name:", folder_name)
        if folder_name == "fold1":
            malignant_case_dict = {
                'train': ['073', '032', '047', '063', '037', '014', '001', '045', '049', '057', '046', '030', '010',
                          '048', '019', '018', '004', '026', '056', '053', '034', '050', '033', '022', '044', '054',
                          '043', '069', '036',
                          '012', '002', '064', '006', '071', '039', '059', '060'],
                'valid': ['067', '035', '027', '068', '024', '021', '052', '070', '031', '072', '025'],
                'test': ['055', '005', '016', '065', '028', '066', '020', '042', '051', '029', '040']}
            benign_case_dict = {'train': ['058', '061', '008', '017', '011', '007'], 'valid': [
                '023', '003'], 'test': ['041', '062']}

        if folder_name == "fold2":
            malignant_case_dict = {
                'train': ['073', '032', '047', '063', '037', '014', '001', '045', '049', '057', '046', '055', '005',
                          '016', '065', '028', '066', '020', '042', '051', '029', '040', '067', '035', '027', '068',
                          '024', '021', '052',
                          '070', '031', '072', '025', '071', '039', '059', '060'],
                'valid': ['033', '022', '044', '054', '043', '069', '036', '012', '002', '064', '006'],
                'test': ['030', '010', '048', '019', '018', '004', '026', '056', '053', '034', '050']}
            benign_case_dict = {'train': ['041', '062', '023', '003', '011', '007'], 'valid': [
                '008', '017'], 'test': ['058', '061']}

        if folder_name == "fold3":
            malignant_case_dict = {
                'train': ['055', '005', '016', '065', '028', '066', '020', '042', '051', '029', '040', '030', '010',
                          '048', '019', '018', '004', '026', '056', '053', '034', '050', '033', '022', '044', '054',
                          '043', '069', '036',
                          '012', '002', '064', '006', '071', '039', '059', '060'],
                'valid': ['073', '032', '047', '063', '037', '014', '001', '045', '049', '057', '046'],
                'test': ['067', '035', '027', '068', '024', '021', '052', '070', '031', '072', '025']}
            benign_case_dict = {'train': ['041', '062', '058', '061', '008', '017'], 'valid': [
                '011', '007'], 'test': ['023', '003']}

        if folder_name == "fold4":
            malignant_case_dict = {
                'train': ['073', '032', '047', '063', '037', '014', '001', '045', '049', '057', '046', '030', '010',
                          '048', '019', '018', '004', '026', '056', '053', '034', '050', '067', '035', '027', '068',
                          '024', '021', '052',
                          '070', '031', '072', '025', '071', '039', '059', '060'],
                'valid': ['055', '005', '016', '065', '028', '066', '020', '042', '051', '029', '040'],
                'test': ['033', '022', '044', '054', '043', '069', '036', '012', '002', '064', '006']}
            benign_case_dict = {'train': ['058', '061', '023', '003', '011', '007'], 'valid': [
                '041', '062'], 'test': ['008', '017']}
        if folder_name == "fold5":
            malignant_case_dict = {
                'train': ['055', '005', '016', '065', '028', '066', '020', '042', '051', '029', '040', '067', '035',
                          '027', '068', '024', '021', '052', '070', '031', '072', '025', '033', '022', '044', '054',
                          '043', '069', '036',
                          '012', '002', '064', '006', '071', '039', '059', '060'],
                'valid': ['030', '010', '048', '019', '018', '004', '026', '056', '053', '034', '050'],
                'test': ['073', '032', '047', '063', '037', '014', '001', '045', '049', '057', '046']}
            benign_case_dict = {'train': ['041', '062', '023', '003', '008', '017'], 'valid': [
                '058', '061'], 'test': ['011', '007']}

        aug_list = [["brightness_spatial", "noise_spatial"],
                    ["brightness_spatial", "noise_spatial"]]
        benign_dict, malignant_dict = get_all_case_data(
            args.dataset_path, args.aug_path, None, aug_list, malignant_case_dict, benign_case_dict)
        train_image_list = get_dataset_from_list_dict_v4(
            benign_case_dict["train"], malignant_case_dict["train"], benign_dict, malignant_dict, trainset_numbers,
            None)
        test_image_list = get_dataset_from_list_dict_v4(
            benign_case_dict["test"], malignant_case_dict["test"], benign_dict, malignant_dict, testset_numbers, None)

        # print("train set:", len(train_image_list))
        # print("test set:", len(test_image_list))
        # # exit()

        train_benign_nums, train_malignant_nums = benign_malignant_images_check(
            train_image_list, benign_list, malignant_list)
        test_benign_nums, test_malignant_nums = benign_malignant_images_check(
            test_image_list, benign_list, malignant_list)

        print("Train set: benign images:{}, malignant images:{}".format(
            train_benign_nums, train_malignant_nums))
        print("Test set: benign images:{}, malignant images:{}".format(
            test_benign_nums, test_malignant_nums))

        train_set = Custom_DataSet(
            train_image_list, transforms=transform_train)
        trainloader = torch.utils.data.DataLoader(train_set,
                                                  batch_size=args.batch_size,
                                                  shuffle=True)
        test_set = Custom_DataSet(test_image_list, transforms=transform_test)
        testloader = torch.utils.data.DataLoader(test_set,
                                                 batch_size=args.batch_size,
                                                 shuffle=False)

        model_names = ["densenet169_fc", "densenet169_fc", "densenet169_fc"]
        net_list = []
        for model_name in model_names:
            print("Model Name:{}".format(model_name))
            return_net = gen_model(model_name)
            net_list.append(return_net)

        model_nums = 3
        true_list = []
        valid_labels_list = []
        predict_list = []
        score_list = []
        all_predict_prob = []
        valid_predict_prob_list = []

        # 相同模型融合
        for idx in [0, 2]:
            print("idx:", idx)
            model_net = net_list[idx]
            checkpoint = torch.load(
                '2023_03_01/{}/{}/checkpoint/{}_940_ES_fold{}.pth'.format(folder_name, model_name, model_name, idx))
            model_net.load_state_dict(checkpoint['net'])

            true_label_list, predicted_list, predict_prob_list = test(
                model_net)
            valid_label_list, _, valid_predict_prob = model_evaluate(model_net)
            all_predict_prob.append(predict_prob_list)
            true_list.append(true_label_list)

            valid_labels_list.append(valid_label_list)
            valid_predict_prob_list.append(valid_predict_prob)

        # # 不同模型融合
        # for idx, (model_net, model_name) in enumerate(zip(net_list, model_names)):
        #     print("idx:", idx)
        #     # if idx ==1:
        #     #     idx +=1
        #     checkpoint = torch.load(
        #         '2023_03_01/{}/{}/checkpoint/{}_940_ES_fold{}.pth'.format(folder_name, model_name, model_name, idx))
        #     model_net.load_state_dict(checkpoint['net'])

        #     true_label_list, predicted_list, predict_prob_list = test(
        #         model_net)
        #     valid_label_list, _, valid_predict_prob = model_evaluate(model_net)
        #     all_predict_prob.append(predict_prob_list)
        #     true_list.append(true_label_list)

        #     valid_labels_list.append(valid_label_list)
        #     valid_predict_prob_list.append(valid_predict_prob)

        test_concate = concate_result(all_predict_prob)
        train_concate = concate_result(valid_predict_prob_list)

        # print("train feature:", len(train_concate))
        # print("train labels:", len(valid_labels_list))
        # print("test feature:", len(test_concate))
        # print("test labels:", len(true_list))
        # exit()
        h_acc, h_f1_score_value, h_auc_value, h_ppv, h_npv, h_sen, h_spec = process_stacking(
            train_concate, valid_labels_list, test_concate, true_list)

        h_acc_list.append(h_acc)
        h_f1_score_value_list.append(h_f1_score_value)
        h_auc_value_list.append(h_auc_value)
        h_ppv_list.append(h_ppv)
        h_npv_list.append(h_npv)
        h_sen_list.append(h_sen)
        h_spec_list.append(h_spec)

    print(
        "Stacking: Acc, F1-score, AUC, PPV, NPV, Sen, Spec: {:.2f}  {:.2f}  {:.2f}  {:.2f}  {:.2f}  {:.2f}  {:.2f}".format(
            np.mean(h_acc_list), np.mean(
                h_f1_score_value_list), np.mean(h_auc_value_list),
            np.mean(h_ppv_list), np.mean(h_npv_list), np.mean(h_sen_list), np.mean(h_spec_list)))