import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os


def plot_loss(log_file, output_path, num):
    epochs, training_losses, val_losses = [], [], []

    with open(log_file, 'r') as file:
        for line in file:
            epoch, training_loss, val_loss = line.strip().split(',')
            epochs.append(int(epoch))
            training_losses.append(float(training_loss))
            val_losses.append(float(val_loss))

    # Plot loss value
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, training_losses, label='training loss')
    plt.plot(epochs, val_losses, label='validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and validation Loss')
    plt.legend()
    save_path = os.path.join(output_path, f'training_validation_loss_{num}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

def plot_and_save_metrics(metrics_list, metric_names, plot_title, save_path):
    """
    複数のメトリクスをグラフにプロットし、画像として保存する関数。

    :param metrics_list: 各メトリクスの値のリストのリスト（例: [test_acc_list, sensitivity_list, specificity_list]）
    :param metric_names: 各メトリクスの名前のリスト（例: ["Accuracy", "Sensitivity", "Specificity"]）
    :param plot_title: グラフのタイトル
    :param save_path: 保存するファイルパス
    """
    plt.figure(figsize=(10, 6))
    for metrics, name in zip(metrics_list, metric_names):
        plt.plot(metrics, label=name)

    plt.title(plot_title)
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_learning_curves(training_losses, validation_losses, training_accuracies, validation_accuracies, save_path):
    epochs = range(1, len(training_losses) + 1)
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, training_losses, 'bo-', label='Training loss', linewidth=1, markersize=3)
    plt.plot(epochs, validation_losses, 'ro-', label='Validation loss', linewidth=1, markersize=3, color='orange')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, training_accuracies, 'bo-', label='Training accuracy', linewidth=1, markersize=3)
    plt.plot(epochs, validation_accuracies, 'ro-', label='Validation accuracy', linewidth=1, markersize=3, color='orange')
    plt.title('Training and Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_accuracy_over_epochs(training_accuracies, validation_accuracies, save_path):
    """
    Plot the training and validation accuracy over each epoch.

    :param training_accuracies: List of training accuracies for each epoch.
    :param validation_accuracies: List of validation accuracies for each epoch.
    :param save_path: Path where the plot image will be saved.
    """
    epochs = range(1, len(training_accuracies) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, training_accuracies, 'b-o', label='Training Accuracy', linewidth=1, markersize=3)
    plt.plot(epochs, validation_accuracies, 'r-o', label='Validation Accuracy', linewidth=1, markersize=3)
    plt.title('Training and Validation Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def plot_and_save_auc_curve(y_true, y_scores, save_path):
    """
    This function plots the ROC Curve and saves it to a file.

    Parameters:
    - y_true: list, actual binary labels for each sample (0 or 1).
    - y_scores: list, target scores, probabilities of the positive class.
    - save_path: str, path to save the ROC Curve plot.

    Returns:
    - auc_score: float, calculated area under the ROC curve.
    """
    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # Save the figure
    plt.savefig(save_path)
    plt.close()

    return roc_auc


def main():
    test_acc_list = [0.58, 0.86, 0.67, 0.44]
    sensitivity_list = [0.68, 0.86, 0.97, 0.74]
    specificity_list = [0.34, 0.45, 0.67, 0.89]
    output_path = '../../data/test'
    # Save test data
    metrics_list = [test_acc_list, sensitivity_list, specificity_list]
    metric_names = ["Accuracy", "Sensitivity", "Specificity"]
    plot_title = "Test Metrics Over Epochs"
    save_path = f"{output_path}/test_metrics.png"

    # グラフをプロットして保存
    plot_and_save_metrics(metrics_list, metric_names, plot_title, save_path)

if __name__ == '__main__':
    main()
