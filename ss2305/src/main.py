from train import train
from test import do_test

def main():
    data_dir = '../data/VIUS'
    output_path = '../data/result/2'
    lr = 0.0001
    batch_size = 16
    max_epoch = 50
    results = train(data_dir, output_path, learning_rate=lr, batch_size=batch_size, max_epoch=max_epoch)
    print(f"AUC value of validation data: {results:.3f}")

    test_path = '../data/IVUS/test'
    model_path = '../data/result/2/model_best_20231115123720.pth'
    output_path = '../data/result/2'

    do_test(test_data_path=test_path, model_path=model_path, output_path=output_path, model_name='VGG16', gpu_id=0,
            time_stamp='')
