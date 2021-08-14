'''
sudo python3  -m torch.distributed.launch Evaluation_lstm_PNN.py

'''
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
from tqdm import tqdm
import PNN_TestFileDatabase
from VGGFace_conv6_FeatureExtraction import VGG_Face
from LSTM_conv6_PNN import FineTuneLstmModel
import random
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, recall_score, accuracy_score, f1_score, auc,roc_curve
from sklearn.preprocessing import label_binarize

from torch.nn import functional as F

parser = argparse.ArgumentParser(description='Training')  # description: 描述这个参数解析器是用于干什么的
# 当'-'和'--'同时出现的时候，系统默认后者为参数名，前者不是，但是在命令行输入的时候没有这个区分
parser.add_argument('--local_rank', type=int, )

# test_batch_size
parser.add_argument('-tb', '--test_batch_size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 1)')
# workers
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
# epochs
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
# start_epoch
parser.add_argument('--start_epochs', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
args = parser.parse_known_args()[0]

classes = ('中性', '高兴', '疑惑', '紧张',
           '轻松', '尴尬')

best_prec1 = 0


def set_seed(seed=1):  # seed的数值可以随意设置，本人不清楚有没有推荐数值
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN在使用deterministic模式时（下面两行），可能会造成性能下降（取决于model）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_GPUS(model, model_path):
    state_dict = torch.load(model_path)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    return model


def main():
    global args, start_loss, distance, best_prec1
    set_seed()
    args = parser.parse_args()

    # Load Data
    test_dataset = PNN_TestFileDatabase.FlameSet()  # for test  0.3
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.test_batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=False)

    # Multiple GPU training , load device
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 想使用的GPU编号
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # GPU开始编号，依次往下
    print("load into device:", device)

    # init visual feature extractor
    featureExtractor = VGG_Face()
    featureExtractor = load_GPUS(featureExtractor, '/mnt/nfs-shared/xinda/Werewolf-XL/werewolf_video/vgg16_face_30.pth')
    featureExtractor = nn.DataParallel(featureExtractor)
    featureExtractor.to(device)
    print('load visual feature extractor success')

    # load LSTM model, 多GPU计算
    model = FineTuneLstmModel()
    # model_CKPT = torch.load('/home/xinda/werewolf_lstm/werewolf_video/PNN_saved_model/PNN_LSTM_50.pth')
    model_CKPT = torch.load('/mnt/nfs-shared/xinda/Werewolf-XL/werewolf-XL_202103/Classification/checkpoints/PNN_LSTM_2layer_params_25_69.9546.pth')
    model.load_state_dict(model_CKPT)
    model.to(device)
    model.eval()

    with torch.no_grad():
        groundTruth = []
        prediction_max = []
        prediction_prob =[]
        test_correct = 0
        test_total = 0
        for i, videoFrames in enumerate(tqdm(test_loader)):
            label = videoFrames['label'].to(device)
            videoFrames = torch.squeeze(videoFrames['videoFrames']).to(device)
            length = videoFrames.shape[0]
            Outputs = []

            if length < 16:
                lack = 16 - length
                repeat_frames = videoFrames[-1:, ...].repeat(lack, 1, 1, 1)
                videoFrames = torch.cat((videoFrames, repeat_frames), 0)

            circle = int(length / 8) - 1
            for k in range(circle):
                start = 0 + 8 * k
                end = 16 + 8 * k
                features = featureExtractor(videoFrames[start:end, ...].float())
                output, hidden = model(features.unsqueeze(0))
                output_mean = torch.mean(output, dim=0)  # one serie of frames = 16
                Outputs.append(output_mean.data.cpu().numpy().tolist())  # All series of frames

            Outputs = torch.Tensor(Outputs)

            if Outputs.shape[0]>1:
                outputs_average = torch.mean(Outputs, dim=0).unsqueeze(0)  # average of All series' output

            groundTruth.append(label.item())
            _, predicted = torch.max(outputs_average.data, 1)
            prediction_max.append(predicted.item())
            prediction_prob_b = F.softmax(outputs_average.data)
            prediction_prob.append(prediction_prob_b.data.numpy().reshape(6).tolist())

            test_total += label.size(0)
            test_correct += (predicted == label.data.cpu()).sum().item()


            if i%500 ==0:
                print("ground truth.length =", len(groundTruth))
                print("prediction.length =",len(prediction_max))

                accuracy = accuracy_score(prediction_max, groundTruth)
                print("accuracy = ", accuracy)
                f1 = f1_score(prediction_max, groundTruth, average="weighted")
                print("f1", f1)
                label = label_binarize(groundTruth, classes=list(range(6)))
                print("AUC = ", roc_auc_score(label, prediction_prob, average='micro'))

                test_accuracy = 100 * test_correct / test_total

                print('Accuracy of the network on the Test images: %d' % (test_accuracy))

        accuracy = accuracy_score(prediction_max, groundTruth)
        print("accuracy = ", accuracy)
        f1 = f1_score(prediction_max, groundTruth, average="weighted")
        print("prediction_max", f1)
        label = label_binarize(groundTruth, classes=list(range(6)))
        print("AUC = ", roc_auc_score(label, prediction_prob, average='micro'))

        test_accuracy = 100 * test_correct / test_total

        print('Accuracy of the network on the Test images: %d' % (test_accuracy))
        df = pd.DataFrame(data={"pnn_prediction": prediction_max, "pnn_groundtruth": groundTruth})
        df.to_csv("eval_pnn.csv")

        pro = np.array(prediction_prob)
        df2 = pd.DataFrame(pro)
        df2.to_csv("/mnt/nfs-shared/xinda/Werewolf-XL/werewolf-XL_202103/Classification/prediction_result/categorical_lstm_6pnn_202103.csv")



if __name__ == '__main__':
    print("start")
    main()