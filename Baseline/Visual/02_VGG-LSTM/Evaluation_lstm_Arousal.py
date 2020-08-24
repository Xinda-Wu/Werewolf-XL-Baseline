"""
吴鑫达模型优化：
1。 lossMIN调整
2。 tensorboard一起优化
sudo python3  -m torch.distributed.launch Evaluation_lstm_Arousal.py

1. 特征提取 从 layer-6上面： 4096
2. 查看loss 为什么都是 0

"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import numpy as np
from tqdm import tqdm
import Arousal_FileDatabase
from VGGFace_conv6 import VGG_Face
from LSTM_conv6 import FineTuneLstmModel
import random
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt


parser = argparse.ArgumentParser(description='Training')  # description: 描述这个参数解析器是用于干什么的
# 当'-'和'--'同时出现的时候，系统默认后者为参数名，前者不是，但是在命令行输入的时候没有这个区分
parser.add_argument('--local_rank', type=int, )

# test_batch_size
parser.add_argument('-tb', '--test_batch_size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 1)')
# workers
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')

args = parser.parse_known_args()[0]


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


def load_checkpoint(model, path, optimizer):
    if os.listdir(path) != None:
        model_CKPT = torch.load(path)
        model.load_state_dict(model_CKPT['state_dict'])
        print('loaded checkpoint')
        optimizer.load_state_dict(model_CKPT['optimizer'])
    return model, optimizer

def RMSE_evl(groundTruth,prediction):
    return sqrt(mean_squared_error(groundTruth, prediction))

def Spearman_eval(groundTruth,prediction):
    data = {'result':prediction, 'y_test':groundTruth}
    print(data)
    df = pd.DataFrame(data, columns=['result','y_test'])
    spearman = df.corr(method="spearman" )
    return  spearman

def CCC_eval(groundTruth,prediction):
    def getPvar(vals, mean):
        N = len(vals)
        su = 0
        for i in range(len(vals)):
            su = su + ((vals[i] - mean) * (vals[i] - mean))
        pvar = (1 / N) * su
        return pvar

    def getMean(vals):
        su = 0
        for i in range(len(vals)):
            su = su + vals[i]
        mean = su / (len(vals))
        return mean

    def getMeanofDiffs(xvals, yvals):
        su = 0
        for i in range(len(xvals)):
            su = su + ((xvals[i] - yvals[i]) * (xvals[i] - yvals[i]))
        meanodiffs = su / (len(xvals))
        return meanodiffs

    def getCCC(pvarfe, pvarexp, meanofdiff, meanfe, meanexp):
        bottom = pvarfe + pvarexp + ((meanfe - meanexp) * (meanfe - meanexp))
        answer = 1 - (meanofdiff / bottom)
        return answer

    prediction = prediction
    ground = groundTruth
    meanfe = getMean(ground)
    meanexp = getMean(prediction)
    meanofdiff = getMeanofDiffs(ground, prediction)
    pvarfe = getPvar(ground, meanfe)
    pvarexp = getPvar(prediction, meanexp)
    ccc = getCCC(pvarfe, pvarexp, meanofdiff, meanfe, meanexp)

    return ccc

def Evaluation(test_loader, featureExtractor, model, device):
    print(" start to test! ")
    # model.eval()  # switch to train
    with torch.no_grad():
        groundTruth = []
        prediction = []
        for i, videoFrames in enumerate(tqdm(test_loader)):
            label = videoFrames['label'].numpy()
            print(label)
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
                output_mean = torch.mean(output).unsqueeze(0)  # one serie of frames = 16
                Outputs.append(output_mean)  # All series of frames

            Outputs = torch.Tensor(Outputs)
            outputs_average = torch.mean(Outputs)  # average of All series' output
            outputs_average = outputs_average.numpy()
            if np.isnan(outputs_average):
                continue
            else:
                groundTruth.append(label.item())
                prediction.append(outputs_average.item())

            if i ==10:
                print("GroundTruth =", groundTruth)
                print("Prediction = ",prediction)
                print()
                rmse = RMSE_evl(groundTruth,prediction)
                spearman = Spearman_eval(groundTruth,prediction)
                ccc = CCC_eval(groundTruth,prediction)
                print("RMSE = ",rmse)
                print("Spearman = ", spearman)
                print("CCC = ", ccc)
                break




def main():
    global args, start_loss, distance
    set_seed()
    args = parser.parse_args()

    # Load Data
    test_dataset = Arousal_FileDatabase.FlameSet()
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.test_batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=False)

    # Multiple GPU training , load device
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'  # 想使用的GPU编号
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # GPU开始编号，依次往下
    print("load into device:", device)

    # init visual feature extractor
    featureExtractor = VGG_Face()
    featureExtractor = load_GPUS(featureExtractor, './vgg16_face_30.pth')
    featureExtractor = nn.DataParallel(featureExtractor)
    featureExtractor.to(device)
    print('load visual feature extractor success')

    # load LSTM model, 多GPU计算
    model = FineTuneLstmModel()
    model_CKPT = torch.load('/home/dell/Xinda/werewolf_video/saved_model_Arousal/lstm_loss_165.3575.pth.tar')
    model_state = model_CKPT['state_dict']
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    with torch.no_grad():
        groundTruth = []
        prediction = []
        Nan_list = []
        for i, videoFrames in enumerate(tqdm(test_loader)):
            label = videoFrames['label'].numpy()
            # print(label)
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
                output_mean = torch.mean(output).unsqueeze(0)  # one serie of frames = 16
                Outputs.append(output_mean)  # All series of frames

            Outputs = torch.Tensor(Outputs)
            outputs_average = torch.mean(Outputs)  # average of All series' output
            outputs_average = outputs_average.numpy()
            if np.isnan(outputs_average):
                Nan_list.append({str(i): label})
                continue
            else:
                groundTruth.append(label.item())
                prediction.append(outputs_average.item())

            # if i%500 ==0:
            #     print("GroundTruth =", groundTruth)
            #     print("Prediction = ",prediction)
            #     print()
            #     rmse = RMSE_evl(groundTruth,prediction)
            #     spearman = Spearman_eval(groundTruth,prediction)
            #     ccc = CCC_eval(groundTruth,prediction)
            #     print("RMSE = ",rmse)
            #     print("Spearman = ", spearman)
            #     print("CCC = ", ccc)
        print()
        print("GroundTruth.length = , ", len(groundTruth))
        print("Prediction.length = ", len(prediction))
        df = pd.DataFrame(data={"a_prediction": prediction, "a_groundtruth": groundTruth})
        df.to_csv("eval_arousal.csv")
        print("eval_arousal.csv save success!!")

        rmse = RMSE_evl(groundTruth, prediction)
        spearman = Spearman_eval(groundTruth, prediction)
        ccc = CCC_eval(groundTruth, prediction)
        print("RMSE = ", rmse)
        print("Spearman = ", spearman)
        print("CCC = ", ccc)
        print(Nan_list)


if __name__ == '__main__':
    main()
