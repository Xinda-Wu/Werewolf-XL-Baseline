"""
吴鑫达模型优化：
sudo python3 -m torch.distributed.launch Arousal_main_conv6.py
"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from Pleasure_TestFileDataset import *
from Pleasure_TrainFileDataset import *
from Pleasure_ValidFileDataset import *
from VGGFace_conv6_FeatureExtraction import VGG_Face
from torch.utils.tensorboard import SummaryWriter
from LSTM_conv6 import FineTuneLstmModel
import random
from pytorchtools import *
from math import sqrt
from sklearn.metrics import mean_squared_error
torch.set_default_tensor_type(torch.FloatTensor)

import logging
logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler("./log/20210628_LSTM_Regression_Pleasure_Latest_log.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Writer will output to ./runs/ directory by default
writer_train = SummaryWriter(log_dir='./runs/Regression_2/Pleasure/train/')
writer_valid = SummaryWriter(log_dir='./runs/Regression_2/Pleasure/valid/')
writer_test = SummaryWriter(log_dir='./runs/Regression_2/Pleasure/test/')

parser = argparse.ArgumentParser(description='Training')  # description: 描述这个参数解析器是用于干什么的
# 当'-'和'--'同时出现的时候，系统默认后者为参数名，前者不是，但是在命令行输入的时候没有这个区分
parser.add_argument('--local_rank', type=int, )
# batch_size
parser.add_argument('-b', '--batch-size', default=12, type=int,
                    metavar='N', help='mini-batch size (default: 1)')

# Early stopping
parser.add_argument('-p', '--patience', default=5, type=int,
                    help='early stopping (default: 7)')

# valid_batch_size
parser.add_argument('-vb', '--valid_batch_size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 1)')


# test_batch_size
parser.add_argument('-tb', '--test_batch_size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 1)')
# workers
parser.add_argument('-j', '--workers', default=20, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
# epochs
parser.add_argument('--epochs', default=101, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
# start_epoch
parser.add_argument('--start_epochs', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
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


def RMSE_evl(groundTruth,prediction):
    return sqrt(mean_squared_error(groundTruth, prediction))


def Spearman_eval(groundTruth,prediction):
    data = {'result':prediction, 'y_test':groundTruth}
    df = pd.DataFrame(data, columns=['result','y_test'])
    spearman = df.corr(method="spearman")
    spearman_values = spearman.iloc[0].values[1]
    return spearman_values


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


def load_checkpoint(model, path, optimizer):
    if os.listdir(path) != None:
        model_CKPT = torch.load(path)
        model.load_state_dict(model_CKPT['state_dict'])
        print('loaded checkpoint')
        optimizer.load_state_dict(model_CKPT['optimizer'])
    return model, optimizer


def train(train_loader, featureExtractor, model, criterion, optimizer, epoch, device, cv):
    model.train()  # switch to train
    print(f"Training Progress: CV = {cv}/10, Epoch = {epoch}/100")
    train_loss = []
    for i, videoFrames in enumerate(tqdm(train_loader)):
        label = videoFrames['label'].to(device)
        videoFrames = videoFrames['videoFrames'].reshape([-1, 3, 224, 224]).to(device)
        features = featureExtractor(videoFrames.float())
        output, hidden = model(features.unsqueeze(0))
        output = output.reshape([-1, 16, 1])
        output = output.reshape([-1, 16])
        output_mean = torch.mean(output, dim=1)
        loss = criterion(output_mean.float(), label.float())
        train_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_average = round(np.average(train_loss), 4)
    writer_train.add_scalar(f"CV_{cv}_MSELoss", loss_average, epoch)
    return loss_average


def valid(valid_loader, featureExtractor, model, criterion, optimizer, epoch, device, cv):
    print(" start to valid! ")
    model.eval()
    valid_losses = []
    with torch.no_grad():
        for i, videoFrames in enumerate(tqdm(valid_loader)):
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
                output_mean = torch.mean(output).unsqueeze(0)  # one serie of frames = 16
                Outputs.append(output_mean)  # All series of frames

            Outputs = torch.Tensor(Outputs)
            outputs_average = torch.mean(Outputs)  # average of All series' output
            loss = criterion(outputs_average.to(device), label)

            if torch.isnan(loss):
                continue
            else:
                valid_losses.append(loss.item())

    valid_losses_avg = round(np.average(valid_losses), 4)
    writer_valid.add_scalar(f"CV_{cv}_MSELoss", valid_losses_avg, epoch)
    print(" Valid: Epoch = {0}/{1}, losses = {2}".format(epoch, args.epochs, valid_losses_avg))

    return valid_losses_avg


def test(test_loader, featureExtractor, model, device, cv):
    model.eval()  # switch to train
    with torch.no_grad():
        groundTruth = []
        prediction = []
        for i, videoFrames in enumerate(tqdm(test_loader)):
            label = videoFrames['label'].numpy()
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
                groundTruth.append(label.item())
                prediction.append(0)
            else:
                groundTruth.append(label.item())
                prediction.append(outputs_average.item())

        rmse = RMSE_evl(groundTruth, prediction)
        spearman = Spearman_eval(groundTruth, prediction)
        ccc = CCC_eval(groundTruth, prediction)
        print(f"Test result RMSE = {rmse}, Spearman = {spearman}, CCC = {ccc}")
        logger.info(f"CV ={cv}, Test result RMSE = {rmse}, Spearman = {spearman}, CCC = {ccc}")
        # save result
        df = pd.DataFrame(data={"p_prediction": prediction, "p_groundtruth": groundTruth})
        df.to_csv(f"./Prediction/20210628/eval_pleasure_CV_{cv}_RMSE_{rmse}_Spearman_{spearman}_CCC_{ccc}.csv")
        return [rmse, spearman, ccc]


def main():
    global args, start_loss, distance
    set_seed()
    args = parser.parse_args()

    # Multiple GPU training , load device
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'  # 想使用的GPU编号
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # GPU开始编号，依次往下
    torch.distributed.init_process_group(backend="nccl", init_method='tcp://localhost:23470', rank=0, world_size=1)
    print("load into device:", device)

    # init visual feature extractor
    featureExtractor = VGG_Face()
    featureExtractor = load_GPUS(featureExtractor, '/mnt/nfs-shared/xinda/Werewolf-XL/werewolf_video/vgg16_face_30.pth')
    featureExtractor = nn.DataParallel(featureExtractor)
    featureExtractor.to(device)
    print('load visual feature extractor success')

    # -----------------
    # Cross Validation
    # -----------------
    average_rmse_test, average_pearson_test, average_ccc_test = [], [], []
    for i in range(1,11):
        # Load Data
        train_dataset = FlameSet_Train(i)
        valid_dataset = FlameSet_Valid(i)
        test_dataset = FlameSet_Test(i)

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size, shuffle=False,
                                                   num_workers=args.workers,
                                                   pin_memory=False)  # pin_memory: 如果True，数据加载器会在返回之前将Tensors复制到CUDA固定内存

        valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                                  batch_size=args.test_batch_size, shuffle=False,
                                                  num_workers=args.workers, pin_memory=False)

        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=args.test_batch_size, shuffle=False,
                                                  num_workers=args.workers, pin_memory=False)

        # load LSTM model, 多GPU计算
        model = FineTuneLstmModel()
        model = model.to(device)
        model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)

        # loss criterion and optimizer
        criterion = nn.MSELoss()
        criterion = criterion.to(device)  # 并行化损失函数
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, )

        # Training on epochs
        avg_train_loss = []
        avg_valid_loss = []

        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=f"./checkpoints/20210628/early_stopping_checkpoint_CV_{i}.pt")

        for epoch in range(args.start_epochs, args.epochs):


            # -------------------
            # train the model
            # -------------------
            avg_train_loss_item = train(train_loader, featureExtractor, model, criterion, optimizer, epoch, device, cv=i)
            avg_train_loss.append(avg_train_loss_item)

            # -------------------
            # Validate the model
            # -------------------
            if epoch % 5 == 0:
                loss = valid(valid_loader, featureExtractor, model, criterion, optimizer, epoch, device, cv=i)
                avg_valid_loss.append(loss)
                # save model
                # early_stopping needs the validation loss to check if it has decresed,
                # and if it has, it will make a checkpoint of the current model
                early_stopping(loss, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

        # -------------------
        # Test the model
        # -------------------
        model.load_state_dict(torch.load(f"./checkpoints/20210628/early_stopping_checkpoint_CV_{i}.pt"))
        test_result = test(test_loader, featureExtractor, model, device, cv=i)
        average_rmse_test.append(test_result[0])
        average_pearson_test.append(test_result[1])
        average_ccc_test.append(test_result[2])
        print(f"CV {i}/10 >>>>LSTM Regression Average Results of Pleasure: RMSE.= {np.average(average_rmse_test)},\
         Spearman = {np.average(average_pearson_test)}, CCC = {np.average(average_ccc_test)}")
        logger.info(f"CV {i}/10 LSTM Regression Average Results of Pleasure: RMSE.= {np.average(average_rmse_test)},\
         Spearman = {np.average(average_pearson_test)}, CCC = {np.average(average_ccc_test)}")


if __name__ == '__main__':
    main()
