# "python3 -m torch.distributed.launch PNN_LSTM.py"
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import numpy as np
from tqdm import tqdm
import PNN_TrainFileDataset
import PNN_TestFileDatabase
import PNN_ValidFileDatabase
from VGGFace_conv6_FeatureExtraction import VGG_Face
from torch.utils.tensorboard import SummaryWriter
from LSTM_conv6_PNN import FineTuneLstmModel
from pytorchtools import *
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import random
import random
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, recall_score, accuracy_score, f1_score, auc,roc_curve
from sklearn.preprocessing import label_binarize
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import functional as F
from torch.utils.data import WeightedRandomSampler, DataLoader

# torch.set_default_tensor_type(torch.FloatTensor)

import logging
logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler("./log/LSTM_Classification_Latest_log.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


# Writer will output to ./runs/ directory by default
writer_train = SummaryWriter(log_dir='./runs/Classification/PNN_train/')
writer_valid = SummaryWriter(log_dir='./runs/Classification/PNN_valid/')
writer_test = SummaryWriter(log_dir='./runs/Classification/PNN_test/')

parser = argparse.ArgumentParser(description='Training')  # description: 描述这个参数解析器是用于干什么的
# 当'-'和'--'同时出现的时候，系统默认后者为参数名，前者不是，但是在命令行输入的时候没有这个区分
parser.add_argument('--local_rank', type=int, )
# batch_size
parser.add_argument('-b', '--batch-size', default=12, type=int,
                    metavar='N', help='mini-batch size (default: 1)')

# Early stopping
parser.add_argument('-p', '--patience', default=2, type=int,
                    help='early stopping (default: 7)')

#  valid_batch_size
parser.add_argument('-vb', '--valid_batch_size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 1)')
#  test_batch_size
parser.add_argument('-tb', '--test_batch_size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 1)')
# workers
parser.add_argument('-j', '--workers', default=20, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
# epochs
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
# start_epoch
parser.add_argument('--start_epochs', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
args = parser.parse_known_args()[0]

classes = ('中性', '高兴', '疑惑', '紧张','轻松', '尴尬')


best_prec1 = 0

def set_seed(seed=1):  # seed setting
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


def train(train_loader, featureExtractor, model, criterion, optimizer, epoch, device, cv):
    model.train()  # switch to train
    print(f"Training Progress: CV = {cv}/10, Epoch = {epoch}/100")
    train_loss = []
    train_correct = 0
    train_total = 0
    for i, videoFrames in enumerate(tqdm(train_loader)):
        label = videoFrames['label'].to(device)
        videoFrames = videoFrames['videoFrames'].reshape([-1, 3, 224, 224]).to(device)
        features = featureExtractor(videoFrames.float())
        output, hidden = model(features.unsqueeze(0)) # input(seq_len, batch, input_size)
        output = output.reshape([-1, 16, 6])
        output_mean = torch.mean(output, dim=1)
        loss = criterion(output_mean, label)
        train_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(output_mean.data, 1)
        train_total += label.size(0)
        train_correct += (predicted == label).sum().item()

    train_accuracy = 100 * train_correct / train_total
    print('Accuracy of the network on the Training images: %d %%' % (train_accuracy))
    loss_average = round(np.average(train_loss), 4)
    writer_train.add_scalar(f"CV_{i}_MSELoss", loss_average, epoch)
    writer_train.add_scalar(f"CV_{i}_Accuracy", train_accuracy, epoch)
    return loss_average


def valid(valid_loader, featureExtractor, model, criterion, optimizer, epoch, device, cv):
    print(" start to valid! ")
    model.eval()  # switch to train
    valid_losses = []
    valid_correct = 0
    valid_total = 0
    valid_accuracy = 0.0
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
                output_mean = torch.mean(output, dim=0)  # one serie of frames = 16
                Outputs.append(output_mean.data.cpu().numpy().tolist())  # All series of frames

            Outputs = torch.Tensor(Outputs)


            if Outputs.shape[0]>1:
                outputs_average = torch.mean(Outputs, dim=0).unsqueeze(0)  # average of All series' output

            loss = criterion(outputs_average.to(device), label)
            if torch.isnan(loss):
                continue
            else:
                valid_losses.append(loss.item())
            _, predicted = torch.max(outputs_average.data, 1)
            valid_total += label.size(0)
            valid_correct += (predicted == label.data.cpu()).sum().item()

        # 与train 的loss大小规模相近（可以在同一张图中显示）
        valid_losses_avg = round(np.average(valid_losses), 4)
        valid_accuracy = 100 * valid_correct / valid_total
        writer_valid.add_scalar(f"CV_{i}_MSELoss", valid_losses_avg, epoch)
        writer_valid.add_scalar(f"CV_{i}_Accuracy", valid_accuracy, epoch)
        print('CV = %d, Epoch %d, Accuracy of the network on the Valid images: %d %%' % (cv, epoch, valid_accuracy))
        return valid_losses_avg



def test(test_loader, featureExtractor, model, epoch, device, cv):
    model.eval()  # switch to train
    groundTruth = []
    prediction_max = []
    prediction_prob = []
    test_correct = 0
    test_total = 0
    with torch.no_grad():
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

            if Outputs.shape[0] > 1:
                outputs_average = torch.mean(Outputs, dim=0).unsqueeze(0)  # average of All series' output

            groundTruth.append(label.item())
            _, predicted = torch.max(outputs_average.data, 1)
            prediction_max.append(predicted.item())
            prediction_prob_b = F.softmax(outputs_average.data)
            prediction_prob.append(prediction_prob_b.data.numpy().reshape(6).tolist())

            test_total += label.size(0)
            test_correct += (predicted == label.data.cpu()).sum().item()

        accuracy = accuracy_score(prediction_max, groundTruth)
        f1 = f1_score(prediction_max, groundTruth, average="weighted")
        label = label_binarize(groundTruth, classes=list(range(6)))
        auc = roc_auc_score(label, prediction_prob, average='micro')
        print(f"CV {cv}/10, Epoch {epoch}/100, accuracy = {accuracy}, F1-Score = {f1}, AUC = {auc}", )

        test_accuracy = 100 * test_correct / test_total
        print('CV = %d, Epoch %d, Accuracy of the network on the Test images: %d' % (cv, epoch, test_accuracy))

        # Raw
        df = pd.DataFrame(data={"pnn_prediction": prediction_max, "pnn_groundtruth": groundTruth})
        df.to_csv(f"./Prediction_202106/CV_{cv}_Epoch_{epoch}_ACC_{test_accuracy}_eval_pnn_2.csv")

        pro = np.array(prediction_prob)
        df2 = pd.DataFrame(pro)
        df2.to_csv(f"./Prediction_202106/CV_{cv}_Epoch_{epoch}_ACC_{test_accuracy}_Categorical_lstm_6pnn_202106_2.csv")
        print(f"save cv {cv}")
        return [accuracy, f1, auc]


def main():
    global args, start_loss, distance, best_prec1
    set_seed()
    args = parser.parse_args()

    # Multiple GPU training , load device
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'  # 想使用的GPU编号
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # GPU开始编号，依次往下
    # DDP
    torch.distributed.init_process_group(backend="nccl", init_method='tcp://localhost:23465', rank=0, world_size=1)

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
    average_acc_test, average_f1_test, average_auc_test = [], [], []
    for i in range(4,5): #
        # Load Data
        train_dataset = PNN_TrainFileDataset.FlameSet_Train(i)  # for training 0.7
        valid_dataset = PNN_ValidFileDatabase.FlameSet_Valid(i)  # for test  0.3
        test_dataset = PNN_TestFileDatabase.FlameSet_Test(i)  # for test  0.3

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size, shuffle=False,
                                                   num_workers=args.workers,
                                                   pin_memory=False)  # pin_memory: 如果True，数据加载器会在返回之前将Tensors复制到CUDA固定内存

        valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                                   batch_size=args.valid_batch_size, shuffle=False,
                                                   num_workers=args.workers, pin_memory=False)

        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=args.test_batch_size, shuffle=False,
                                                  num_workers=args.workers, pin_memory=False)
        # load LSTM model, 多GPU计算
        model = FineTuneLstmModel()
        model = model.to(device)
        model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)

        # loss criterion and optimizer
        train_y = pd.read_csv(
            f'/mnt/nfs-shared/xinda/Werewolf-XL/Werewolf-XL_202106/2_LSTM/LSTM/Split_dataset/CV_Features/ClassificationFeatures/Train_CV_{i}.csv').iloc[
                  :, 5]
        encoder = LabelEncoder().fit(train_y)  # #训练LabelEncoder, 把y_train中的类别编码为0，1，2，3，4，5
        y = encoder.transform(train_y)
        y_train = pd.DataFrame(encoder.transform(train_y))  # 使用训练好的LabelEncoder对源数据进行编码
        class_weights = torch.tensor(list(compute_class_weight('balanced', np.unique(y_train), y_train))).float()
        print("Class Weight = ", class_weights)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        criterion = criterion.to(device)  # 并行化损失函数
        # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Training on epochs
        avg_train_loss = []
        avg_valid_loss = []

        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=args.patience, verbose=True,
                                       path=f"./checkpoints/early_stopping_checkpoint_CV_{i}.pt")

        # Training on epochs
        for epoch in range(args.start_epochs,  args.epochs):
            # tst
            # if epoch ==1:
            #     # -------------------
            #     # Test the model
            #     # -------------------
            #     print(">>>>> Test Model")
            #     model.load_state_dict(torch.load(f"./checkpoints/early_stopping_checkpoint_CV_{i}.pt"))
            #     test_result = test(test_loader, featureExtractor, model, epoch=epoch, device=device, cv=i)
            #     average_acc_test.append(test_result[0])
            #     average_f1_test.append(test_result[1])
            #     average_auc_test.append(test_result[2])
            #     print(
            #         f"CV {i}/10 >>>>LSTM Classificaiton Average Results of Arousal: ACC.= {np.average(average_acc_test)},\
            #         F1 Score = {np.average(average_f1_test)}, AUC = {np.average(average_auc_test)}")
            #
            #     loss = valid(valid_loader, featureExtractor, model, criterion, optimizer, epoch, device, cv=i)
            #     avg_valid_loss.append(loss)
            #     # save model
            #     torch.save(model, './checkpoints/Arousal_lstm_loss_' + str("%.4f" % loss) + '.pth')
            #
            #     # early_stopping needs the validation loss to check if it has decresed,
            #     # and if it has, it will make a checkpoint of the current model
            #     early_stopping(loss, model)
            #     if early_stopping.early_stop:
            #         print("Early stopping")
            #         break


            # -------------------
            # train the model
            # -------------------
            avg_train_loss_item = train(train_loader, featureExtractor, model, criterion, optimizer, epoch, device,
                                        cv=i)
            avg_train_loss.append(avg_train_loss_item)

            # -------------------
            # Validate the model
            # -------------------
            if epoch % 5 == 0:
                loss = valid(valid_loader, featureExtractor, model, criterion, optimizer, epoch, device, cv=i)
                avg_valid_loss.append(loss)
                # save model
                torch.save(model, f'./checkpoints/CV_{i}_Classification_lstm_loss_' + str("%.4f" % loss) + '.pth')

                # early_stopping needs the validation loss to check if it has decresed,
                # and if it has, it will make a checkpoint of the current model
                early_stopping(loss, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    torch.save(model, f'./checkpoints/CV_{i}_Classification_lstm_loss_{str("%.4f" % loss)}_earlyStopping_{epoch}.pth')
                    break

                    # -------------------
                    # Test the model
                    # -------------------


        model.load_state_dict(torch.load(f"./checkpoints/early_stopping_checkpoint_CV_{i}.pt"))
        test_result = test(test_loader, featureExtractor, model, epoch, device, cv=i)
        average_acc_test.append(test_result[0])
        average_f1_test.append(test_result[1])
        average_auc_test.append(test_result[2])
        print(f"CV {i}/10 >>>>LSTM Classificaiton Average Results of Arousal: ACC.= {np.average(average_acc_test)},\
            F1 Score = {np.average(average_f1_test)}, AUC = {np.average(average_auc_test)}")
        logger.info(
            f"CV {i}/10 LSTM Regression Average Results of Arousal: RMSE.= {np.average(average_acc_test)},\
            F1 Score = {np.average(average_f1_test)}, AUC = {np.average(average_auc_test)}")



        # # for epoch in range(0, 1):
        #     # ===============  Saving Model  ===================
        #     if epoch % 10 == 0 and epoch != 0:
        #         torch.save(model.module.state_dict(), './checkpoints/PNN_LSTM_2layer_' + str(epoch) + ".pth")
        #
        #     # ===============  Training Model  ===================
        #     train(train_loader, featureExtractor, model, criterion, optimizer, epoch, device)
        #
        #
        #     # ===============  Valid Model  ===================
        #     if epoch % 5==0 :
        #         accuracy = valid(valid_loader, featureExtractor, model, criterion, optimizer, epoch, device)
        #         # ===============  Saving Model  ===================
        #         is_best = accuracy > best_prec1
        #         best_prec1 = max(accuracy, best_prec1)
        #         if is_best:
        #             torch.save({'epoch': epoch,
        #                         'state_dict': model.module.state_dict(),
        #                         'optimizer': optimizer.state_dict(),
        #                         'best_prec1': best_prec1, },
        #                        './checkpoints/PNN_LSTM_2layer_params_' + str(epoch) + "_" + str("%.4f" % best_prec1) + '.pth')
        #             torch.save(model.module.state_dict(), './checkpoints/bes_PNN_LSTM_2layer_' + str(epoch) + ".pth")
        #
        #             print("save the VGGFace model  {0} model".format(best_prec1))
        #
        #     print('Finished Training')



if __name__ == '__main__':
    # main()


    # -----
    # Evaluation
    # -------
    set_seed()
    args = parser.parse_args()

    # Multiple GPU training , load device
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 想使用的GPU编号
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # GPU开始编号，依次往下
    # DDP
    torch.distributed.init_process_group(backend="nccl", init_method='tcp://localhost:23468', rank=0, world_size=1)

    print("load into device:", device)

    # init visual feature extractor
    featureExtractor = VGG_Face()
    featureExtractor = load_GPUS(featureExtractor, '/mnt/nfs-shared/xinda/Werewolf-XL/werewolf_video/vgg16_face_30.pth')
    featureExtractor = nn.DataParallel(featureExtractor)
    featureExtractor.to(device)
    print('load visual feature extractor success')

    test_dataset = PNN_TestFileDatabase.FlameSet_Test(4)  # for test  0.3
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.test_batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=False)
    # load LSTM model, 多GPU计算
    model = FineTuneLstmModel()
    model = model.to(device)
    model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    model.load_state_dict(torch.load(f"./checkpoints/CV_4_Classification_lstm_loss_1.3821.pth"))
    test_result = test(test_loader, featureExtractor, model, 999, device, cv=4)

