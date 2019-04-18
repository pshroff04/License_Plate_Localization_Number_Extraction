#encoding:utf-8
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import argparse
import numpy as np
from os import path, mkdir
from load_data import *
from roi_pooling import roi_pooling_ims
from shutil import copyfile
import utils

# import sys
# sys.argv = ['adsa.py','-i', './Test/home/booy/booy/ccpd_dataset/ccpd_base/', '-m', './weights_4.pth']

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="path to the input folder")
ap.add_argument("-m", "--model", required=True,
                help="path to the model file")
# ap.add_argument("-s", "--store", required=True,
#                 help="path to the store folder")
args = vars(ap.parse_args())

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
use_gpu = torch.cuda.is_available()
print ('Is GPU: {}'.format(use_gpu))


imgSize = (480, 480)
eval_batchSize = 1
resume_file = str(args["model"])
nh =256
provNum, alphaNum, adNum = 38, 25, 35
#Use to convert the 'index label' (ground truth) getting from dataloader to actual license plate string and then to encode it using converter.encode
####
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
             "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

## Define output variable for RNN
nOut = len(provinces[:-1]) + len(ads) +1 # +1 for CTC need '-'

#The utility function for encode and decode
converter = utils.strLabelConverter()
class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True, batch_first=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        #input of shape (batch,seq_len, input_size)
        #input = input.contiguous()
        recurrent, _ = self.rnn(input) # why hidden layers are not initialized???
        #output of shape (seq_len, batch, num_directions * hidden_size)
        b, T, h = recurrent.size()
        #print('Recurent.size() {}'.format(recurrent.size()))
        t_rec = recurrent.contiguous().view(b * T, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.contiguous().view(b, T, -1)

        return output

#Detection Module
class wR2(nn.Module):
    def __init__(self, num_classes=1000):
        super(wR2, self).__init__()
        hidden1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(num_features=48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=160, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=160),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden5 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden6 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden7 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden8 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden9 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden10 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        self.features = nn.Sequential(
            hidden1,
            hidden2,
            hidden3,
            hidden4,
            hidden5,
            hidden6,
            hidden7,
            hidden8,
            hidden9,
            hidden10
        )
        self.classifier = nn.Sequential(
            nn.Linear(23232, 100),
            # nn.ReLU(inplace=True),
            nn.Linear(100, 100),
            # nn.ReLU(inplace=True),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        x1 = self.features(x)# N*(w*h*192) #(N,C,W,H)
        x11 = x1.view(x1.size(0), -1) #(N,c*w*h = 23232)
        x = self.classifier(x11)
        return x

#Recognition Module
class fh02(nn.Module):
    def __init__(self, num_points, num_classes=nOut, wrPath=None):
        super(fh02, self).__init__()
        self.load_wR2(num_points, wrPath)
#         self.classifier1 = nn.Sequential(
#             # nn.Dropout(),
#             nn.Linear(53248, 128),
#             # nn.ReLU(inplace=True),
#             # nn.Dropout(),
#             nn.Linear(128, provNum),
#         )
#         self.classifier2 = nn.Sequential(
#             # nn.Dropout(),
#             nn.Linear(53248, 128),
#             # nn.ReLU(inplace=True),
#             # nn.Dropout(),
#             nn.Linear(128, alphaNum),
#         )
#         self.classifier3 = nn.Sequential(
#             # nn.Dropout(),
#             nn.Linear(53248, 128),
#             # nn.ReLU(inplace=True),
#             # nn.Dropout(),
#             nn.Linear(128, adNum),
#         )
#         self.classifier4 = nn.Sequential(
#             # nn.Dropout(),
#             nn.Linear(53248, 128),
#             # nn.ReLU(inplace=True),
#             # nn.Dropout(),
#             nn.Linear(128, adNum),
#         )
#         self.classifier5 = nn.Sequential(
#             # nn.Dropout(),
#             nn.Linear(53248, 128),
#             # nn.ReLU(inplace=True),
#             # nn.Dropout(),
#             nn.Linear(128, adNum),
#         )
#         self.classifier6 = nn.Sequential(
#             # nn.Dropout(),
#             nn.Linear(53248, 128),
#             # nn.ReLU(inplace=True),
#             # nn.Dropout(),
#             nn.Linear(128, adNum),
#         )
#         self.classifier7 = nn.Sequential(
#             # nn.Dropout(),
#             nn.Linear(53248, 128),
#             # nn.ReLU(inplace=True),
#             # nn.Dropout(),
#             nn.Linear(128, adNum),
#         )
        self.rnn = nn.Sequential(
        BidirectionalLSTM(416,nh,nh),
        BidirectionalLSTM(nh,nh,num_classes)
        )
        #self.rnn = nn.DataParallel(rnn, dim=1, device_ids=range(torch.cuda.device_count()))

    def load_wR2(self, num_points, path=None):
        self.wR2 = wR2(num_points)
        if use_gpu:
            self.wR2 = torch.nn.DataParallel(self.wR2, device_ids=range(torch.cuda.device_count()))
        if not path is None:
            self.wR2.load_state_dict(torch.load(path))
            # self.wR2 = self.wR2.cuda()
        # for param in self.wR2.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        x0 = self.wR2.module.features[0](x)
        _x1 = self.wR2.module.features[1](x0)
        x2 = self.wR2.module.features[2](_x1)
        _x3 = self.wR2.module.features[3](x2)
        x4 = self.wR2.module.features[4](_x3)
        _x5 = self.wR2.module.features[5](x4)

        x6 = self.wR2.module.features[6](_x5)
        x7 = self.wR2.module.features[7](x6)
        x8 = self.wR2.module.features[8](x7)
        x9 = self.wR2.module.features[9](x8)
        x9 = x9.view(x9.size(0), -1)
        boxLoc = self.wR2.module.classifier(x9)

        # x, y, w, h --> x1, y1, x2, y2
        assert boxLoc.data.size()[1] == 4
        if use_gpu:
            postfix = Variable(torch.FloatTensor([[1,0,1,0],[0,1,0,1],[-0.5,0,0.5,0],[0,-0.5,0,0.5]]).cuda(), requires_grad=False)
        else:
            postfix = Variable(torch.FloatTensor([[1,0,1,0],[0,1,0,1],[-0.5,0,0.5,0],[0,-0.5,0,0.5]]), requires_grad=False)
        boxNew = boxLoc.mm(postfix).clamp(min=0, max=1)

        h1, w1 = _x1.data.size()[2], _x1.data.size()[3]
        h2, w2 = _x3.data.size()[2], _x3.data.size()[3]
        h3, w3 = _x5.data.size()[2], _x5.data.size()[3]
        if use_gpu:
            p1 = Variable(torch.FloatTensor([[w1,0,0,0],[0,h1,0,0],[0,0,w1,0],[0,0,0,h1]]).cuda(), requires_grad=False)
            p2 = Variable(torch.FloatTensor([[w2,0,0,0],[0,h2,0,0],[0,0,w2,0],[0,0,0,h2]]).cuda(), requires_grad=False)
            p3 = Variable(torch.FloatTensor([[w3,0,0,0],[0,h3,0,0],[0,0,w3,0],[0,0,0,h3]]).cuda(), requires_grad=False)
        else:
            p1 = Variable(torch.FloatTensor([[w1,0,0,0],[0,h1,0,0],[0,0,w1,0],[0,0,0,h1]]), requires_grad=False)
            p2 = Variable(torch.FloatTensor([[w2,0,0,0],[0,h2,0,0],[0,0,w2,0],[0,0,0,h2]]), requires_grad=False)
            p3 = Variable(torch.FloatTensor([[w3,0,0,0],[0,h3,0,0],[0,0,w3,0],[0,0,0,h3]]), requires_grad=False)

        # input = Variable(torch.rand(2, 1, 10, 10), requires_grad=True)
        # rois = Variable(torch.LongTensor([[0, 1, 2, 7, 8], [0, 3, 3, 8, 8], [1, 3, 3, 8, 8]]), requires_grad=False)
        roi1 = roi_pooling_ims(_x1, boxNew.mm(p1), size=(16, 8))#(N,64,8,16)
        roi2 = roi_pooling_ims(_x3, boxNew.mm(p2), size=(16, 8))#(N,160,8,16)
        roi3 = roi_pooling_ims(_x5, boxNew.mm(p3), size=(16, 8))#(N,192,8,16)
        rois = torch.cat((roi1, roi2, roi3), 1)#(N,416,8,16)

        #_rois = rois.view(rois.size(0), -1)
        rois_r = rois.view(rois.shape[0],rois.shape[1],-1)  #(N,416,1,128)
        #rois_r = rois_r.squeeze(2)  #(N,416,128)
        rois_r = rois_r.permute(0,2,1)  #(N,128,416)
        #print('Input to RNN: {}'.format(rois_r.shape))
        output = self.rnn(rois_r) #(128,N,nout)

        # y0 = self.classifier1(_rois)
        # y1 = self.classifier2(_rois)
        # y2 = self.classifier3(_rois)
        # y3 = self.classifier4(_rois)
        # y4 = self.classifier5(_rois)
        # y5 = self.classifier6(_rois)
        # y6 = self.classifier7(_rois)
        #return boxLoc, [y0, y1, y2, y3, y4, y5, y6]
        #print('output.shape: {}'.format(output.shape))
        return boxLoc, output


def isEqual(labelGT, labelP):
    # print (labelGT)
    # print (labelP)
    compare = [1 if int(labelGT[i]) == int(labelP[i]) else 0 for i in range(7)]
    # print(sum(compare))
    return sum(compare)

numPoints=4
model_conv = fh02(numPoints, nOut)
model_conv = torch.nn.DataParallel(model_conv, device_ids=range(torch.cuda.device_count()))
model_conv.load_state_dict(torch.load(resume_file))
print('Pretrained weights {} loaded !!...'.format(resume_file))
model_conv = model_conv.cuda()
model_conv.eval()

# efficiency evaluation
# dst = imgDataLoader([args["input"]], imgSize)
# trainloader = DataLoader(dst, batch_size=batchSize, shuffle=True, num_workers=4)
#
# start = time()
# for i, (XI) in enumerate(trainloader):
#     x = Variable(XI.cuda(0))
#     y_pred = model_conv(x)
#     outputY = y_pred.data.cpu().numpy()
#     #   assert len(outputY) == batchSize
# print("detect efficiency %s seconds" %(time() - start))


count = 0
error = 0
sixCorrect = 0
# sFolder = str(args["store"])
# sFolder = sFolder if sFolder[-1] == '/' else sFolder + '/'
# if not path.isdir(sFolder):
#     mkdir(sFolder)

dst = labelTestDataLoader(args["input"].split(','), imgSize)
evalloader = DataLoader(dst, batch_size=eval_batchSize, shuffle=True, num_workers=1)

with open('eval.out', 'w') as outF:
    outF.write('Starting Evaluation ....')
print('Logs will be appended to {} file'.format('eval.out'))

eval_text = Variable(torch.IntTensor(eval_batchSize * 5))
eval_length = Variable(torch.IntTensor(eval_batchSize))
char_correct = 0
label_correct = 0
total_characters = 0
for i, (XI, labels, ims) in enumerate(evalloader):
    YI =[] #List of all the licence plate (string of actual licence plates)
    for label in labels:
        indexs =[int(x) for x in label.split('_')[:7]]
        l =[provinces[indexs[0]], alphabets[indexs[1]]]
        for index in range(2,7):
            l.append(ads[indexs[index]])
        YI.append(''.join(l))
    total_characters += len(YI[0])
    
    t, l = converter.encode(YI)
    utils.loadData(eval_text, t)
    utils.loadData(eval_length, l)

    #YI = [[int(ee) for ee in el.split('_')[:7]] for el in labels]
    if use_gpu:
        x = Variable(XI.cuda())
    else:
        x = Variable(XI)
    # Forward pass: Compute predicted y by passing x to the model
    #print('X: {}'.format(x))
    fps_pred, preds = model_conv(x)

    #Greedy-Best Path decoding
    _, preds = preds.max(2, keepdim=True)
    preds = preds.squeeze(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * 1))
    rsim_preds = converter.decode(preds.data, preds_size.data, raw=True)
    sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
    #print('rsim_preds: {}, sim_preds {} and YI is {}'.format(str(rsim_preds.encode('utf-8')), sim_preds.encode('utf-8'), YI))
    
    if (sim_preds == YI[0].lower()):
        label_correct +=1
    for pred, target in zip(sim_preds, YI[0]):
        if pred == target.lower():
            char_correct += 1
    with open('eval.out', 'a', encoding="utf-8") as outF:
        outF.write('\n Prediction for current test image: {} and YI (Ground-Truth) is {}.'.format(sim_preds.upper(), YI[0]))
        if (i+1) % 50 == 0:
            outF.write('\nImages completed: %s, character-level accuracy: %s, label-accuracy: %s.' % ((i+1),char_correct/total_characters, label_correct/(i+1)))

with open('eval.out', 'a') as outF:
    outF.write('\n=========================Final Accuracy=======================================')
    outF.write('\n Total test images: %s, Character level: %s, label-level: %s' % ((i+1),char_correct/total_characters, label_correct/(i+1)))
    outF.write('\n==============================================================================')
