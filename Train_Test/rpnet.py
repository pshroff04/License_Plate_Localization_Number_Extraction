from __future__ import print_function, division
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import os
import argparse
from time import time
from load_data import *
from roi_pooling import roi_pooling_ims
from torch.optim import lr_scheduler
#from torch_baidu_ctc import ctc_loss, CTCLoss
from tensorboardX import SummaryWriter
import utils

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
                help="path to the input file")
ap.add_argument("-n", "--epochs", default=5,
                help="epochs for train")
ap.add_argument("-b", "--batchsize", default=128,
                help="batch size for train")
ap.add_argument("-se", "--start_epoch", default=0,
                help="start epoch for train")
ap.add_argument("-t", "--test",
                help="dirs for test")
ap.add_argument("-r", "--resume", default='111',
                help="file for re-train")
ap.add_argument("-f", "--folder", required=True,
                help="folder to store model")
ap.add_argument("-w", "--writeFile", default='logs.out',
                help="file for output")
ap.add_argument('-nh', type =int, default = 256, help='hidden size')
ap.add_argument('-dw', required=True, help='weights of detection stage')

args = vars(ap.parse_args())
writer = SummaryWriter()
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
#######
wR2Path = args["dw"]
print('Detection weights: {}'.format(wR2Path))
use_gpu = torch.cuda.is_available()
nh = args['nh']
print ('Is GPU available: {}'.format(use_gpu))

#numClasses = 7

#This denotes the points for rectangle box of licence plate detection
numPoints = 4
classifyNum = 35

imgSize = (480, 480)
# lpSize = (128, 64)
#provNum, alphaNum, adNum = 38, 25, 35
batchSize = int(args["batchsize"]) if use_gpu else 2
trainDirs = args["images"].split(',')
testDirs = args["test"].split(',')

#Folder to save the models in between
modelFolder = str(args["folder"]) if str(args["folder"])[-1] == '/' else str(args["folder"]) + '/'
storeName = modelFolder + 'weights_'
if not os.path.isdir(modelFolder):
    os.mkdir(modelFolder)


epochs = int(args["epochs"])
#   initialize the output file
if not os.path.isfile(args['writeFile']):
    with open(args['writeFile'], 'wb') as outF:
        pass

#The utility function for encode and decode
converter = utils.strLabelConverter()

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

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
        self.load_wR2(wrPath)
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

    def load_wR2(self, path):
        self.wR2 = wR2(numPoints)
        if use_gpu:
            self.wR2 = torch.nn.DataParallel(self.wR2, device_ids=range(torch.cuda.device_count()))
        if not path is None:
            self.wR2.load_state_dict(torch.load(path))
            # self.wR2 = self.wR2.cuda()
        for param in self.wR2.parameters():
            param.requires_grad = False

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
        #print('BoxLox is {} and shape is {}'.format(boxLoc, boxLoc.shape))

        h1, w1 = _x1.data.size()[2], _x1.data.size()[3]
        p1 = Variable(torch.FloatTensor([[w1,0,0,0],[0,h1,0,0],[0,0,w1,0],[0,0,0,h1]]).cuda(), requires_grad=False)
        h2, w2 = _x3.data.size()[2], _x3.data.size()[3]
        p2 = Variable(torch.FloatTensor([[w2,0,0,0],[0,h2,0,0],[0,0,w2,0],[0,0,0,h2]]).cuda(), requires_grad=False)
        h3, w3 = _x5.data.size()[2], _x5.data.size()[3]
        p3 = Variable(torch.FloatTensor([[w3,0,0,0],[0,h3,0,0],[0,0,w3,0],[0,0,0,h3]]).cuda(), requires_grad=False)

        # x, y, w, h --> x1, y1, x2, y2
        assert boxLoc.data.size()[1] == 4
        postfix = Variable(torch.FloatTensor([[1,0,1,0],[0,1,0,1],[-0.5,0,0.5,0],[0,-0.5,0,0.5]]).cuda(), requires_grad=False)
        boxNew = boxLoc.mm(postfix).clamp(min=0, max=1)
        #print('boxNew is {} and shape is {}'.format(boxNew , boxNew .shape))

        # input = Variable(torch.rand(2, 1, 10, 10), requires_grad=True)
        # rois = Variable(torch.LongTensor([[0, 1, 2, 7, 8], [0, 3, 3, 8, 8], [1, 3, 3, 8, 8]]), requires_grad=False)
        #print(' boxNew.mm(p1) shape: {}'.format( torch.mm(boxNew,p1)))
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

epoch_start = int(args["start_epoch"])
resume_file = str(args["resume"])
if not resume_file == '111':
    # epoch_start = int(resume_file[resume_file.find('pth') + 3:]) + 1
    if not os.path.isfile(resume_file):
        print ("fail to load existed model! Existing ...")
        exit(0)
    print ("Load existed model! %s" % resume_file)
    model_conv = fh02(numPoints, nOut)
    model_conv = torch.nn.DataParallel(model_conv, device_ids=range(torch.cuda.device_count()))
    model_conv.load_state_dict(torch.load(resume_file))
    model_conv = model_conv.cuda()
else:
    model_conv = fh02(numPoints, nOut, wR2Path)
    if use_gpu:
        model_conv = torch.nn.DataParallel(model_conv, device_ids=range(torch.cuda.device_count()))
        model_conv = model_conv.cuda()

print(model_conv)
#print(get_n_params(model_conv))

#criterion = nn.CrossEntropyLoss()
criterion = nn.CTCLoss(reduction='mean')
# optimizer_conv = optim.RMSprop(model_conv.parameters(), lr=0.01, momentum=0.9)
#optimizer_conv = optim.SGD(model_conv.parameters(), lr=0.001, momentum=0.9)
optimizer_conv = optim.Adam(model_conv.parameters(), lr= 0.001)

dst = labelFpsDataLoader(trainDirs, imgSize)
trainloader = DataLoader(dst, batch_size=batchSize, shuffle=True, num_workers=8)
lrScheduler = lr_scheduler.StepLR(optimizer_conv, step_size=5, gamma=0.1)

#image = Variable(torch.FloatTensor(batchSize, 3, 480, 480))

print('Total number of training images: {}'.format(len(dst)))

def isEqual(labelGT, labelP):
    compare = [1 if int(labelGT[i]) == int(labelP[i]) else 0 for i in range(7)]
    # print(sum(compare))
    return sum(compare)

def eval(model, test_dirs):
    count, error, correct,eval_batchSize = 0, 0, 0,1
    dst = labelTestDataLoader(test_dirs, imgSize)
    testloader = DataLoader(dst, batch_size=1, shuffle=True, num_workers=8)
    start = time()
    eval_text = Variable(torch.IntTensor(eval_batchSize * 5))
    eval_length = Variable(torch.IntTensor(eval_batchSize))
    correct = 0
    for i, (XI, labels, ims) in enumerate(testloader):
        count += 1
        #Changes for the encoder [1234df ] -> [0,1,2,3]
        YI =[] #List of all the licence plate (string of actual licence plates)
        for label in labels:
            indexs =[int(x) for x in label.split('_')[:7]]
            l =[provinces[indexs[0]], alphabets[indexs[1]]]
            for index in range(2,7):
                l.append(ads[indexs[index]])
            YI.append(''.join(l))
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
        fps_pred, preds = model(x)
        
        _, preds = preds.max(2, keepdim=True)
        preds = preds.squeeze(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * eval_batchSize))
        rsim_preds = converter.decode(preds.data, preds_size.data, raw=True)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        #print('rsim_preds: {}, sim_preds {} and YI is {}'.format(str(rsim_preds.encode('utf-8')), sim_preds.encode('utf-8'), YI))
        try:
            print('rsim_preds: {}, sim_preds {} and YI is {}'.format(rsim_preds, sim_preds, YI))
        except Exception as e:
            print('Exception in printing the decoded value: {}'.format(error))
        try:
            for pred, target in zip(sim_preds, YI):
                if pred == target.lower():
                    correct += 1
        except Exception as e:
            print('Exception while calculating correct in Eval')

        #   compare YI, outputY
        # try:
        #     if isEqual(labelPred, YI[0]) == 7:
        #         correct += 1
        #     else:
        #         pass
        # except:
        #     error += 1
    #return count, correct
text = Variable(torch.IntTensor(batchSize * 5))
length = Variable(torch.IntTensor(batchSize))
def train_model(model, criterion, optimizer, num_epochs=25):

    for epoch in range(epoch_start, num_epochs):
        lossAver = []
        model.train(True)
        lrScheduler.step()
        start = time()

        for i, (XI, Y, labels, ims) in enumerate(trainloader):
            if not len(XI) == batchSize:
                continue
            #print('BatchSize var {} and from input dimension {}'.format(batchSize,XI.shape[0]))
            #Changes for the encoder [1234df ] -> [0,1,2,3]
            YI =[] #List of all the licence plate (string of actual licence plates)
            #print('Labels for this batch: {}'.format(labels))
            for label in labels:
                indexs =[int(x) for x in label.split('_')[:7]]
                l =[provinces[indexs[0]], alphabets[indexs[1]]]
                for index in range(2,7):
                    l.append(ads[indexs[index]])
                YI.append(''.join(l))

            #YI = [''.join([str(ee) for ee in el.split('_')[:7]]) for el in labels]

            t, l = converter.encode(YI)
            utils.loadData(text, t)
            utils.loadData(length, l)

            Y = np.array([el.numpy() for el in Y]).T
            if use_gpu:
                x = Variable(XI.cuda())
                y = Variable(torch.FloatTensor(Y).cuda(), requires_grad=False)
            else:
                x = Variable(XI)
                y = Variable(torch.FloatTensor(Y), requires_grad=False)
            # Forward pass: Compute predicted y by passing x to the model
            optimizer.zero_grad()
            #print('X shape: {}'.format(x.shape))
            try:
                fps_pred, y_pred = model(x) #x : (BatchSize,3,480,480)
            except Exception as e:
                print('There was an error for a batch: {}'.format(e))
                continue
            y_pred = y_pred.permute(1,0,2)
            #print('Y output shape to ctc: {}'.format(y_pred.shape))
            #y_pred = #(128,N,nOut)
            #Detection loss
            loss = 0.0
            loss += 0.8 * nn.L1Loss().cuda()(fps_pred[:,:2], y[:,:2])
            loss += 0.2 * nn.L1Loss().cuda()(fps_pred[:,2:], y[:,2:])
            
            y_pred = nn.functional.log_softmax(y_pred,dim=2)
            preds_size = Variable(torch.IntTensor([y_pred.size(0)] * batchSize))
            # Recognition loss- CTC Loss
            loss += criterion(y_pred, text, preds_size, length)

            # Zero gradients, perform a backward pass, and update the weights.
            loss.backward()
            optimizer.step()
            
            lossAver.append(loss.item())
            writer.add_scalar('data/loss',loss.item(),i)
            if (i+1) % 100 == 1:
                with open(args['writeFile'], 'a') as outF:
                    print('Trained %s images, use %s seconds, Average loss %s\n' % ((i+1)*batchSize, time() - start, sum(lossAver) / len(lossAver) if len(lossAver)>0 else 'NoLoss'))
                    outF.write('Trained %s images, use %s seconds, Average loss %s\n' % ((i+1)*batchSize, time() - start, sum(lossAver) / len(lossAver) if len(lossAver)>0 else 'NoLoss'))
                    torch.save(model.state_dict(), storeName + str(epoch)+'_'+str(i)+'.pth')
        with open(args['writeFile'], 'a') as outF:
            print('===========')
            print ('Epoch: %s completed. Avg loss: %s startTime: %s\n' % (epoch, sum(lossAver) / len(lossAver), time()-start))
            print('===========')
            outF.write('==Epoch: %s completed. Avg loss: %s startTime: %s\n==' % (epoch, sum(lossAver) / len(lossAver), time()-start))
            torch.save(model.state_dict(), storeName + str(epoch)+'.pth')
        model.eval()
        eval(model, testDirs)
        #count, correct, error, precision, avgTime = eval(model, testDirs)
        # with open(args['writeFile'], 'a') as outF:
        #     outF.write('%s %s %s\n' % (epoch, sum(lossAver) / len(lossAver), time() - start))
        #     #outF.write('*** total %s error %s precision %s avgTime %s\n' % (count, error, precision, avgTime))
        # if epoch%50 == 0:
        #     torch.save(model.state_dict(), storeName + str(epoch))
    return model


model_conv = train_model(model_conv, criterion.cuda(), optimizer_conv, num_epochs=epochs)
