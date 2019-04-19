# Licence Plate Localization and Recognition 

In this project, I tried to solve the problem of automatic detection of licence plate from given image and recognizing the licence number. The dataset used is CCPD (Chinese Car Parking Dataset). 


## Prerequisite
*	Python 3.6
*	Pytorch v1.0
*	Numpy, imutils, opencv2
*	Tkinter (To run GUI)
*	TensorboardX

## Structure
This repository is divided into two parts. First one is  *Train_Test* and another folder is *GUI* . Both of them can run independently. The *Train_Test* folder contains the code to train and test the network. The *GUI* folder is use to run the Tkinter based GUI which can be used to evaluate the result or demo purpose. 

## Dataset
The CCPD dataset can be downloaded from [here](https://drive.google.com/file/d/1fFqCXjhk7vE9yLklpJurEwP9vdLZmrJd/view).

## Train
Make sure you are in *Train_Test* folder. First we have to train the detection module for a certain epochs (~100) so that it can converge. After that both modules can be trained end-to-end.
Train Detection Module:
```
python wR2.py -i </path/to/trainImages> -b <Batch_size> 
```
The weights will be stored in a folder *wR2* which will be created in same directory as this file is present.  Once you get the weights, use it to load detection module in whole network and train end-to-end by running the following command:
```
python rpnet.py -i </path/to/trainImages> -t <path/to/valImages> -b <Batch_size>-f <Directory/to/store/weights> -w <path/to/log_file> -dw <path/to/detection_Module_Weights> 
```
## Test
The project uses different file to evaluate the network. Once you have the pretrained weights of the whole network. Use the following command to evaluate results in test dataset.
```
python rpnetEval.py -i </path/to/test_Images> -m <path/to/trained_weights>
```
You can see the logs and evaluation results in eval.out file which will be created as part of above code.

## Run GUI
Once you have pretrained weights. Transfer it to GUI folder. Either rename the *.pth file to weights_4.pth or change in GUI.py file accordingly.
```
python GUI.py
```
## Demo

Hit [this](https://www.youtube.com/watch?v=Jc53MyVG9Q0) to see the demo.
