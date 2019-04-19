from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
import numpy as np
import cv2
from model import fh02
import utils
import torch
import torch.nn as nn
from torch.autograd import Variable

#Make sure of gpu/cpu compatibility 
device  = torch.device('cpu')
if torch.cuda.is_available():
    device  = torch.device('cuda')

#Write code to make sure all the requisite file is present.
resume_file = './weights_4.pth'
    
#Initialize the Tkinder GUI
top = Tk()
top.title = 'GUI'
top.geometry('720x480')

#Initialize the Canvas
canvas_width = 480
canvas_height = 480
canvas = Canvas(top, width=canvas_width,height=canvas_height, bd=0,bg='white')
canvas.grid(row=1, column=0)

#Initialize the Model
model = fh02()
model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
model.load_state_dict(torch.load(resume_file, map_location=device))
model.to(device)
model.eval()

#The utility function for encode and decode
converter = utils.strLabelConverter()

def showImg():
    File = askopenfilename(title='Upload Image')
    global cvimage
    cvimage = cv2.imread(File)
    #print('Type cvimage :{} and cvimage shape: {}'.format(type(cvimage), cvimage.shape))
    image = cv2.cvtColor(cvimage, cv2.COLOR_BGR2RGB)
    h,w,c = image.shape
    if h > canvas_height or w > canvas_width:
        image  = cv2.resize(image,(canvas_height,canvas_width))
    
    image = Image.fromarray(image)
    #image = Image.open(File)
    imgfile = ImageTk.PhotoImage(image)
    canvas.image = imgfile  # <--- keep reference of image
    canvas.create_image(2,2,anchor='nw',image=imgfile)

#e = StringVar()

open_button = Button(top, text ='Upload', command = showImg)
#submit_button.grid(row=0,column=1)
open_button.place(relx=0.83, rely=0.1, anchor=N)

#Label for Upper left coordinate
label = Label(top, text='Upper_Left')
label.place(relx=0.73, rely=0.3, anchor=N)

#entry boxes for Upper left coordinate
ulx = Entry (top, width=5)
ulx.place(relx=0.84, rely=0.3, anchor=N)
uly = Entry (top, width=5)
uly.place(relx=0.93, rely=0.3, anchor=N)


#Label for Bottom Right coordinate
label = Label(top, text='Bottom_Right')
label.place(relx=0.73, rely=0.4, anchor=N)

#entry boxes for Bottom Right coordinate
brx = Entry (top, width=5)
brx.place(relx=0.84, rely=0.4, anchor=N)
bry = Entry (top, width=5)
bry.place(relx=0.93, rely=0.4, anchor=N)

label = Label(top, text='License Number')
label.place(relx=0.74, rely=0.5, anchor=N)

lp = Entry (top, width=10)
lp.place(relx=0.92, rely=0.5, anchor=N)

#Draw Functionality===============
def draw():
    #print(int(ulx.get()),int(uly.get()),int(brx.get()),int(bry.get()))
    canvas.create_rectangle(int(ulx.get()),int(uly.get()),int(brx.get()),int(bry.get()),  width= 2, outline="#fb0", tag='rec')
    canvas.create_text(int(ulx.get())+28,int(uly.get())-10,text=lp.get(), tag='text', font=("Purisa", 14))

predict_button = Button(top, text ='Draw', command = draw)
predict_button.place(relx=0.75, rely=0.65, anchor=N)

#Clear Functionality===============
def clear_rec():

    canvas.delete('rec')
    canvas.delete('text')

predict_button = Button(top, text ='Clear', command = clear_rec)
predict_button.place(relx=0.90, rely=0.65, anchor=N)


#Inference=================
'''FLow: get the canvas.image reference-> get numpy from that-> 
then get tensor object -> call a function to return the coordinates and license number 
'''
def predict():
    #call function with argument: input: cvimage -> output of cv2.imread ; output: four coordinates of bounding box and licence number
    #Prepare Input in form of required shape and convert to tensor
    img = cv2.resize(cvimage, (480,480))

    resizedImage = np.transpose(img, (2,0,1))
    resizedImage = resizedImage.astype('float32')
    resizedImage /= 255.0
    input_tensor = torch.from_numpy(resizedImage)
    c,h,w = input_tensor.shape

    batch_input = input_tensor.view(1,c,h,w)
    batch_input.to(device)

    XI = Variable(batch_input)

    fps_pred, preds = model(XI) #call to model
    
    #CTC Decoding
    _, preds = preds.max(2, keepdim=True)
    preds = preds.squeeze(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * 1))
    sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
    #print(sim_preds)
    [cx, cy, w, h] = fps_pred.data.cpu().numpy()[0].tolist()
    lx,ly = ((cx - w/2)*img.shape[1], (cy - h/2)*img.shape[0])
    rx,ry = ((cx + w/2)*img.shape[1], (cy + h/2)*img.shape[0])
    text = sim_preds.upper()

    canvas.create_rectangle(int(lx),int(ly),int(rx),int(ry),  width= 2, outline="#fb0", tag='rec')
    canvas.create_text(int(lx)+80,int(ly)-10,text=text, font=("Purisa", 20), fill="red", tag='text')

predict_button = Button(top, text ='Predict', command=predict)
predict_button.place(relx=0.83, rely=0.80, anchor=N) 

top.mainloop()