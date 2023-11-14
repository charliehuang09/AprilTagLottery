import cv2
import numpy as np
import torch
from torchvision.transforms import Resize
from torchvision.utils import make_grid

def overlay(x, outputs):
    x = make_grid(x)
    outputs = make_grid(outputs)
    x = x.cpu().numpy()
    outputs = outputs.cpu().numpy()

    outputs = outputs.swapaxes(0, 2)
    # outputs = outputs.swapaxes(0, 1)

    outputs = cv2.resize(outputs, (x.shape[1], x.shape[2]))

    outputs = outputs.swapaxes(0, 2)
    # outputs = outputs.swapaxes(1, 2)
    # exit(0)

    x /= 255
    x += outputs

    x = np.fmin(1, x)
    
    return x
  
def accuracy(output, target):
  output = output.argmax(1)
  train_acc = torch.sum(output == target)
  output =  train_acc / len(output)
  return output
  
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def convert_binary(input):
   if input > 0.5:
      return True
   return False

def convert_segmentation(data, mask):
   data = data.to(torch.uint8)
   data = data.cpu()
   mask = mask.detach().cpu()
   mask.apply_(convert_binary)
   mask = mask.to(torch.bool)

   resize = Resize((540,960), antialias=True)
   mask = resize(mask)
   
   return data, mask 

def convert_plt(image):
   image = np.array(image)
   image = np.swapaxes(image,0, 2)
   image = np.swapaxes(image, 0, 1)

def convert_torch(image):
   image = np.array(image)
   image = np.swapaxes(image,0, 2)
   image = np.swapaxes(image, 1, 2)