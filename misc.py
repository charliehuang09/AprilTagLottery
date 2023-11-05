import cv2
import numpy as np
import torch
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
