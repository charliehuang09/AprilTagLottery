import torch
import os
import cv2
import warnings
import numpy as np
from misc import softmax
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, PATH='Train'):
        self.imgs = []
        for filename in os.listdir(PATH):
            path = os.path.join(PATH, filename)
            cap = cv2.VideoCapture(path)
            ret = True
            while ret:
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.imgs.append(frame)
        self.imgs = np.array(self.imgs)

    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, index):
        img = self.imgs[index]
        (corners, ids, rejected) = cv2.aruco.detectMarkers(img, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000))
        label = np.zeros(shape=(img.shape[0], img.shape[1]))
        if ids == [0]:
            corners = corners[0].astype(np.int32)
            label = cv2.fillPoly(label, pts = corners, color =(255,255,255))
        label = cv2.resize(label, (287, 127))
        img = cv2.resize(img, (960, 540))

        img = img.astype(np.float32)
        label = label.astype(np.float32)

        img = img.swapaxes(0, 2)
        img = img.swapaxes(1, 2)
        
        label = np.array([label])

        return img, label


class ValidDataset(Dataset):
    def __init__(self, PATH='Valid'):
        self.x = []
        self.y = []
        for filename in os.listdir(PATH):
            img = cv2.imread(os.path.join(PATH, filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = img[:1080*4, :1920*4]
            (corners, ids, rejected) = cv2.aruco.detectMarkers(img, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000))
            label = np.zeros(shape=(img.shape[0], img.shape[1]))
            if ids == [0]:
                corners = corners[0].astype(np.int32)
                label = cv2.fillPoly(label, pts = corners, color =(255,255,255))
            img = cv2.resize(img, (960, 540))
            img = img.astype(np.float32)
            img = img.swapaxes(0, 2)
            img = img.swapaxes(1, 2)

            label = cv2.resize(label, (287, 127))
            label = softmax(label)
            label = np.array([label])

            label = torch.tensor(label, dtype=torch.float32)
            img = torch.tensor(img, dtype=torch.float32)

            self.x.append(img)
            self.y.append(label)

    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        return self.x[index], self.y[index]

        