import torch
import os
import cv2
import warnings
import numpy as np
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, length, PATH='/Users/charlie/Documents/ML/AprilTagImitation/Train'):
        print("loading dataset...")
        self.length = length
        self.imgs = []
        for filename in os.listdir(PATH):
            path = os.path.join(PATH, filename)
            cap = cv2.VideoCapture(path)
            ret = True
            while ret:
                ret, frame = cap.read()
                if not ret:
                    continue
                self.imgs.append(frame)
                # break #testing only
        self.imgs = np.array(self.imgs)
        print("finished loading dataset")

    def __len__(self):
        return self.length
    def __getitem__(self, index):
        index = np.random.randint(0, len(self.imgs))
        img = self.imgs[index]
        (corners, ids, rejected) = cv2.aruco.detectMarkers(img, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000))
        label = np.zeros(shape=(img.shape[0], img.shape[1]))
        print(ids)
        if ids == [0]:
            corners = corners[0].astype(np.int32)
            label = cv2.fillPoly(label, pts = corners, color =(255,255,255))
        label = cv2.resize(label, (671, 351))

        img = img.astype(np.float32)
        label = label.astype(np.float32)

        img = img.swapaxes(0, 2)
        img = img.swapaxes(1, 2)
        
        label = np.array([label])

        return img, label

