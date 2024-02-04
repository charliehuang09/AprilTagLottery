from torchvision.utils import make_grid
import torch
import numpy as np

class imageLogger:
    def __init__(self, writer, name, max_size):
        self.writer = writer
        self.name = name
        self.max_size = max_size
        self.imgs = []
        self.idx = 1
        self.prefix = ""
    
    def addImage(self, img):
        if len(self.imgs) < self.max_size:
            self.imgs.append(img.detach().cpu().numpy())
        return
    
    def writeImage(self):
        imgs = np.array(self.imgs)
        imgs = torch.Tensor(imgs)
        grid = make_grid(imgs)
        self.writer.add_image(self.prefix + '/' + self.name, grid, self.idx)
        self.idx += 1
        self.imgs = []
        return
    
    def setPrefix(self, prefix):
        self.prefix = prefix
        return
