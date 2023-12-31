import torch
import torch.nn as nn
from torchvision.transforms import CenterCrop

class Unet(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        #first block
        self.conv1 = nn.Conv2d(3, channels * 1, (3, 3))
        self.conv2 = nn.Conv2d(channels * 1, channels * 1, (3, 3))
        #first max pool
        self.maxpool3 = nn.MaxPool2d((2,2))

        #second block
        self.conv4 = nn.Conv2d(channels * 1, channels * 2, (3,3))
        self.conv5 = nn.Conv2d(channels * 2, channels * 2, (3,3))
        #second max pool
        self.maxpool6 = nn.MaxPool2d((2,2))

        #third block
        self.conv7 = nn.Conv2d(channels * 2, channels * 4, (3,3))
        self.conv8 = nn.Conv2d(channels * 4, channels * 4, (3,3))
        #third max pool
        self.maxpool9 = nn.MaxPool2d((2,2))

        #forth block
        self.conv10 = nn.Conv2d(channels * 4, channels * 8, (3, 3))
        self.conv11 = nn.Conv2d(channels * 8, channels * 8, (3,3))
        #forth max pool
        self.maxpool12 = nn.MaxPool2d((2,2))

        #-------------------------------------------

        self.conv13 = nn.Conv2d(channels * 4, channels * 8, (3,3))
        self.conv14 = nn.Conv2d(channels * 8, channels * 8, (3,3))
        
        #-------------------------------------------

        #first block
        self.up15 = nn.ConvTranspose2d(channels * 16, channels * 8, (2,2), 2)
        #cat
        self.conv16 = nn.Conv2d(channels * 16, channels * 8, (3,3))
        self.conv17 = nn.Conv2d(channels * 8, channels * 8, (3,3))

        #second block
        self.up18 = nn.ConvTranspose2d(channels * 8, channels * 4, (2,2), 2)
        #cat
        self.conv19 = nn.Conv2d(channels * 8, channels * 4, (3,3))
        self.conv20 = nn.Conv2d(channels * 4, channels * 4, (3,3))

        #third block
        self.up21 = nn.ConvTranspose2d(channels * 4, channels * 2, (2,2), 2)
        #cat
        self.conv22 = nn.Conv2d(channels * 4, channels * 2, (3,3))
        self.conv23 = nn.Conv2d(channels * 2, channels * 2, (3,3))

        #forth block
        self.up24 = nn.ConvTranspose2d(channels * 2, channels * 1, (2,2), 2)
        #cat
        self.conv25 = nn.Conv2d(channels * 2, channels * 1, (3,3))
        self.conv26 = nn.Conv2d(channels * 1, channels * 1, (3,3))
        self.conv27 = nn.Conv2d(channels * 1, 1, (1,1))
    
    def forward(self, x):
        #first block
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x1 = x
        x = self.maxpool3(x)

        #second block
        x = self.conv4(x)
        x = self.activation(x)
        x = self.conv5(x)
        x = self.activation(x)
        x2 = x
        x = self.maxpool6(x)

        #third block
        x = self.conv7(x)
        x = self.activation(x)
        x = self.conv8(x)
        x = self.activation(x)
        x3 = x
        x = self.maxpool9(x)

        #forth block
        # x = self.conv10(x)
        # x = self.activation(x)
        # x = self.conv11(x)
        # x = self.activation(x)
        # x4 = x
        # x = self.maxpool12(x)

        #-------------------------------------------
        x = self.conv13(x)
        x = self.activation(x)
        x = self.conv14(x)
        x = self.activation(x)

        #-------------------------------------------

        #first block
        # x = self.up15(x)
        # crop = CenterCrop((x.shape[2], x.shape[3]))
        # x4 = crop(x4)
        # x = torch.cat([x4, x], dim=1)
        # x = self.conv16(x)
        # x = self.activation(x)
        # x = self.conv17(x)
        # x = self.activation(x)

        #second block
        x = self.up18(x)
        crop = CenterCrop((x.size()[2], x.size()[3]))
        x3 = crop(x3)
        x = torch.cat([x3, x], dim=1) 
        x = self.conv19(x)
        x = self.activation(x)
        x = self.conv20(x)
        x = self.activation(x)

        #third block
        x = self.up21(x)
        crop = CenterCrop((x.size()[2], x.size()[3]))
        x2 = crop(x2)
        x = torch.cat([x2, x], dim=1)
        x = self.conv22(x)
        x = self.activation(x)
        x = self.conv23(x)
        x = self.activation(x)
        
        #forth block
        x = self.up24(x)
        crop = CenterCrop((x.size()[2], x.size()[3]))
        x1 = crop(x1)
        x = torch.cat([x1, x], dim=1)
        x = self.conv25(x)
        x = self.activation(x)
        x = self.conv26(x)
        self.activation(x)
        x = self.conv27(x)
        x = self.sigmoid(x)
        # x = self.activation(x)

        return x

