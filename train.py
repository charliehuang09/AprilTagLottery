from torch.utils.tensorboard import SummaryWriter
import torch
from torchsummary import summary
from torchvision import datasets, transforms
from torch.optim import Adam
import config
from tqdm import trange
from torch import nn
from torch.utils.data import DataLoader
from misc import accuracy, overlay
from logger import Logger
from lottery import Lottery
import numpy as np
from model import Unet
from imageLogger import imageLogger
from dataloader import TrainDataset, ValidDataset

np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.use_deterministic_algorithms(True)

model = Unet()
summary(model, (3, 960, 540))
writer = SummaryWriter()
# writer = SummaryWriter('archive')

valid = ValidDataset()
valid_loader = DataLoader(valid, batch_size=1)
train = TrainDataset()
train_loader = DataLoader(train, batch_size=32)

device = torch.device('mps')
opt = Adam(model.parameters(), lr=config.warmup_lr)
loss_fn = nn.MSELoss()

hparams = {
    "lr":config.lr,
    "Epoch": config.epoch,
    "Pretrain Epochs": config.pretrain_epoch,
    "Warmup Steps": config.warmup_steps,
    "Warmup Lr": config.warmup_lr,
    "Purne Percent": config.prune_percent,
    "Prune Iterations": config.iterations,
    "Iteration Epochs": config.iteration_epoch,
    "Seed": config.seed,
}
# writer.add_hparams(hparams, {})
trainLossLogger = Logger(writer, "trainLossLogger")
testLossLogger = Logger(writer, "testLossLogger")
trainLossLogger.setPrefix("Pretrain")
testLossLogger.setPrefix("Pretrain")
trainLossLogger.setWrite(True)
testLossLogger.setWrite(True)

trainXImageLogger = imageLogger(writer, 'trainXImage', 4)
trainYImageLogger = imageLogger(writer, 'trainYImage', 4)

validXImageLogger = imageLogger(writer, 'validXImage', 4)
validYImageLogger = imageLogger(writer, 'validYImage', 4)

model = model.to(device)
model.train()

for epoch in range(config.pretrain_epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        opt.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        opt.step()
        trainLossLogger.add(loss.item(), len(output))
        trainXImageLogger.addImage(data[0] / 255)
        trainYImageLogger.addImage(output[0])

    for batch_idx, (data, target) in enumerate(valid_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = loss_fn(output, target)
        testLossLogger.add(loss.item(), len(output))
        validXImageLogger.addImage(data[0] / 255)
        validYImageLogger.addImage(output[0])

    
    if epoch == config.warmup_steps:
        opt = Adam(model.parameters(), lr=config.lr)
    
    trainXImageLogger.writeImage()
    trainYImageLogger.writeImage()
    validXImageLogger.writeImage()
    validYImageLogger.writeImage()
    print(f"Train Loss: {trainLossLogger.get()} Test Loss: {testLossLogger.get()} Epoch: {epoch + 1}")
exit(0)

lottery = Lottery(model, config.prune_percent, config.iterations)

for iteration in range(0, config.iterations):
    print(f"Iteration {iteration + 1} ----------------------------------------------------")
    opt = Adam(model.parameters(), lr=config.lr)
    for epoch in range(config.iteration_epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            model = lottery.clampWeights(model)
            data, target = data.to(device), target.to(device)
            opt.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            opt.step()
            trainLossLogger.add(loss.item(), len(output))
        for batch_idx, (data, target) in enumerate(valid_loader):
            model = lottery.clampWeights(model)
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_fn(output, target)
            testLossLogger.add(loss.item(), len(output))
            testAccuracyLogger.add(accuracy(output, target), 1)

        print(f"Train Loss: {trainLossLogger.get()} Test Loss: {testLossLogger.get()} Epoch: {epoch + 1}")
    lottery.updateMask(model)
    model = lottery.applyMask(model)

trainLossLogger.clear()
testLossLogger.clear()

trainLossLogger.setWrite(True)
testLossLogger.setWrite(True)
opt = Adam(model.parameters(), lr=config.lr)
print("FINAL-Training----------------------------------------------------------------")
for epoch in range(config.epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
        model = lottery.clampWeights(model)
        data, target = data.to(device), target.to(device)
        opt.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        opt.step()
        trainLossLogger.add(loss.item(), len(output))
    # for batch_idx, (data, target) in enumerate(valid_loader):
    #     model = lottery.clampWeights(model)
    #     data, target = data.to(device), target.to(device)
    #     output = model(data)
    #     loss = loss_fn(output, target)
    #     testLossLogger.add(loss.item(), len(output))
    #     testAccuracyLogger.add(accuracy(output, target), 1)
    print(f"Train Loss: {trainLossLogger.get()} Test Loss: {testLossLogger.get()} Epoch: {epoch + 1}")


print("FINAL")
print(f"Train Loss: {trainLossLogger.get()} Test Loss: {testLossLogger.get()} Epoch: {epoch + 1}")

writer.add_scalar("Final trainLossLogger", trainLossLogger.getMin(), 0) 
writer.add_scalar("Final testLossLogger", testLossLogger.getMin(), 0)

print(lottery.getMask())