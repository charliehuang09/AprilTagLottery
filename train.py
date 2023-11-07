from torch.utils.tensorboard import SummaryWriter
import torch
from torchsummary import summary
from torchvision import datasets, transforms
from torch.optim import Adam
import config
from tqdm import trange
from torch import nn
from torch.utils.data import DataLoader
from misc import accuracy, overlay, convert_segmentation
from logger import Logger
from lottery import Lottery
import numpy as np
from model import Unet
from imageLogger import imageLogger
from dataloader import TrainDataset, ValidDataset
from torchvision.utils import draw_segmentation_masks

np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.use_deterministic_algorithms(True)

model = Unet()
summary(model, (3, 960, 540))
writer = SummaryWriter()
# writer = SummaryWriter('./archive/')

valid = ValidDataset()
valid_loader = DataLoader(valid, batch_size=1)
train = TrainDataset()
train_loader = DataLoader(train, batch_size=32, shuffle=True)

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
trainLossLogger = Logger(writer, "train/LossLogger")
validLossLogger = Logger(writer, "valid/LossLogger")
trainLossLogger.setPrefix("Pretrain")
validLossLogger.setPrefix("Pretrain")
trainLossLogger.setWrite(True)
validLossLogger.setWrite(True)

trainPredOverlayLogger = imageLogger(writer, 'train/PredOverlay', 4)
trainLabelOverlayLogger = imageLogger(writer, 'train/LabelOverlay', 4)
trainPredsLogger = imageLogger(writer, 'train/Preds', 4)

validPredOverlayLogger = imageLogger(writer, 'valid/PredOverlay', 4)
validLabelOverlayLogger = imageLogger(writer, 'valid/LabelOverlay', 4)
validPredsLogger = imageLogger(writer, 'valid/Preds', 4)

trainPredOverlayLogger.setPrefix('Pretrain')
trainLabelOverlayLogger.setPrefix('Pretrain')
trainPredsLogger.setPrefix('Pretrain')

validPredOverlayLogger.setPrefix('Pretrain')
validLabelOverlayLogger.setPrefix('Pretrain')
validPredsLogger.setPrefix('Pretrain')


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
        trainPredOverlayLogger.addImage(draw_segmentation_masks(*convert_segmentation(data[0], output[0]), 0.7) / 255)
        trainLabelOverlayLogger.addImage(draw_segmentation_masks(*convert_segmentation(data[0], target[0] / 255), 0.7) / 255)
        trainPredsLogger.addImage(output[0])

    for batch_idx, (data, target) in enumerate(valid_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = loss_fn(output, target)
        validLossLogger.add(loss.item(), len(output))
        validPredOverlayLogger.addImage(draw_segmentation_masks(*convert_segmentation(data[0], output[0]), 0.7) / 255)
        validLabelOverlayLogger.addImage(draw_segmentation_masks(*convert_segmentation(data[0], target[0] / 255), 0.7) / 255)
        validPredsLogger.addImage(output[0])

    
    if epoch == config.warmup_steps:
        opt = Adam(model.parameters(), lr=config.lr)
    
    trainPredOverlayLogger.writeImage()
    trainLabelOverlayLogger.writeImage()
    validPredOverlayLogger.writeImage()
    validLabelOverlayLogger.writeImage()
    trainPredsLogger.writeImage()
    validPredsLogger.writeImage()
    print(f"Train Loss: {trainLossLogger.get()} Test Loss: {validLossLogger.get()} Epoch: {epoch + 1}")

trainLossLogger.clear()
validLossLogger.clear()

trainLossLogger.setPrefix("Prune")
validLossLogger.setPrefix("Prune")

trainPredOverlayLogger.setPrefix('Prune')
trainLabelOverlayLogger.setPrefix('Prune')
trainPredsLogger.setPrefix('Prune')

validPredOverlayLogger.setPrefix('Prune')
validLabelOverlayLogger.setPrefix('Prune')
validPredsLogger.setPrefix('Prune')

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
            trainPredOverlayLogger.addImage(draw_segmentation_masks(*convert_segmentation(data[0], output[0]), 0.7) / 255)
            trainLabelOverlayLogger.addImage(draw_segmentation_masks(*convert_segmentation(data[0], target[0] / 255), 0.7) / 255)
            trainPredsLogger.addImage(output[0])

        for batch_idx, (data, target) in enumerate(valid_loader):
            model = lottery.clampWeights(model)
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_fn(output, target)
            validLossLogger.add(loss.item(), len(output))
            validPredOverlayLogger.addImage(draw_segmentation_masks(*convert_segmentation(data[0], output[0]), 0.7) / 255)
            validLabelOverlayLogger.addImage(draw_segmentation_masks(*convert_segmentation(data[0], target[0] / 255), 0.7) / 255)
            validPredsLogger.addImage(output[0])

        print(f"Train Loss: {trainLossLogger.get()} Test Loss: {validLossLogger.get()} Epoch: {epoch + 1}")
    lottery.updateMask(model)
    model = lottery.applyMask(model)

trainLossLogger.clear()
validLossLogger.clear()

trainLossLogger.setPrefix("Final")
validLossLogger.setPrefix("Final")

trainPredOverlayLogger.setPrefix('Final')
trainLabelOverlayLogger.setPrefix('Final')
trainPredsLogger.setPrefix('Final')

validPredOverlayLogger.setPrefix('Final')
validLabelOverlayLogger.setPrefix('Final')
validPredsLogger.setPrefix('Final')

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
        trainPredOverlayLogger.addImage(draw_segmentation_masks(*convert_segmentation(data[0], output[0]), 0.7) / 255)
        trainLabelOverlayLogger.addImage(draw_segmentation_masks(*convert_segmentation(data[0], target[0] / 255), 0.7) / 255)
        trainPredsLogger.addImage(output[0])

    for batch_idx, (data, target) in enumerate(valid_loader):
        model = lottery.clampWeights(model)
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = loss_fn(output, target)
        validLossLogger.add(loss.item(), len(output))
        validPredOverlayLogger.addImage(draw_segmentation_masks(*convert_segmentation(data[0], output[0]), 0.7) / 255)
        validLabelOverlayLogger.addImage(draw_segmentation_masks(*convert_segmentation(data[0], target[0] / 255), 0.7) / 255)
        validPredsLogger.addImage(output[0])
    print(f"Train Loss: {trainLossLogger.get()} Test Loss: {validLossLogger.get()} Epoch: {epoch + 1}")


print("FINAL")
print(f"Train Loss: {trainLossLogger.get()} Test Loss: {validLossLogger.get()} Epoch: {epoch + 1}")

writer.add_scalar("Final trainLossLogger", trainLossLogger.getMin(), 0) 
writer.add_scalar("Final testLossLogger", validLossLogger.getMin(), 0)

print(lottery.getMask())