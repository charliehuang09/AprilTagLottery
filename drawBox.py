import config
import numpy as np
import torch
dx = [0, 0, 1, -1]
dy = [1, -1, 0, 0]
def BFS(image, x, y, threshold=config.threshold):
    size = 0
    xq = []
    yq = []
    xlimit = image.shape[0]
    ylimit = image.shape[1]
    xmin = xlimit
    ymin = ylimit
    xmax = 0
    ymax = 0
    xq.append(x)
    yq.append(y)
    while(len(xq) > 0):
        currx = xq[0]
        curry = yq[0]
        xq.pop(0)
        yq.pop(0)
        for i in range(len(dx)):
                if currx + dx[i] >= 0 and curry + dy[i] >= 0 and currx + dx[i] < xlimit and curry + dy[i] < ylimit:
                    if image[currx + dx[i]][curry + dy[i]] > threshold:
                        size += 1
                        xmin = min(xmin, currx + dx[i])
                        ymin = min(ymin, curry + dy[i])
                        xmax = max(xmax, currx + dx[i])
                        ymax = max(ymax, curry + dy[i])
                        image[currx + dx[i]][curry + dy[i]] = 0
                        xq.append(currx + dx[i])
                        yq.append(curry + dy[i])

    return image, size, torch.tensor([xmin, ymin, xmax, ymax])



def drawBox(image, threshold=config.threshold):
    output_corners = torch.tensor([0,0,0,0])
    output_size = -1
    for i in range(len(image)):
        for j in range(len(image[i])):
            if image[i][j] > threshold:
                image, size, corners = BFS(image, i, j, threshold)
                if output_size < size:
                    output_size = size
                    output_corners = corners

    return image, output_size, corners


