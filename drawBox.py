import config
import numpy as np
import torch
dx = [0, 0, 1, -1]
dy = [1, -1, 0, 0]
def BFS(image, x, y):
    size = 0
    xq = []
    yq = []
    xmin = x
    ymin = y
    xmax = x
    ymax = y
    xq.append(x)
    yq.append(y)
    while(len(xq) > 0):
        currx = xq[i]
        curry = yq[i]
        xq.pop(0)
        yq.pop(0)
        for i in range(len(dx)):
            if image[currx + dx[i]][curry + dy[i]] > config.threshold:
                size += 1
                xmin = min(xmin, currx + dx[i])
                ymin = min(ymin, curry + dy[i])
                xmax = max(xmax, currx + dx[i])
                ymax = max(ymax, curry + dy[i])
                image[currx + dx[i]][curry + dy[i]] = 0

    return image, size, torch.tensor([xmin, ymin, xmax, ymax])



def drawBox(image):
    output_corners = torch.tensor([0,0,0,0])
    output_size = -1
    for i in range(len(input)):
        for j in range(len(input[i])):
            if image[i][j] > config.threshold:
                image, size, corners = BFS(image, i, j)
                if output_size < size:
                    output_size = size
                    output_corners = corners

    return image, output_size, corners


