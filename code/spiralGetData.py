# %%
# Create spiral matrix

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def create_spiral(n):
    # Initialize a n x n matrix
    matrix = [[0] * n for _ in range(n)]

    x, y = 0, 0

    # Direction vectors (right, down, left, up)
    dx = [0, 1, 0, -1]
    dy = [1, 0, -1, 0]
    direction = 0

    for i in range(n * n - 1, -1, -1):  # Start from n*n - 1 (35 for 6x6) and go down to 0
        matrix[x][y] = i

        nx = x + dx[direction]
        ny = y + dy[direction]

        # Change direction if next position is out of bounds or already filled
        if nx < 0 or nx >= n or ny < 0 or ny >= n or matrix[nx][ny] != 0:
            direction = (direction + 1) % 4  # Change direction
            nx = x + dx[direction]
            ny = y + dy[direction]

        x, y = nx, ny

    return matrix
