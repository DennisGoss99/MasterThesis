import datetime
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset, random_split
from torch.nn import functional as F
from tqdm import tqdm
import os
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import argparse
import numpy as np
import time
import traceback

import sys
module_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(module_dir)

from dataSetCombiner import getDataSet
from modelHelper import count_parameters, parseParameter, model_saveFile, logger_write_parameter, setup_logger
from spiralGetData import create_spiral

torch.manual_seed(1337)

LEARNING_RATE = 3e-5

BATCH_SIZE = 8

IMAGE_SIZE = 64
BLOCK_SIZE = IMAGE_SIZE * IMAGE_SIZE - 1

CHANNELS_IMG = 3

N_EMBD = 128
N_HEAD = 6
N_LAYER = 6
DROPOUT = 0.2

VERSION = "5.0.1.0_newData_128"

#----------------------------------------------

spiral_indices = torch.tensor(create_spiral(IMAGE_SIZE))

@torch.no_grad()
def convertBackToImg(idx):
    positions_in_spiral = torch.argsort(spiral_indices.flatten())
    reconstructed_tensor = torch.zeros((3,IMAGE_SIZE*IMAGE_SIZE), device=device)
    idx = torch.cat((idx, torch.zeros((1,3), device=device)), dim=0)
    reconstructed_tensor[:,positions_in_spiral] = rearrange(idx, 'h c -> c h')
    reconstructed_tensor = reconstructed_tensor.view(3,IMAGE_SIZE, IMAGE_SIZE)
    return rearrange(reconstructed_tensor, 'c h w -> 1 c h w')

@torch.no_grad()
def get_batch(data):
    B, C ,H ,W = data.shape

    spiral_data = torch.zeros_like(data.view(B, C, -1)).to(device)

    spiral_data[:,:,spiral_indices.flatten()] = data.view(B, C, -1)

    x = spiral_data[:, :, :BLOCK_SIZE]
    y = spiral_data[:, :, 1:BLOCK_SIZE+1]

    x, y = x.to(device), y.to(device)
    x = rearrange(x, 'b c h -> b h c')
    y = rearrange(y, 'b c h -> b h c')
    return x, y


@torch.no_grad()
def validate(model, dataloader):
    total_loss = 0
    total_samples = 0
    for idx, (data, _) in enumerate(dataloader):
        dataRaw = data.to(device)
        x, y = get_batch(dataRaw)
        
        _, loss = model(x, y)

        total_loss += loss.item()
        total_samples += 1
    return total_loss / total_samples

#----------------------------------------------

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
        
class ColumnTransformer(nn.Module):

    def __init__(self):
        super().__init__()
        self.pixel_embedding = nn.Sequential(
            nn.Linear(CHANNELS_IMG, N_EMBD//5, device = device),
            nn.ReLU(),
            nn.Linear(N_EMBD//5, N_EMBD//2, device = device),
            nn.ReLU(),
            nn.Linear(N_EMBD//2, N_EMBD, device = device),
            nn.Dropout(DROPOUT),
        )
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[Block(N_EMBD, n_head=N_HEAD) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD) # final layer norm
        self.lm_head = nn.Sequential(
            nn.Linear(N_EMBD, N_EMBD//2, device=device),
            nn.ReLU(),
            nn.Linear(N_EMBD//2, N_EMBD//5, device=device),
            nn.ReLU(),
            nn.Linear(N_EMBD//5, CHANNELS_IMG, device=device),
            nn.Sigmoid(),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx, targets=None):
        B, H, C = idx.shape
        
        tok_emb = self.pixel_embedding(idx) # (B, H, N_EMBD)
        pos_emb = self.position_embedding_table(torch.arange(H, device=device)) # (H, N_EMBD)

        x = tok_emb + pos_emb # (B,H,N_EMBD)


        x = self.blocks(x) # (B,H,N_EMBD)
        x = self.ln_f(x) # (B,H,N_EMBD)
        logits = self.lm_head(x) # (B,H,C)

        if targets is None:
            loss = None
        else:
            # B, T, C = logits.shape
            loss = F.mse_loss(logits, targets)
            
        return logits, loss

def main():

    args = parseParameter()
    
    outputdir = args.out

    if not os.path.exists(outputdir +'/tempModel/'):
        os.makedirs(outputdir + '/tempModel/')

    logger = setup_logger(outputdir, VERSION)

    dataset = getDataSet(args.path, args.dataset, IMAGE_SIZE, IMAGE_SIZE, repeatData=1,
               random_vertical_flip=False, random_horizontal_flip=False,
               crop_type='random', grayscale=False, color_jitter=False, jitter_brightness=0,
               jitter_contrast=0, jitter_saturation=0, jitter_hue=0)

    total_size = len(dataset)

    if args.valsize.endswith('%'):
        val_size = int(total_size * (int(args.valsize[:-1]) / 100))
    else: 
        val_size = int(args.valsize)

    train_size = total_size - val_size

    if args.trainvalsize.endswith('%'):
        train_val_size = int(train_size * (int(args.trainvalsize[:-1]) / 100))
    else: 
        train_val_size = int(args.valsize)


    train_size = len(dataset) - val_size  
    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_val_indices = np.random.choice(len(train_data), train_val_size, replace=False)
    train_val_data = Subset(train_data, train_val_indices)

    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    train_val_loader = DataLoader(train_val_data, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=True)

    m = ColumnTransformer()
    m = m.to(device)

    optimizer = optim.Adam(m.parameters(), lr=LEARNING_RATE)

    iter_i = args.iter
    eval_i = 2
    eval_img = 2

    writer = SummaryWriter(f"{outputdir}/tempLog/{VERSION}-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}/")



    logger_write_parameter(
        logger=logger,
        device=device, 
        dataset=args.dataset,
        train_size=train_size,
        valsize=val_size,
        trainvalsize=train_val_size,
        repeatdataset=args.repeatdataset,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        block_size=BLOCK_SIZE,
        channels_img=CHANNELS_IMG,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        dropout=DROPOUT,
        version=VERSION,
        model_parameter=count_parameters(m)
    )

    start_time = time.time()

    for e in range(0,iter_i):
        loss_sum = 0.0
        for idx, (data, _) in enumerate(train_loader):

            dataRaw = data.to(device)
            x, y = get_batch(dataRaw) # (B,C,H)
        
            logits, loss = m(x, y)
            
            loss_sum += loss.item()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            

            if idx % eval_img == 0 and idx != 0:
                combined_images = torch.cat([convertBackToImg(y[0]), convertBackToImg(logits[0])], dim=3)
                image_grid = vutils.make_grid(combined_images)
                writer.add_image('Generated vs Original', image_grid, e* len(train_loader)// eval_img + idx // eval_img)

            if(idx % eval_i == 0 and idx != 0):
                train_loss = loss_sum / eval_i
                val_loss = 0.0
                val_loss = validate(m, val_loader)
                train_val_loss = validate(m, train_val_loader)
                logger.debug(f'Epoch {e}, Iteration {idx}: Train Loss: {train_loss}, Train_val Loss: {train_val_loss}, Validation Loss: {val_loss}, {int((time.time() - start_time) // 3600):02d}:{int(((time.time() - start_time) % 3600) // 60):02d}:{int((time.time() - start_time) % 60):02d}')
                torch.save(m, model_saveFile(outputdir, VERSION, e))
                loss_sum = 0.0
                writer.add_scalars('Losses', {'Training Loss': train_val_loss, 'Validation Loss': val_loss}, e * len(train_loader) // eval_i + idx // eval_i)
    
    writer.close()

    logger.info('---end---')

    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        traceback.print_exc()