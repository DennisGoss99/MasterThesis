import datetime
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
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


torch.manual_seed(1337)

LEARNING_RATE = 3e-5

BATCH_SIZE = 64

BLOCK_SIZE = 256 
IMAGE_SIZE = 257 # Must be at least >= Blocksize +1

CHANNELS_IMG = 3

N_EMBD = 512
N_HEAD = 8
N_LAYER = 8
DROPOUT = 0.2

VERSION = "2.1.8.1_newData_256"

#----------------------------------------------

@torch.no_grad()
def get_batch(data):
    # C ,H ,W = data.shape

    x = data[:, :BLOCK_SIZE, :BATCH_SIZE]
    y = data[:, 1:BLOCK_SIZE+1, :BATCH_SIZE]

    x, y = x.to(device), y.to(device)
    x = rearrange(x, 'c h b -> b h c')
    y = rearrange(y, 'c h b -> b h c')
    return x, y

@torch.no_grad()
def validate(model, dataloader):
    total_loss = 0
    total_samples = 0
    for idx, (data, _) in enumerate(dataloader):
        dataRaw = data.squeeze(0).to(device)
        x, y = get_batch(dataRaw)
        y = y[:,:BLOCK_SIZE,:]
        
        _, loss = model(x, y)

        total_loss += loss.item()
        total_samples += 1
    return total_loss / total_samples

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["Master_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(
        backend='nccl',  # 'nccl' is recommended for GPU use, 'gloo' can be used for CPU
        rank=rank,
        world_size=world_size
    )


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


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
        outputdir: str,
        logger = None,
        val_data = None,
        train_val_data = None,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.model = DDP(model, device_ids=[gpu_id])
        self.outputdir = outputdir
        self.logger = logger

        self.eval_i = save_every
        self.start_time = time.time()

        if(self.gpu_id == 0):
            self.val_loader = DataLoader(val_data, batch_size=1, shuffle=True)
            self.train_val_loader = DataLoader(train_val_data, batch_size=1, shuffle=True)

            self.writer = SummaryWriter(f"{self.outputdir}/tempLog/{VERSION}-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}/")
            self.logger = setup_logger(self.outputdir, VERSION)


    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])

        if(self.gpu_id == 0):
            self.logger.debug(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        else: 
            print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")

        loss_sum = 0.0

        self.train_data.sampler.set_epoch(epoch)
        for idx, (source, _) in enumerate(self.train_data):
            source = source.to(self.gpu_id)
            sourceRaw = source.squeeze(0)
            
            x, y = get_batch(sourceRaw) # (B,C,H)
            y = y[:,:BLOCK_SIZE,:]

            logits, loss = self.model(x, y)
            loss_sum += loss.item()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            if(self.gpu_id == 0 and idx % self.eval_i == 0 and idx != 0):
                train_loss = loss_sum / self.eval_i
                val_loss = 0.0
                val_loss = validate(self.model, self.val_loader)
                train_val_loss = validate(self.model, self.train_val_loader)

                self.logger.debug(f'Epoch {epoch}, Iteration {idx}: Train Loss: {train_loss}, Train_val Loss: {train_val_loss}, Validation Loss: {val_loss}, {int((time.time() - self.start_time) // 3600):02d}:{int(((time.time() - self.start_time) % 3600) // 60):02d}:{int((time.time() - self.start_time) % 60):02d}')
                
                loss_sum = 0.0
                self.writer.add_scalars('Losses', {'Training Loss': train_val_loss, 'Validation Loss': val_loss}, epoch * b_sz // self.eval_i + idx // self.eval_i)
    
            # else:
            #     print(f"GPU{self.gpu_id}: Epoch {epoch}, Iteration {idx}")

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = model_saveFile(self.outputdir, VERSION, epoch)
        torch.save(ckp, PATH)
        if(self.gpu_id == 0):
            self.logger.debug(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)

            if self.gpu_id == 0:
                self._save_checkpoint(epoch)

        if(self.gpu_id == 0):
            self.writer.close()


def main(rank: int, world_size: int, args, save_every: int, total_epochs: int, train_data, val_data, train_val_data):

    if(rank == 0):
        logger = setup_logger(args.out, VERSION)

        logger_write_parameter(
            logger=logger,
            device=device, 
            dataset=args.dataset,
            train_size=-1,
            valsize=0,
            trainvalsize=0,
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
            model_parameter=count_parameters(ColumnTransformer())
        )
    


    ddp_setup(rank, world_size)

    model = ColumnTransformer()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader = DataLoader(train_data, batch_size=1, pin_memory=True, shuffle=False, sampler=DistributedSampler(train_data))

    if(rank == 0):
        trainer = Trainer(model, train_loader, optimizer, rank, save_every, args.out, logger, val_data, train_val_data)
    else:
        trainer = Trainer(model, train_loader, optimizer, rank, save_every, args.out)

    trainer.train(total_epochs)
    destroy_process_group()



if __name__ == '__main__':
    try:
            args = parseParameter()
            outputdir = args.out

            if not os.path.exists(outputdir +'/tempModel/'):
                os.makedirs(outputdir + '/tempModel/')

            


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

            world_size = torch.cuda.device_count()

            mp.spawn(main, args=(world_size, args, 1000, args.iter, train_data, val_data, train_val_data), nprocs=world_size)

            

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
        raise e