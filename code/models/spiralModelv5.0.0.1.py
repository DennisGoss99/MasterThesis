import datetime
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from tqdm import tqdm
import os
from torch.utils.data import ConcatDataset
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

torch.manual_seed(1337)

LEARNING_RATE = 3e-5

BATCH_SIZE = 16

IMAGE_SIZE = 64
BLOCK_SIZE = IMAGE_SIZE * IMAGE_SIZE - 1

CHANNELS_IMG = 3

N_EMBD = 128
N_HEAD = 6
N_LAYER = 6
DROPOUT = 0.2

VERSION = "5.0.0.2_bigData"

#----------------------------------------------

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

spiral_indices = torch.tensor(create_spiral(IMAGE_SIZE))


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
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_saveFile(version):
    return f"tempModel/model{VERSION}_FloorEP{version}.pth"

@torch.no_grad()
def validate(model, dataloader):
    total_loss = 0
    total_samples = 0
    for idx, (data, _) in enumerate(dataloader):
        dataRaw = data.squeeze(0).to(device)
        x, y = get_batch(dataRaw)
        
        _, loss = model(x, y)

        total_loss += loss.item()
        total_samples += 1
    return total_loss / total_samples

#----------------------------------------------

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device: ",device)


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


transform = transforms.Compose(
    [
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ]
)

combined_dataset_Nr = 2

combined_dataset = torch.utils.data.ConcatDataset([datasets.ImageFolder(root="C:/Users/Dennis/Desktop/Pro/AITest/imgGen/Data/Data_CSGO_Floor", transform=transform) for _ in range(combined_dataset_Nr)])

total_size = len(combined_dataset)
val_size = 500
train_size = total_size - val_size  # Calculate training size

train_data, val_data = random_split(combined_dataset, [train_size, val_size])


print(f'train_data: {len(train_data)}, val_data: {len(val_data)}')

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

print(f'BLOCK_SIZE: {BLOCK_SIZE}, BATCH_SIZE: {BATCH_SIZE}, CHANNELS_IMG: {CHANNELS_IMG}, IMAGE_SIZE: {IMAGE_SIZE}, N_EMBD: {N_EMBD}')



m = ColumnTransformer()
m = m.to(device)

optimizer = optim.Adam(m.parameters(), lr=LEARNING_RATE)

#print parameterscount
print(f'Model has {count_parameters(m):,} trainable parameters')


iter_i = 3
eval_i = 100
eval_img = 100

writer = SummaryWriter(f"tempLog/{VERSION}_Floor/")
model_path = 'tempModel/'
if not os.path.exists(model_path):
    os.makedirs(model_path)

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
            print(f'Epoch {e}, Iteration {idx}: Train Loss: {train_loss}, Validation Loss: {val_loss}')
            torch.save(m, model_saveFile(e))
            loss_sum = 0.0

            writer.add_scalars('Losses', {'Training Loss': train_loss, 'Validation Loss': val_loss}, e * len(train_loader) // eval_i + idx // eval_i)

writer.close()