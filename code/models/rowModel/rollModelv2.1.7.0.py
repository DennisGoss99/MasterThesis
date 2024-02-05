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

torch.manual_seed(1337)

LEARNING_RATE = 3e-5

BATCH_SIZE = 128

BLOCK_SIZE = 128
IMAGE_SIZE = 129 + 12

CHANNELS_IMG = 3

N_EMBD = 128
N_HEAD = 6
N_LAYER = 6
DROPOUT = 0.2

VERSION = "2.1.7.0_bigData"

#----------------------------------------------

@torch.no_grad()
def get_batch(data):
    # C ,H ,W = data.shape

    x = data[:, :BLOCK_SIZE, :BATCH_SIZE]
    y = data[:, 1:BLOCK_SIZE*2+1, :BATCH_SIZE]

    x, y = x.to(device), y.to(device)
    x = rearrange(x, 'c h b -> b h c')
    y = rearrange(y, 'c h b -> b h c')
    return x, y

@torch.no_grad()
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def model_saveFile(version, train='pretrain'):
    return f"tempModel/model{train}{VERSION}_FloorEP{version}.pth"

@torch.no_grad()
def getDataSet(path, dataset_name, size_x, size_y, equalize = False, repeatData=1):

    equalizingPercentage = 0.10

    DataDic = {
        "AllData_1080x": ["Data_Polyhaven1k", "Data_Poliigon", "Data_Minecraft/1024x1024", "Data_freepbr1k", "Data_CSGO_Floor"],
        "AllData_512x": ["Data_Minecraft/512x512", "Data_CSGO_Floor_qHD"],
        "CsGoFloor_1080x": ["Data_CSGO_Floor"],
        "CsGoFloor_512x": ["Data_CSGO_Floor_qHD"],
        "FreePBR": ["Data_freepbr1k"],
        "Polyhaven": ["Data_Polyhaven1k"],
        "Poliigon": ["Data_Poliigon"],
    }

    # Prüfen, ob der angegebene Datensatzname im Wörterbuch vorhanden ist
    if dataset_name not in DataDic:
        raise ValueError(f"Datensatzname '{dataset_name}' nicht im DataDic gefunden.")

    selected_data_paths = [f"{path}\\{folder}" for folder in DataDic[dataset_name]]

    transform = transforms.Compose(
        [
            transforms.RandomCrop((size_x, size_y)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ]
    )

    datasets_list = []

    for folder_path in selected_data_paths: 
        dataset = datasets.ImageFolder(root=folder_path, transform=transform)
        if equalize and ("Data_CSGO_Floor" in folder_path or "Data_CSGO_Floor_qHD" in folder_path):
            subset_size = int(len(dataset) * equalizingPercentage)
            indices = np.random.choice(range(len(dataset)), subset_size, replace=False)
            dataset = Subset(dataset, indices)

        datasets_list.append(dataset)

    combined_dataset = torch.utils.data.ConcatDataset(datasets_list * repeatData)

    return combined_dataset

@torch.no_grad()
def write_parameter(dataset, train_size, valsize, repeatdataset):
    path = f"tempModel/modelParameter{VERSION}.txt"
    with open(path, 'w') as file:
        file.write(f'LEARNING_RATE={LEARNING_RATE}\n')
        file.write(f'BATCH_SIZE={BATCH_SIZE}\n')
        file.write(f'IMAGE_SIZE={IMAGE_SIZE}\n')
        file.write(f'BLOCK_SIZE={BLOCK_SIZE}\n')
        file.write(f'CHANNELS_IMG={CHANNELS_IMG}\n')
        file.write(f'N_EMBD={N_EMBD}\n')
        file.write(f'N_HEAD={N_HEAD}\n')
        file.write(f'N_LAYER={N_LAYER}\n')
        file.write(f'DROPOUT={DROPOUT}\n')
        file.write(f'VERSION={VERSION}\n')
        file.write(f'----------------\n')
        file.write(f'device={device}\n')
        file.write(f'dataset={dataset}\n')
        file.write(f'train_size={train_size}\n')
        file.write(f'valsize={valsize}\n')
        file.write(f'repeatdataset={repeatdataset}\n')

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

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, help='dataset path', required=True)
    parser.add_argument('-d','--dataset', type=str, help='Select between: \"AllData_1080x\", \"AllData_512x\", \"CsGoFloor_1080x\", \"CsGoFloor_512x\", \"FreePBR\", \"Polyhaven\", \"Poliigon\"', required=True)
    parser.add_argument('-v', '--valsize', type=int, help='size of the validation dataset [default 500]', required=False , default=500)
    parser.add_argument('-i', '--iter', type=int, help='number of iterations [default 1]', required=False , default=1)
    parser.add_argument('-r', '--repeatdataset', type=int, help='repeat dataset [default 1]', required=False , default=1)
    parser.add_argument('-e', '--equalize', action='store_true', help='equalize Dataset [default: False]')

    args = parser.parse_args()

    print("device: ",device)
    print(f'The path to the dataset is: {args.path}')

    dataset = getDataSet(args.path, args.dataset, IMAGE_SIZE, IMAGE_SIZE, args.equalize, args.repeatdataset)

    total_size = len(dataset)
    val_size = 500
    train_size = total_size - val_size  # Calculate training size

    train_data, val_data = random_split(dataset, [train_size, val_size])


    print(f'train_data: {len(train_data)}, val_data: {len(val_data)}')

    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=True)

    print(f'BLOCK_SIZE: {BLOCK_SIZE}, BATCH_SIZE: {BATCH_SIZE}, CHANNELS_IMG: {CHANNELS_IMG}, IMAGE_SIZE: {IMAGE_SIZE}, N_EMBD: {N_EMBD}')

    m = ColumnTransformer()
    m = m.to(device)

    optimizer = optim.Adam(m.parameters(), lr=LEARNING_RATE)

    #print parameterscount
    print(f'Model has {count_parameters(m):,} trainable parameters')


    iter_i = args.iter
    eval_i = 1000
    eval_img = 1000

    writer = SummaryWriter(f"tempLog/{VERSION}_Floor/pretrain/")

    model_path = 'tempModel/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    write_parameter(args.dataset, train_size, val_size, args.repeatdataset)

    for e in range(0,iter_i):
        loss_sum = 0.0
        for idx, (data, _) in enumerate(train_loader):

            dataRaw = data.squeeze(0).to(device)
            x, y = get_batch(dataRaw) # (B,C,H)
            y = y[:,:BLOCK_SIZE,:]
        
            logits, loss = m(x, y)
            
            loss_sum += loss.item()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if idx % eval_img == 0 and idx != 0:
                combined_images = torch.cat([rearrange(y, 'b h c -> 1 c h b'), rearrange(logits, 'b h c -> 1 c h b')], dim=3)
                image_grid = vutils.make_grid(combined_images)
                writer.add_image('Generated vs Original', image_grid, e* len(train_loader)// eval_img + idx // eval_img)


            if(idx % eval_i == 0 and idx != 0):
                train_loss = loss_sum / eval_i
                val_loss = 0.0
                # val_loss = validate(m, val_loader)
                print(f'Epoch {e}, Iteration {idx}: Train Loss: {train_loss}, Validation Loss: {val_loss}')
                torch.save(m, model_saveFile(e))
                loss_sum = 0.0

                writer.add_scalars('Losses', {'Training Loss': train_loss, 'Validation Loss': val_loss}, e * len(train_loader) // eval_i + idx // eval_i)
    writer.close()

if __name__ == '__main__':
    main()   