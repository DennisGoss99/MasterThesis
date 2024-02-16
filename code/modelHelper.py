import torch
import argparse

@torch.no_grad()
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def model_saveFile(version, epoch, train='pretrain'):
    return f"tempModel/model{train}{version}_FloorEP{epoch}.pth"

def parseParameter():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, help='dataset path', required=True)
    parser.add_argument('-d','--dataset', type=str, help='Select between: \"AllData_1080x\", \"AllData_512x\", \"CsGoFloor_1080x\", \"CsGoFloor_512x\", \"FreePBR\", \"Polyhaven\", \"Poliigon\"', required=True)
    parser.add_argument('-v', '--valsize', type=int, help='size of the validation dataset [default 500]', required=False , default=500)
    parser.add_argument('-i', '--iter', type=int, help='number of iterations [default 1]', required=False , default=1)
    parser.add_argument('-r', '--repeatdataset', type=int, help='repeat dataset [default 1]', required=False , default=1)
    parser.add_argument('-e', '--equalize', action='store_true', help='equalize Dataset [default: False]')

    args = parser.parse_args()
    return args

@torch.no_grad()
def write_parameter(device, dataset, train_size, valsize, repeatdataset, equalize, learning_rate, batch_size, image_size, block_size, channels_img, n_embd, n_head, n_layer, dropout, version):
    path = f"tempModel/modelParameter{version}.txt"
    with open(path, 'w') as file:
        file.write(f'LEARNING_RATE={learning_rate}\n')
        file.write(f'BATCH_SIZE={batch_size}\n')
        file.write(f'IMAGE_SIZE={image_size}\n')
        file.write(f'BLOCK_SIZE={block_size}\n')
        file.write(f'CHANNELS_IMG={channels_img}\n')
        file.write(f'N_EMBD={n_embd}\n')
        file.write(f'N_HEAD={n_head}\n')
        file.write(f'N_LAYER={n_layer}\n')
        file.write(f'DROPOUT={dropout}\n')
        file.write(f'VERSION={version}\n')
        file.write(f'----------------\n')
        file.write(f'device={device}\n')
        file.write(f'dataset={dataset}\n')
        file.write(f'train_size={train_size}\n')
        file.write(f'valsize={valsize}\n')
        file.write(f'repeatdataset={repeatdataset}\n')
        file.write(f'equalize={equalize}\n')
