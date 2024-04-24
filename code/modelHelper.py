import torch
import argparse
import logging
import datetime

@torch.no_grad()
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_saveFile(outputdir, version, epoch, train=''):
    return f"{outputdir}/tempModel/model{train}{version}_EP{epoch}-{datetime.datetime.now().strftime('%Y%m%d')}.pth"

def parseParameter():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, help='dataset path', required=True)
    parser.add_argument('-o', '--out', type=str, help='output dir path', required=False, default='./')
    parser.add_argument('-d','--dataset', type=str, help='Select between: \"AllData_1080x\", \"AllData_512x\", \"CsGoFloor_1080x\", \"CsGoFloor_512x\", \"FreePBR\", \"Polyhaven\", \"Poliigon\"', required=True)
    parser.add_argument('-v', '--valsize', type=str, help='size of the validation dataset [default 10%]', required=False , default="10%")
    parser.add_argument('-tv', '--trainvalsize', type=str, help='size of the train validation dataset [default 10%]', required=False , default="10%")
    parser.add_argument('-i', '--iter', type=int, help='number of iterations [default 1]', required=False , default=1)
    parser.add_argument('-r', '--repeatdataset', type=int, help='repeat dataset [default 1]', required=False , default=1)

    args = parser.parse_args()
    return args



@torch.no_grad()
def logger_write_parameter(logger, device, dataset, train_size, valsize, trainvalsize, repeatdataset, learning_rate, batch_size, image_size, block_size, channels_img, n_embd, n_head, n_layer, dropout, version, model_parameter):
    logger.info(f'LEARNING_RATE={learning_rate}')
    logger.info(f'BATCH_SIZE={batch_size}')
    logger.info(f'IMAGE_SIZE={image_size}')
    logger.info(f'BLOCK_SIZE={block_size}')
    logger.info(f'CHANNELS_IMG={channels_img}')
    logger.info(f'N_EMBD={n_embd}')
    logger.info(f'N_HEAD={n_head}')
    logger.info(f'N_LAYER={n_layer}')
    logger.info(f'DROPOUT={dropout}')
    logger.info(f'MODEL_PARAMETER={model_parameter}')
    logger.info(f'VERSION={version}')
    logger.info(f'----------------')
    logger.info(f'device={device}')
    logger.info(f'gpus={torch.cuda.device_count()}')
    logger.info(f'dataset={dataset}')
    logger.info(f'train_size={train_size}')
    logger.info(f'valsize={valsize}')
    logger.info(f'trainvalsize={trainvalsize}')
    logger.info(f'repeatdataset={repeatdataset}')
    logger.info(f'----------------')


def setup_logger(outputdir, version):
    logger = logging.getLogger('model_logger')
    logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(f"{outputdir}/tempModel/modelLog{version}-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")    
    stream_handler.setLevel(logging.DEBUG)
    file_handler.setLevel(logging.DEBUG)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger