# %%
import torch
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import Subset

@torch.no_grad()
def getDataSet(path, dataset_name, size_x, size_y, equalize = False, repeatData=1):

    equalizingPercentage = 0.10

    DataDic = {
        "AllData_1080x": ["Data_Polyhaven1k", "Data_Poliigon", "Data_Minecraft/1024x1024", "Data_freepbr1k", "Data_CSGO_Floor"],
        "AllData_512x": ["Data_Minecraft/512x512", "Data_CSGO_Floor_qHD"],
        "Minecraft_1024x": ["Data_Minecraft/1024x1024"],
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

# %%

path = r"C:\Users\Dennis\Desktop\Pro\AITest\imgGen\Data"

dataset = getDataSet(path, "AllData_1080x", 512, 512, True, 1)
len(dataset)


