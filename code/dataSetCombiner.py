# %%
import torch
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import Subset

@torch.no_grad()
def getDataSet(path, dataset_name, size_x, size_y, equalize = False, repeatData=1):

    equalizingPercentage = 0.10

    DataDic = {
        "AllData_1080x": ["Data_Polyhaven1k", "Data_Poliigon", "Data_Minecraft/1024x1024", "Data_freepbr1k", "Data_CSGO_Floor", "Data_Portal", "Data_HalfLife2", "Data_UnityAssets/x1024", "Data_BioShockInf", "Data_StarWarsJFO", "Data_Dishonored"],
        "AllData_512x": ["Data_Minecraft/512x512", "Data_CSGO_Floor_qHD", "Data_UnityAssets/x512"],
        "Minecraft_1024x": ["Data_Minecraft/1024x1024"],
        "CsGoFloor_1080x": ["Data_CSGO_Floor"],
        "CsGoFloor_512x": ["Data_CSGO_Floor_qHD"],
        "FreePBR": ["Data_freepbr1k"],
        "Polyhaven": ["Data_Polyhaven1k"],
        "Poliigon": ["Data_Poliigon"],
        "Portal" : ["Data_Portal"],
        "HalfLife2" : ["Data_HalfLife2"],
        "UnityAssets_1024x" : ["Data_UnityAssets/x1024"],
        "UnityAssets_512x" : ["Data_UnityAssets/x512"],
        "BioShockInf" : ["Data_BioShockInf"],
        "StarWarsJFO" : ["Data_StarWarsJFO"],
        "Dishonored" : ["Data_Dishonored"],
    }

    specific_datasets = [
        "Data_CSGO_Floor",
        "Data_CSGO_Floor_qHD",
        "Data_Portal",
        "Data_HalfLife2"
    ]

    if dataset_name not in DataDic:
        raise ValueError(f"Dataset'{dataset_name}' is not in DataDic")

    selected_data_paths = [f"{path}/{folder}" for folder in DataDic[dataset_name]]

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
        if equalize and any(specific_dataset in folder_path for specific_dataset in specific_datasets):
            subset_size = int(len(dataset) * equalizingPercentage)
            indices = np.random.choice(range(len(dataset)), subset_size, replace=False)
            dataset = Subset(dataset, indices)

        datasets_list.append(dataset)

    combined_dataset = torch.utils.data.ConcatDataset(datasets_list * repeatData)

    return combined_dataset
