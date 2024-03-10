# %%
import torch
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import Subset

@torch.no_grad()
def getDataSet(path, dataset_name, size_x, size_y, repeatData=1):

    DataDic = {
        "AllData_x1024": [  
            'Data_Abzu/x1024',
            'Data_BatmanArkhamKnight/x1024',
            'Data_BioShockInf/x1024',
            'Data_Borderlands/x1024',
            'Data_Borderlands2/x1024',
            'Data_BrothersATOTS/x1024',
            'Data_CitiesSkylines/x1024',
            'Data_CsGo/x1024',
            'Data_Dauntless/x1024',
            'Data_Dishonored/x1024',
            'Data_Freepbr1k/x1024',
            'Data_GarrysMod/x1024',
            'Data_HalfLife2/x1024',
            'Data_Left4Dead2/x1024',
            'Data_Maneater/x1024',
            'Data_Minecraft/x1024',
            'Data_MutantYearZeroRTE/x1024',
            'Data_Phasmophobia/x1024',
            'Data_Poliigon/x1024',
            'Data_Polyhaven1k/x1024',
            'Data_Portal2/x1024',
            'Data_Relicta/x1024',
            'Data_RemnantFromTheAshes/x1024',
            'Data_Ruiner/x1024',
            'Data_StarWarsJFO/x1024',
            'Data_TeamFortress2/x1024',
            'Data_TrainSimWorld/x1024',
            'Data_TrainSimWorld2/x1024',
            'Data_UnityAssets/x1024',
            'Data_Vampyr/x1024',
            'Data_XCOM/x1024',
            'Data_XCOM2/x1024'],
        "AllData_x512": [
            'Data_Abzu/x512',
            'Data_BatmanArkhamKnight/x512',
            'Data_BioShockInf/x512',
            'Data_Borderlands/x512',
            'Data_Borderlands2/x512',
            'Data_BrothersATOTS/x512',
            'Data_CitiesSkylines/x512',
            'Data_CsGo/x512',
            'Data_Dauntless/x512',
            'Data_Dishonored/x512',
            'Data_Freepbr1k/x512',
            'Data_GarrysMod/x512',
            'Data_HalfLife2/x512',
            'Data_Left4Dead2/x512',
            'Data_Maneater/x512',
            'Data_Minecraft/x512',
            'Data_MutantYearZeroRTE/x512',
            'Data_OperatorNdmQG/x512',
            'Data_Phasmophobia/x512',
            'Data_Poliigon/x512',
            'Data_Polyhaven1k/x512',
            'Data_Portal/x512',
            'Data_Portal2/x512',
            'Data_Relicta/x512',
            'Data_RemnantFromTheAshes/x512',
            'Data_Ruiner/x512',
            'Data_StarWarsJFO/x512',
            'Data_TeamFortress2/x512',
            'Data_TrainSimWorld/x512',
            'Data_TrainSimWorld2/x512',
            'Data_UnityAssets/x512',
            'Data_Vampyr/x512',
            'Data_XCOM/x512',
            'Data_XCOM2/x512'],

        'Abzu_x1024' : ['Data_Abzu/x1024'],
        'BatmanArkhamKnight_x1024'  : ['Data_BatmanArkhamKnight/x1024'],
        'BioShockInf_x1024' : ['Data_BioShockInf/x1024'],
        'Borderlands_x1024' : ['Data_Borderlands/x1024'],
        'Borderlands2_x1024'  : ['Data_Borderlands2/x1024'],
        'BrothersATOTS_x1024'  : ['Data_BrothersATOTS/x1024'],
        'CitiesSkylines_x1024' : ['Data_CitiesSkylines/x1024'],
        'CsGo_x1024' : ['Data_CsGo/x1024'],
        'Dauntless_x1024' : ['Data_Dauntless/x1024'],
        'Dishonored_x1024' : ['Data_Dishonored/x1024'],
        'Freepbr1k_x1024' : ['Data_Freepbr1k/x1024'],
        'GarrysMod_x1024' : ['Data_GarrysMod/x1024'],
        'HalfLife2_x1024' : ['Data_HalfLife2/x1024'],
        'Left4Dead2_x1024' : ['Data_Left4Dead2/x1024'],
        'Maneater_x1024' : ['Data_Maneater/x1024'],
        'Minecraft_x1024' : ['Data_Minecraft/x1024'],
        'MutantYearZeroRTE_x1024'  : ['Data_MutantYearZeroRTE/x1024'],
        'Phasmophobia_x1024' : ['Data_Phasmophobia/x1024'],
        'Poliigon_x1024' : ['Data_Poliigon/x1024'],
        'Polyhaven1k_x1024' : ['Data_Polyhaven1k/x1024'],
        'Portal2_x1024' : ['Data_Portal2/x1024'],
        'Relicta_x1024' : ['Data_Relicta/x1024'],
        'RemnantFromTheAshes_x1024'  : ['Data_RemnantFromTheAshes/x1024'],
        'Ruiner_x1024' : ['Data_Ruiner/x1024'],
        'StarWarsJFO_x1024' : ['Data_StarWarsJFO/x1024'],
        'TeamFortress2_x1024'  : ['Data_TeamFortress2/x1024'],
        'TrainSimWorld_x1024'  : ['Data_TrainSimWorld/x1024'],
        'TrainSimWorld2_x1024'  : ['Data_TrainSimWorld2/x1024'],
        'UnityAssets_x1024' : ['Data_UnityAssets/x1024'],
        'Vampyr_x1024' : ['Data_Vampyr/x1024'],
        'XCOM_x1024' : ['Data_XCOM/x1024'],
        'XCOM2_x1024' : ['Data_XCOM2/x1024'],
            
        'Abzu_x512' : ['Data_Abzu/x512'],
        'BatmanArkhamKnight_x512'  : ['Data_BatmanArkhamKnight/x512'],
        'BioShockInf_x512' : ['Data_BioShockInf/x512'],
        'Borderlands_x512' : ['Data_Borderlands/x512'],
        'Borderlands2_x512'  : ['Data_Borderlands2/x512'],
        'BrothersATOTS_x512'  : ['Data_BrothersATOTS/x512'],
        'CitiesSkylines_x512' : ['Data_CitiesSkylines/x512'],
        'CsGo_x512' : ['Data_CsGo/x512'],
        'Dauntless_x512' : ['Data_Dauntless/x512'],
        'Dishonored_x512' : ['Data_Dishonored/x512'],
        'Freepbr1k_x512' : ['Data_Freepbr1k/x512'],
        'GarrysMod_x512' : ['Data_GarrysMod/x512'],
        'HalfLife2_x512' : ['Data_HalfLife2/x512'],
        'Left4Dead2_x512' : ['Data_Left4Dead2/x512'],
        'Maneater_x512' : ['Data_Maneater/x512'],
        'Minecraft_x512' : ['Data_Minecraft/x512'],
        'MutantYearZeroRTE_x512'  : ['Data_MutantYearZeroRTE/x512'],
        'OperatorNdmQG_x512' : ['Data_OperatorNdmQG/x512'],
        'Phasmophobia_x512' : ['Data_Phasmophobia/x512'],
        'Poliigon_x512' : ['Data_Poliigon/x512'],
        'Polyhaven1k_x512' : ['Data_Polyhaven1k/x512'],
        'Portal_x512' : ['Data_Portal/x512'],
        'Portal2_x512' : ['Data_Portal2/x512'],
        'Relicta_x512' : ['Data_Relicta/x512'],
        'RemnantFromTheAshes_x512'  : ['Data_RemnantFromTheAshes/x512'],
        'Ruiner_x512' : ['Data_Ruiner/x512'],
        'StarWarsJFO_x512' : ['Data_StarWarsJFO/x512'],
        'TeamFortress2_x512'  : ['Data_TeamFortress2/x512'],
        'TrainSimWorld_x512'  : ['Data_TrainSimWorld/x512'],
        'TrainSimWorld2_x512'  : ['Data_TrainSimWorld2/x512'],
        'UnityAssets_x512' : ['Data_UnityAssets/x512'],
        'Vampyr_x512' : ['Data_Vampyr/x512'],
        'XCOM_x512' : ['Data_XCOM/x512'],
        'XCOM2_x512' : ['Data_XCOM2/x512'],
    }

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
        datasets_list.append(dataset)

    combined_dataset = torch.utils.data.ConcatDataset(datasets_list * repeatData)

    return combined_dataset