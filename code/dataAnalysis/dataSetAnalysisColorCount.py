import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms
import os
import sys

module_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
sys.path.append(module_dir)
from dataSetCombiner import getDataSet


def count_colors_in_batch(rgb_array, color_counts):
    for color in map(tuple, rgb_array):
        if color in color_counts:
            color_counts[color] += 1
        else:
            color_counts[color] = 1

datasetStrings = ["FreePBR", "Polyhaven", "Poliigon", "Minecraft_1024x", "CsGoFloor_1080x"]

path = "/scratch/usr/nwmdgthk/allData/Data"

for datasetString in datasetStrings:

    dataset = getDataSet(path, datasetString, 1024, 1024, True, 1)
    dataset_len = len(dataset)
    print(datasetString, dataset_len)

    color_counts = {} 
    percentage = 100

    for i, (data, _) in enumerate(dataset):

        if percentage == 100:
            selected_rgb_values = (data.view(3, -1).t() * 255).to(torch.int32).numpy()
        else:
            reshaped_tensor = data.view(3, -1).t().numpy() * 255
            indices = np.random.choice(reshaped_tensor.shape[0], size=(reshaped_tensor.shape[0] * percentage) // 100, replace=False)
            selected_rgb_values = reshaped_tensor[indices].astype(int)
        
        # if (i + 1) % 100 == 0:
        print(f"{i + 1}/{len(dataset)}")

        count_colors_in_batch(selected_rgb_values, color_counts)

    print(f"Anzahl der Farben: {len(color_counts)}")
        
    df_colors = pd.DataFrame(list(color_counts.items()), columns=['Farbe', 'Anzahl'])
    df_colors_sorted = df_colors.sort_values(by='Farbe').reset_index(drop=True)

    csv_file_path = f"output/dataSetAnalysis-{datasetString}.csv"
    df_colors_sorted.to_csv(csv_file_path, index=False)