import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms
import os
import sys
import time

module_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
sys.path.append(module_dir)
from dataSetCombiner import getDataSet


def count_colors_in_batch(rgb_array, color_counts):
    for color in map(tuple, rgb_array):
        if color in color_counts:
            color_counts[color] += 1
        else:
            color_counts[color] = 1



if os.name == 'posix':
    path = "/scratch/usr/nwmdgthk/allData/Data/"
else:
    path = r"C:\Users\Dennis\Desktop\Pro\AITest\imgGen\Data"

output_dir = "tempOutput/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


datasetStrings = ['OperatorNdmQG_x512', 'Phasmophobia_x512', 'Poliigon_x512', 'Polyhaven1k_x512', 'Portal_x512', 'Portal2_x512', 'Relicta_x512', 'RemnantFromTheAshes_x512', 'Ruiner_x512', 'StarWarsJFO_x512', 'TeamFortress2_x512', 'TrainSimWorld_x512', 'TrainSimWorld2_x512', 'UnityAssets_x512', 'Vampyr_x512', 'XCOM_x512', 'XCOM2_x512', 'AllData_x512']

datasetString_len = len(datasetStrings)

for datasetString_i, datasetString in enumerate(datasetStrings, start=1):

    dataset = getDataSet(path, datasetString, 512, 512, 1)

    start_time = time.time()
    total_items = len(dataset)
    update_frequency = max(total_items // 100, 1)  # Update every 1% of the total items

    print(datasetString, total_items)

    color_counts = {} 


    for i, (data, _) in enumerate(dataset):

        if (i + 1) % update_frequency == 0 or i == 0 or i == total_items - 1:
            # Current progress calculation
            progress = ((i + 1) / total_items) * 100
            
            # Time elapsed calculation
            elapsed_time = time.time() - start_time
            
            # Estimate remaining time
            if i > 0:  # Avoid division by zero
                estimated_total_time = elapsed_time / (i + 1) * total_items
                estimated_remaining_time = estimated_total_time - elapsed_time
                remaining_time_formatted = time.strftime("%H:%M:%S", time.gmtime(estimated_remaining_time))
            else:
                remaining_time_formatted = "Calculating..."

            print(f"\t {datasetString_i}/{datasetString_len} Processing {i + 1}/{total_items} ({progress:.2f}%) - Estimated Time Remaining: {remaining_time_formatted}")

        selected_rgb_values = (data.view(3, -1).t() * 255).to(torch.int32).numpy()
        
        count_colors_in_batch(selected_rgb_values, color_counts)

    print(f"Number of colors: {len(color_counts)}\n__________________________\n")
        
    df_colors = pd.DataFrame(list(color_counts.items()), columns=['Color', 'Count'])
    df_colors_sorted = df_colors.sort_values(by='Color').reset_index(drop=True)

    csv_file_path = f"{output_dir}/dataSetAnalysis-{datasetString}.csv"
    df_colors_sorted.to_csv(csv_file_path, index=False)
