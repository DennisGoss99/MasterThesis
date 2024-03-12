import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from skimage import io, color
from mpl_toolkits.axes_grid1 import make_axes_locatable
from concurrent.futures import ProcessPoolExecutor
import os

datasets = ['Abzu_x512', 'BatmanArkhamKnight_x512', 'BioShockInf_x512', 'Borderlands_x512', 'Borderlands2_x512', 'BrothersATOTS_x512', 'CitiesSkylines_x512', 'CsGo_x512', 'Dauntless_x512', 'Dishonored_x512', 'Freepbr1k_x512', 'GarrysMod_x512', 'HalfLife2_x512', 'Left4Dead2_x512', 'Maneater_x512', 'Minecraft_x512', 'MutantYearZeroRTE_x512', 'OperatorNdmQG_x512', 'Phasmophobia_x512', 'Poliigon_x512', 'Polyhaven1k_x512', 'Portal_x512', 'Portal2_x512', 'Relicta_x512', 'RemnantFromTheAshes_x512', 'Ruiner_x512', 'StarWarsJFO_x512', 'TeamFortress2_x512', 'TrainSimWorld_x512', 'TrainSimWorld2_x512', 'UnityAssets_x512', 'Vampyr_x512', 'XCOM_x512', 'XCOM2_x512', 'AllData_x512']

if os.name == 'posix':
    path = "/scratch/usr/nwmdgthk/dataAnalysisCSVData"
else:
    path = r"C:\Users\Dennis\Desktop\Pro\MasterThesis\code\dataAnalysis\tempOutput"

output_dir = "plots/lab"
plot_lim = 128      

def plot_ab(df, name, bins=65, ax_main=None):
    plot_lim = 128
    
    if ax_main is None:
        fig, ax_main = plt.subplots(figsize=(8, 8))
    else:
        fig = ax_main.figure

    counts, y_edges, x_edges = np.histogram2d(df['b'], df['a'], bins=bins)
    x_centers = (x_edges[1:] + x_edges[:-1]) / 2
    y_centers = (y_edges[1:] + y_edges[:-1]) / 2
    ol = 100 - 80 * counts / counts.max()  # Lightness scaled for "beauty"
    oa, ob = np.meshgrid(x_centers, y_centers, indexing='ij')  # Adjust meshgrid indexing
    ol[counts == 0] = 100.0  # Mask areas with zero counts (white)
    oa[counts == 0] = 0.0
    ob[counts == 0] = 0.0
    lab = np.dstack((ol, oa, ob))
    rgb = color.lab2rgb(lab)  # LAB to RGB conversion
    
    ax_main.imshow(rgb, extent=(-plot_lim, plot_lim, -plot_lim, plot_lim), origin='lower')

    ax_main.set_xlabel("a")
    ax_main.set_ylabel("b")
    ax_main.set_xlim(-plot_lim, plot_lim)
    ax_main.set_ylim(-plot_lim, plot_lim)

    # Side histograms
    divider = make_axes_locatable(ax_main)
    ax_histx = divider.append_axes("top", 0.8, pad=0.0, sharex=ax_main)  # Set pad to 0
    ax_histy = divider.append_axes("right", 0.8, pad=0.0, sharey=ax_main)  # Set pad to 0

    ax_histx.hist(df['a'], bins=300, orientation='vertical', color='black', linewidth=0.1, weights=df['LogCount'])
    ax_histy.hist(df['b'], bins=300, orientation='horizontal', color='black', linewidth=0.1, weights=df['LogCount'])


    ax_histx.axis('off')
    ax_histy.axis('off')

    ax_histx.set_xlim(ax_main.get_xlim())

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)  
    plt.savefig(f'{output_dir}/{name}_lab.png')
    plt.close(fig)

def process_dataset(datasetName):
    try:
        print(f'starting {datasetName}')
        csv_file_path_sorted = os.path.join(path, f'dataSetAnalysis-{datasetName}.csv')

        data = pd.read_csv(csv_file_path_sorted)
        df = pd.DataFrame(data)

        # Convert color strings to RGB tuples
        df['RGB'] = df['Color'].apply(lambda x: tuple(map(int, x.strip("()").split(", "))))

        # Convert from RGB to Lab color space
        df['Lab'] = df['RGB'].apply(lambda rgb: color.rgb2lab(np.array([[rgb]]) / 255)[0,0])

        df['a'] = df['Lab'].apply(lambda x: x[1])
        df['b'] = df['Lab'].apply(lambda x: x[2])

        new_row_df = pd.DataFrame([{'a': -plot_lim, 'b': -plot_lim, 'Count': 0}, {'a': plot_lim, 'b': plot_lim, 'Count': 0}]) 
        df = pd.concat([df, new_row_df], ignore_index=True)

        df['LogCount'] = np.log(df['Count'] +1).round().astype(int)

        plot_ab(df, datasetName, bins=300)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("Start")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with ProcessPoolExecutor() as executor:
        executor.map(process_dataset, datasets)

   
    # for dataset in datasets:
    #     process_dataset(dataset)