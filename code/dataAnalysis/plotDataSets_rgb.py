from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import os

# datasets = ['Abzu_x512', 'BatmanArkhamKnight_x512', 'BioShockInf_x512', 'Borderlands_x512', 'Borderlands2_x512', 'BrothersATOTS_x512', 'CitiesSkylines_x512', 'CsGo_x512', 'Dauntless_x512', 'Dishonored_x512', 'Freepbr1k_x512', 'GarrysMod_x512', 'HalfLife2_x512', 'Left4Dead2_x512', 'Maneater_x512', 'Minecraft_x512', 'MutantYearZeroRTE_x512', 'OperatorNdmQG_x512', 'Phasmophobia_x512', 'Poliigon_x512', 'Polyhaven1k_x512', 'Portal_x512', 'Portal2_x512', 'Relicta_x512', 'RemnantFromTheAshes_x512', 'Ruiner_x512', 'StarWarsJFO_x512', 'TeamFortress2_x512', 'TrainSimWorld_x512', 'TrainSimWorld2_x512', 'UnityAssets_x512', 'Vampyr_x512', 'XCOM_x512', 'XCOM2_x512', 'AllData_x512']

datasets = ['AllData_x512', 'Freepbr1k_x512', 'Minecraft_x512', 'UnityAssets_x512']

if os.name == 'posix':
    path = "/scratch/usr/nwmdgthk/dataAnalysisCSVData"
else:
    path = r"C:\Users\Dennis\Desktop\Pro\MasterThesis\code\dataAnalysis\tempOutput"

output_dir = "plots/rgb_gif"

def process_dataset(data):
    try:
        print(f'starting {data}')
        csv_file_path_sorted = os.path.join(path, f'dataSetAnalysis-{data}.csv')
        df = pd.read_csv(csv_file_path_sorted)
        df['Color'] = df['Color'].apply(lambda x: [int(val) for val in x.strip("()").split(",")])


        R = df['Color'].apply(lambda x: x[0])
        G = df['Color'].apply(lambda x: x[1])
        B = df['Color'].apply(lambda x: x[2])
        size = np.log(df['Count']) * 20

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(R, G, B, s=size, c=df['Color'].apply(lambda x: np.array(x)/255).tolist(), alpha=0.6)

        ax.set_xlabel('Red')
        ax.set_ylabel('Green')
        ax.set_zlabel('Blue')
        ax.set_title(f'3D-Visualization of color frequencies in RGB-space \n Dataset [{data}]')
        ax.set_xlim([0, 255])
        ax.set_ylim([0, 255])
        ax.set_zlim([0, 255])
        ax.view_init(elev=30, azim=-60)

        # plt.savefig(f'{output_dir}/{data}.png')
        # print(f'{output_dir}/{data}.png')

        def update(frame):
            print(f'frame {frame} {data}')
            ax.view_init(elev=30, azim=frame)
            return sc,
        
        ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 4), blit=True)
        print(f'Start ploting {data}.gif')
        ani.save(f'{output_dir}/{data}.gif', writer='imagemagick')
        plt.close(fig)
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