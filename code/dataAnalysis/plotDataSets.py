from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

datasets = ["FreePBR", "Polyhaven", "Poliigon", "Minecraft_1024x", "CsGoFloor_1080x"]

def process_dataset(data):
    try:
        print(f'starting {data}')
        csv_file_path_sorted = f'output/dataSetAnalysis-{data}.csv'
        df = pd.read_csv(csv_file_path_sorted)
        df['Farbe'] = df['Farbe'].apply(lambda x: [int(val) for val in x.strip("()").split(",")])


        R = df['Farbe'].apply(lambda x: x[0])
        G = df['Farbe'].apply(lambda x: x[1])
        B = df['Farbe'].apply(lambda x: x[2])
        size = np.log(df['Anzahl']) * 20

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(R, G, B, s=size, c=df['Farbe'].apply(lambda x: np.array(x)/255).tolist(), alpha=0.6)

        ax.set_xlabel('Red')
        ax.set_ylabel('Green')
        ax.set_zlabel('Blue')
        ax.set_title(f'3D-Visualization of color frequencies in RGB-space \n Dataset [{data}]')
        ax.view_init(elev=30, azim=-60)

        plt.savefig(f'output/{data}.png')
        print(f'output/{data}.png')

        def update(frame):
            print(f'frame {frame} {data}')
            ax.view_init(elev=30, azim=frame)
            return sc,
        
        ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 4), blit=True)
        print(f'Start ploting {data}.gif')
        ani.save(f'output/{data}.gif', writer='imagemagick')
        plt.close(fig)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("Start")
    with ProcessPoolExecutor() as executor:
        executor.map(process_dataset, datasets)
        # for dataset in datasets:
        #     process_dataset(dataset)





