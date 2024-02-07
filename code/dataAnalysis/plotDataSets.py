from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

datasets = ["FreePBR", "Polyhaven", "Poliigon", "Minecraft_1024x", "CsGoFloor_1080x"]

for data in datasets:

    csv_file_path_sorted = f'output/dataSetAnalysis-{data}.csv'


    df = pd.read_csv(csv_file_path_sorted)

    df['Farbe'] = df['Farbe'].apply(lambda x: [int(val) for val in x.strip("()").split(",")])


    R = df['Farbe'].apply(lambda x: x[0])
    G = df['Farbe'].apply(lambda x: x[1])
    B = df['Farbe'].apply(lambda x: x[2])
    size = np.log(df['Anzahl']) * 20

    # Erstellen des 3D-Plots
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter-Plot
    sc = ax.scatter(R, G, B, s=size, c=df['Farbe'].apply(lambda x: np.array(x)/255).tolist(), alpha=0.6)

    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    ax.set_title(f'3D-Visualization of color frequencies in RGB-space \n Dataset [{data}]')

    ax.view_init(elev=30, azim=-60 )

    # Animationsfunktion
    def update(frame):
        ax.view_init(elev=30, azim=frame)
        return sc,

    plt.savefig(f'output/{data}.png')

    ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 4), blit=True)

    ani.save(f'output/{data}.gif', writer='imagemagick')
