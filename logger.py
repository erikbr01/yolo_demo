from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Logger:
    def __init__(self) -> None:
        # Store (x,y,z) position
        self.records = np.empty((0,4))
        


    def record_value(self, record):
        self.records = np.concatenate((self.records, record))
        
    
    def export_to_csv(self, output_location):
        cols = ['x', 'y', 'z', 't']
        df = pd.DataFrame(data=self.records, columns=cols)
        df.to_csv(output_location)

    def visualize3D(self, input_location):
        df = pd.read_csv(input_location)
        x_vec = df['x'].to_numpy()
        y_vec = df['y'].to_numpy()
        z_vec = df['z'].to_numpy()
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        
        for i in range(0, len(x_vec)):
            ax.scatter(x_vec[i], y_vec[i], z_vec[i], c='blue')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()


    def visualize2D(self, input_location):
        df = pd.read_csv(input_location)
        x_vec = df['x'].to_numpy()
        y_vec = df['y'].to_numpy()
        
        fig = plt.figure()
        ax = fig.add_subplot()
        
        for i in range(0, len(x_vec)):
            ax.scatter(x_vec[i], y_vec[i], c='blue')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim((0,1920))
        ax.set_ylim((0,1080))
        plt.show()