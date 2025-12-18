from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.io
import scipy as sp

def plot_points_2D(points):
    
    # Your code here
    x = points[0,:]
    y = points[1,:]
    
    ax = plt.gca()
    ax.scatter(x, y)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('2D points')

def plot_points_3D(points, ax=None, size=5, kwargs={}):

    if points.ndim == 1:
        points = points.reshape(3,1)
    
    # Your code here
    x = points[0,:]
    y = points[1,:]
    w = points[2,:]

    if ax is None:
        ax = plt.subplot(111, projection='3d')

    ax.scatter3D(x, y, w, s=size, **kwargs)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('w')
    ax.set_box_aspect([1, 1, 1])

    return ax
