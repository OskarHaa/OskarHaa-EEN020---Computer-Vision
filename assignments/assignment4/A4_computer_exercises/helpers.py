# For your convenience:
# Paste the required functions from previous assignments here.

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.io
import scipy as sp
from supplied import pflat

def triangulate_3D_point_DLT(x1, x2, P1, P2):

    # homogeneous image points
    x1_h = np.array([x1[0], x1[1], 1.0])
    x2_h = np.array([x2[0], x2[1], 1.0])

    A = np.vstack([
        x1_h[0] * P1[2, :] - P1[0, :],
        x1_h[1] * P1[2, :] - P1[1, :],
        x2_h[0] * P2[2, :] - P2[0, :],
        x2_h[1] * P2[2, :] - P2[1, :],
    ])

    _, _, Vt = np.linalg.svd(A)
    X_h = Vt[-1, :]
    
    return X_h / X_h[-1]     

def extract_P_from_E(E):
    
     '''
    A function that extract the four P2 solutions given above
    E - Essential matrix (3x3)
    P - Array containing all four P2 solutions (4x3x4) (i.e. P[i,:,:] is the ith solution) 
    '''
     # Your code here
     U, S, Vt = np.linalg.svd(E)

     W = np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]])

     u3 = U[:, 2]

     # Compute two possible rotation matrices
     R1 = U @ W @ Vt
     R2 = U @ W.T @ Vt

     # Ensure rotations are proper (det = +1)
     if np.linalg.det(R1) < 0: R1 = -R1
     if np.linalg.det(R2) < 0: R2 = -R2

     # Four possible camera matrices
     P = np.zeros((4, 3, 4))
     P[0] = np.hstack((R1,  u3.reshape(3,1)))
     P[1] = np.hstack((R1, -u3.reshape(3,1)))
     P[2] = np.hstack((R2,  u3.reshape(3,1)))
     P[3] = np.hstack((R2, -u3.reshape(3,1)))

     return P


def convert_E_to_F(E,K1,K2):
    '''
    A function that gives you a fundamental matrix from an essential matrix and the two calibration matrices
    E - Essential matrix (3x3)
    K1 - Calibration matrix for the first image (3x3)
    K2 - Calibration matrix for the second image (3x3)
    '''
    # Your code here
    K1_inv = np.linalg.inv(K1)
    K2_inv = np.linalg.inv(K2)

    F = K2_inv.T @ E @ K1_inv
    return F


def estimate_F_DLT(x1s, x2s):
    '''
    x1s and x2s contain matching points
    x1s - 2D image points in the first image in homogenous coordinates (3xN)
    x2s - 2D image points in the second image in homogenous coordinates (3xN)
    '''
    # Your code here
    x1s = pflat(x1s)
    x2s = pflat(x2s)
    
    x1, y1, z1 = x1s
    x2, y2, z2 = x2s
    
    N = x1.shape[0]
    M = np.zeros((N, 9))
    
    # Form M as in lecture 6
    M[:, 0] = x2 * x1     
    M[:, 1] = x2 * y1     
    M[:, 2] = x2 * z1      
    M[:, 3] = y2 * x1      
    M[:, 4] = y2 * y1      
    M[:, 5] = y2 * z1      
    M[:, 6] = z2 * x1      
    M[:, 7] = z2 * y1      
    M[:, 8] = z2 * z1     
    
    # Solve using svd
    U, S, Vt = np.linalg.svd(M)
    v = Vt[-1, :]                 # Extract smallest singular value
    F_tilde = v.reshape(3, 3)  
    
    sigma_min = S[-1]
    residual = np.linalg.norm(M @ v)

    return F_tilde, sigma_min, residual

def enforce_fundamental(F_approx):
    '''
    F_approx - Approximate Fundamental matrix (3x3)
    '''
    # Your code here
    
    # Solve by eqn 20 in lecture 6
    U, S, Vt = np.linalg.svd(F_approx)
    
    # Set smallest singular value, sigma3=0
    S[2] = 0.0
    
    # Reconstruct the enforced F
    F_rank2 = U @ np.diag(S) @ Vt

    return F_rank2

def compute_epipolar_errors(F, x1s, x2s):
    '''
    x1s and x2s contain matching points
    x1s - 2D image points in the first image in homogenous coordinates (3xN)
    x2s - 2D image points in the second image in homogenous coordinates (3xN)
    F - Fundamental matrix (3x3)
    '''
    # Your code here
    x1s = pflat(x1s)
    x2s = pflat(x2s)
    
    l2 = F @ x1s
    a,b,c = l2
    x,y,_ = x2s
    
    d = np.abs(a*x + b*y + c)/np.sqrt(a**2 + b**2)
    
    return d
    
def enforce_essential(E_approx):
    '''
    E_approx - Approximate Essential matrix (3x3)
    '''
    # Your code here
    U, S, Vt = np.linalg.svd(E_approx)
    
    # Force two equal non-zero singular values and the third zero
    S_enforced = np.diag([1.0, 1.0, 0.0])
    
    E = U @ S_enforced @ Vt

    return E

def triangulate_all_points(P1, P2, x1, x2):
    N = x1.shape[1]
    Xs = np.zeros((4, N))
    
    for j in range(N):
        Xs[:, j] = triangulate_point_DLT(P1, P2, x1[:, j], x2[:, j])
        
    return Xs

def count_points_in_front(P1, P2, Xs):
    X1p = P1 @ Xs   
    X2p = P2 @ Xs   
    
    Z1 = X1p[2,:]*np.sign(Xs[3,:])
    Z2 = X2p[2,:]*np.sign(Xs[3,:])
    
    in1 = Z1 > 0
    in2 = Z2 > 0
    tot_point_in_front = np.logical_and(in1, in2)
    
    return np.sum(tot_point_in_front)


def plot_points_2D(points, ax=None, color='yellow', size=5):
    if ax is None:
        ax = plt.gca()
    x = points[0, :]
    y = points[1, :]
    ax.scatter(x, y, c=color, s=size)
    ax.set_aspect('equal')

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
