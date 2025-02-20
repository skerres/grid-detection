import numpy as np
import matplotlib.pyplot as plt
from line_detection_helper import *

import cv2


def compute_line_normalform(I_rgb):
    """
        detect lines in an image, return them in normalform (rho + theta)
    """
    edge_threshold = 0.01               
    blur_sigma     = 1
    line_threshold = 15
    bins           = 500
    window_size = 13

    rho_max   = +np.linalg.norm(I_rgb.shape)
    rho_min   = -rho_max
    theta_min = 0
    theta_max = 2 * np.pi    
    # theta_max = +np.pi    
    # theta_min = -np.pi

    I_rgb      = I_rgb/255.0
    I_gray     = rgb2gray(I_rgb)
    I_blur     = blur(I_gray, blur_sigma)
    Iu, Iv, Im = central_difference(I_blur)
    u,v,theta  = extract_edges(Iu, Iv, Im, edge_threshold)

    rho = u*np.cos(theta) + v*np.sin(theta)

    histrange = [[theta_min, theta_max], [rho_min, rho_max]]
    H, _, _ = np.histogram2d(theta, rho, bins=bins, range=histrange)
    H = H.T # Make rows be rho and columns be theta

    peak_rows,peak_cols = extract_peaks(H, window_size=window_size, threshold=line_threshold)
    peak_rho = rho_min + (rho_max - rho_min)*(peak_rows + 0.5)/bins
    peak_theta = theta_min + (theta_max - theta_min)*(peak_cols + 0.5)/bins

    return peak_rho, peak_theta, H

def plot_base_image_and_lines(I_rgb, peak_rho, peak_theta):
    """
        creates a figure with two plots, one only the image and on the image and on top the lines
    """
    plt.figure(figsize=[6,8])
    plt.subplot(211)
    plt.imshow(I_rgb)
    plt.subplot(212)
    plt.imshow(I_rgb)
    plot_lines(I_rgb, peak_rho, peak_theta)
    plt.tight_layout()
    plt.show()

def plot_compare_image_lines(I_rgb, rho1, theta1, rho2, theta2, save_path = "", title1 = "Before merging lines", title2 = "After merging lines"):
     """
        creates a figure with two plots comparing two line plots. Each plot contains the image and on top lines
        I_rgb: np image
        rho1, theta1: first group of lines in normalform
        rho2, theta2: second group of lines in normalform
    """   
    plt.figure(figsize=[6,8])
    plt.subplot(211)
    plt.imshow(I_rgb)
    plt.title(title1)
    plot_lines(I_rgb, rho1, theta1)
    plt.subplot(212)
    plt.imshow(I_rgb)
    plot_lines(I_rgb, rho2, theta2)
    plt.title(title2)
    plt.tight_layout()
    if save_path is not "":
        plt.savefig(save_path)
        plt.close()

def plot_intersection_image(I_rgb, rho, theta, intersections):
    """
        creates a plot with an image, the lines and the intersection poitns
        I_rgb: numpy array, image
        rho, theta: line in normalform
        intersections; np.array((N,2)) intersections in normalform

    """
    plt.imshow(I_rgb)
    plot_lines(I_rgb, rho, theta)
    plt.scatter(intersections[:, 0], intersections[:, 1], marker='x', color = 'red', s = 100, linewidths = 15)
    plt.tight_layout()

def plot_lines(I_rgb, peak_rho, peak_theta):
    """
        plot lines in normalform
    """
    plt.xlim([0, I_rgb.shape[1]])
    plt.ylim([I_rgb.shape[0], 0])
    for i in range(len(peak_theta)):
        draw_line(peak_rho[i], peak_theta[i], color='yellow')


def plot_hough_transform_histogram(I_rgb, H):
    """
        plots a hough transform histogram
        I_rgb: np.array, image
        H: Hough transform matrix
    """
    plt.figure(figsize=[6,8])
    plt.subplot(211)
    plt.imshow(I_rgb)
    plt.subplot(212)    
    rho_max   = +np.linalg.norm(I_rgb.shape)
    rho_min   = -rho_max
    theta_max = 2 * np.pi  
    theta_min = 0
    # theta_max = +np.pi    
    # theta_min = -np.pi
    plt.imshow(H, extent=[theta_min, theta_max, rho_min, rho_max], aspect='auto')
    plt.xlabel('$\\theta$ (radians)')
    plt.ylabel('$\\rho$ (pixels)')
    plt.colorbar(label='Votes')
    plt.title('Hough transform histogram')
    plt.show()



def shift_rho_theta(rho, theta):
    """
        shift all entries for theta and rho, that rho > 0 and 0 <= theta <= 2 * pi
    """
    rho_shift = []
    theta_shift = []
    for rho1, theta1 in zip(rho, theta):
        if rho1 < 0: 
            rho1 = rho1 * -1 
            theta1 = theta1 + np.pi
        rho_shift.append(rho1)
        theta_shift.append(theta1)
            
    return np.array(rho_shift), np.array(theta_shift)
    # theta_pos = theta[(np.where(theta > 0)]
    # rho = rho[np.where(theta < 0)] * -1
    # theta = theta[np.where(theta < 0)] + np.pi

    
def load_camera_parameters():
    """
    load camera parameters f, cx, cy and distortion coefficient k
            and return camera matrix K and distortion matrix D
    """

    intrinsics = np.loadtxt('data/intrinsics.txt', comments='%')

    f,cx,cy,k = intrinsics
    K = np.zeros((3,3))
    K[0, 0] = f 
    K[1, 1] = f 
    K[0, 2] = cx 
    K[1, 2] = cy 
    K[2, 2] = 1
    D = np.zeros(4)
    D[0] = k

    return K, D