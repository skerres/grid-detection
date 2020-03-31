import numpy as np
import matplotlib.pyplot as plt
from line_detection_helper import *


def compute_line_normalform(I_rgb):
    """
        detect lines in an image, return them in normalform (rho + theta)
    """
    edge_threshold = 0.02
    blur_sigma     = 1
    line_threshold = 20
    bins           = 500
    window_size = 11

    rho_max   = +np.linalg.norm(I_rgb.shape)
    rho_min   = -rho_max
    theta_min = -np.pi
    theta_max = +np.pi

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

def plot_line_detection(I_rgb, peak_theta, peak_rho):

    plt.figure(figsize=[6,8])
    plt.subplot(211)
    plt.imshow(I_rgb)

    plt.subplot(212)
    plt.imshow(I_rgb)
    plt.xlim([0, I_rgb.shape[1]])
    plt.ylim([I_rgb.shape[0], 0])
    for i in range(len(peak_theta)):
        draw_line(peak_theta[i], peak_rho[i], color='yellow')
    plt.tight_layout()
    plt.savefig('data/line_detection_seq1_img0300.png')
    plt.show()

def plot_hough_transform_histogram(I_rgb, H):
    plt.figure(figsize=[6,8])
    plt.subplot(211)
    plt.imshow(I_rgb)
    plt.subplot(212)    
    rho_max   = +np.linalg.norm(I_rgb.shape)
    rho_min   = -rho_max
    theta_min = -np.pi
    theta_max = +np.pi    
    plt.imshow(H, extent=[theta_min, theta_max, rho_min, rho_max], aspect='auto')
    plt.xlabel('$\\theta$ (radians)')
    plt.ylabel('$\\rho$ (pixels)')
    plt.colorbar(label='Votes')
    plt.title('Hough transform histogram')
    plt.show()

path = '/home/sebastian/Studium/Robotic_Vision_TTK_4255/project/grid-detection/data/defished_seq1/'
filename = 'img0300.jpg'
filepath = path + filename
I_rgb      = plt.imread(filepath)
peak_rho, peak_theta, _ = compute_line_normalform(I_rgb)
plot_line_detection(I_rgb, peak_theta, peak_rho)

