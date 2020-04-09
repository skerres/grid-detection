import numpy as np
import matplotlib.pyplot as plt
from line_detection_helper import *

import cv2


def compute_line_normalform(I_rgb):
    """
        detect lines in an image, return them in normalform (rho + theta)
    """
    edge_threshold = 0.02
    blur_sigma     = 1
    line_threshold = 20
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
    plt.figure(figsize=[6,8])
    plt.subplot(211)
    plt.imshow(I_rgb)
    plt.subplot(212)
    plt.imshow(I_rgb)
    plot_lines(I_rgb, peak_rho, peak_theta)
    plt.tight_layout()
    plt.show()

def plot_compare_image_lines(I_rgb, rho1, theta1, rho2, theta2, save_path = "", title1 = "Before merging lines", title2 = "After merging lines"):
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
    plt.figure(figsize=[6,8])
    plt.imshow(I_rgb)
    plot_lines(I_rgb, rho, theta)
    plot_intersections(intersections)
    plt.tight_layout()


def plot_intersections(intersections):
    plt.scatter(intersections[:, 0], intersections[:, 1], marker='x', color = 'red', s = 100, linewidths = 15)

def plot_lines(I_rgb, peak_rho, peak_theta):
    plt.xlim([0, I_rgb.shape[1]])
    plt.ylim([I_rgb.shape[0], 0])
    for i in range(len(peak_theta)):
        draw_line(peak_rho[i], peak_theta[i], color='yellow')


def plot_hough_transform_histogram(I_rgb, H):
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

# path = '/home/sebastian/Studium/Robotic_Vision_TTK_4255/project/grid-detection/data/defished_seq1/'
# filename = 'img0300.jpg'
# filepath = path + filename
# I_rgb      = plt.imread(filepath)
# peak_rho, peak_theta, _ = compute_line_normalform(I_rgb)
# plot_line_detection(I_rgb, peak_rho, peak_theta)


def shift_rho_theta(rho, theta):
    """
        shift all entries for theta and rho, that rho > 0 and 0 <= theta <= pi
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


def test(input_dir, filename):
    filepath = input_dir + filename
    # Read image
    img = cv2.imread(filepath, cv2.IMREAD_COLOR) # road.png is the filename
    # Convert the image to gray-scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the edges in the image using canny detector
    edges = cv2.Canny(gray, 50, 200)
    # Detect points that form a line
    lines = cv2.HoughLinesP(edges, rho = 1, theta = np.pi/180 * 0.1, threshold = 150, minLineLength=30, maxLineGap=250)
    # Draw lines on the image   
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
    # Show result
    cv2.imshow("Result Image", img)
    cv2.imwrite(filename[0:7] + '_opencv.jpg', img) 

if __name__ == "__main__":
    input_dir = 'data/defished_seq3/'
    filename = 'img0002.jpg'
    test(input_dir, filename)