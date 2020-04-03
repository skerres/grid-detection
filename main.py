import matplotlib.pyplot as plt 
import numpy as np
import os
import pickle 

from line_detection import compute_line_normalform, plot_base_image_and_lines, shift_rho_theta, plot_compare_image_lines, plot_intersection_image
from line_merger import merge_related_lines
from line_intersection import compute_intersection_points
from fisheye import load_camera_parameters
from homography_estimation import estimate_camera_position

def perform_line_detection_on_directory(input_directory_path, output_directory_path):
    for file in os.listdir(input_directory_path):
        if file.endswith(".jpg"):
            print(file)
            filepath = input_directory_path + "/" + file
            save_path= output_directory_path + "/" + file
            I_rgb      = plt.imread(filepath)
            K, _ = load_camera_parameters()
            peak_rho, peak_theta, _ = compute_line_normalform(I_rgb)
            shift_rho, shift_theta = shift_rho_theta(peak_rho, peak_theta)
            rho, theta = merge_related_lines(I_rgb, shift_rho, shift_theta)
            intersections = compute_intersection_points(rho, theta, I_rgb.shape)
            # plot_compare_image_lines(I_rgb, peak_rho, peak_theta, rho, theta, save_path= output_directory_path + "/" + file)
            plot_intersection_image(I_rgb, rho, theta, intersections)
            estimate_camera_position(intersections, K)
            plt.savefig(save_path)
            plt.close()

def main():
    # pass
    path = '/home/sebastian/Studium/Robotic_Vision_TTK_4255/project/grid-detection/data/defished_seq1/'
    filename = 'img0004.jpg'
    filepath = path + filename  
    I_rgb      = plt.imread(filepath)
    K, _ = load_camera_parameters()
    peak_rho, peak_theta, _ = compute_line_normalform(I_rgb)
    shift_rho, shift_theta = shift_rho_theta(peak_rho, peak_theta)
    rho, theta = merge_related_lines(I_rgb, shift_rho, shift_theta)
    intersections = compute_intersection_points(rho, theta, I_rgb.shape)

    plot_intersection_image(I_rgb, rho, theta, intersections)
    estimate_camera_position(intersections, K)
    plt.show()

if __name__ == "__main__":
    # main()
    perform_line_detection_on_directory('/home/sebastian/Studium/Robotic_Vision_TTK_4255/project/grid-detection/data/defished_seq1', '/home/sebastian/Studium/Robotic_Vision_TTK_4255/project/grid-detection/data/estimate_grid_seq1')
