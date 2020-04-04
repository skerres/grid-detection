import matplotlib.pyplot as plt 
import numpy as np
import os
import pickle 

from line_detection import compute_line_normalform, plot_base_image_and_lines, shift_rho_theta, plot_compare_image_lines, plot_intersection_image
from line_merger import merge_related_lines
from line_intersection import compute_intersection_points
from fisheye import load_camera_parameters
from homography_estimation import estimate_camera_position, compute_angle, plot_trajectory

def perform_line_detection_on_directory(input_directory, output_directory_grid, output_directory_trajectory):
    for file in sorted(os.listdir(input_directory)):
        if file.endswith(".jpg"):
            print(file)
            image_number = int(file[3:7])
            filepath = input_directory + "/" + file
            save_path_grid = output_directory_grid + "/" + file
            save_path_trajectory = output_directory_trajectory + "/" + file
            I_rgb      = plt.imread(filepath)
            K, _ = load_camera_parameters()
            peak_rho, peak_theta, _ = compute_line_normalform(I_rgb)
            shift_rho, shift_theta = shift_rho_theta(peak_rho, peak_theta)
            rho, theta = merge_related_lines(I_rgb, shift_rho, shift_theta)
            intersections = compute_intersection_points(rho, theta, I_rgb.shape)
            # plot_compare_image_lines(I_rgb, peak_rho, peak_theta, rho, theta, save_path= output_directory_path + "/" + file)
            plot_intersection_image(I_rgb, rho, theta, intersections)
            valid, position = estimate_camera_position(intersections, K)
            plt.savefig(save_path_grid)
            plt.close()
            if valid:
                plot_trajectory(position)
            plt.savefig(save_path_trajectory)
            plt.close()
            # if image_number == 5:
            #     return

def main():
    # pass
    path = '/home/sebastian/Studium/Robotic_Vision_TTK_4255/project/grid-detection/data/defished_seq1/'
    filename = 'img0107.jpg'
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
    base_dir = '/home/sebastian/Studium/Robotic_Vision_TTK_4255/project/grid-detection/data/'
    input_dir = 'defished_seq1'
    output_dir_grid = 'estimate_grid_seq1'
    output_dir_trajectory = 'trajectory_seq1'
    perform_line_detection_on_directory(base_dir + input_dir, base_dir + output_dir_grid, base_dir + output_dir_trajectory)
    # a = np.array((1.01, 1))
    # b = np.array((2, 2))
    # print(compute_angle(a,b))