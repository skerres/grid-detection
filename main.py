import matplotlib.pyplot as plt 
import numpy as np
import os
import pickle 

from line_detection import compute_line_normalform, plot_base_image_and_lines, shift_rho_theta, plot_compare_image_lines, plot_intersection_image, load_camera_parameters
from line_merger import merge_related_lines
from line_intersection import compute_intersection_points, select_intersections
from homography_estimation import estimate_camera_position, compute_angle, plot_trajectory, POS_INV, draw_grid
from fisheye import draw_fisheye_grid_image
from trajectory import plot_trajectory

def perform_line_detection_on_directory(
        input_directory, 
        pickle_path_trajectory, 
        output_directory_grid, 
        output_directory_intersection, 
        output_directory_fisheye_grid, 
        output_directory_trajectory):
    trajectory = []
    position = POS_INV
    for file in sorted(os.listdir(input_directory)):
        if file.endswith(".jpg"):
            image_number = int(file[3:7])
            input_directory_fished_image = 'data/' + str(input_directory[-5:-1]) + '/' + file

            print(file)
            filepath = input_directory + file
            save_path_grid = output_directory_grid + file
            save_path_intersection = output_directory_intersection + file
            save_path_fisheye_grid = output_directory_fisheye_grid + file
            I_rgb      = plt.imread(filepath)
            K, _ = load_camera_parameters()
            peak_rho, peak_theta, _ = compute_line_normalform(I_rgb)
            shift_rho, shift_theta = shift_rho_theta(peak_rho, peak_theta)
            rho, theta = merge_related_lines(I_rgb, shift_rho, shift_theta)
            intersections = compute_intersection_points(rho, theta)
            intersections = select_intersections(intersections, I_rgb.shape, rho, theta)
            # plot_compare_image_lines(I_rgb, peak_rho, peak_theta, rho, theta, save_path= output_directory_path + "/" + file)
            plot_intersection_image(I_rgb, rho, theta, intersections)
            plt.savefig(save_path_intersection)
            plt.close()
            plot_intersection_image(I_rgb, rho, theta, intersections)
            # continue
            valid, estimated_position, grid_image_frame = estimate_camera_position(intersections, K, position)
            # xy_camera_frame = np.array((position[0] / position[2], position[1] / position[2]))
            # xy_image_frame = xy_camera_frame * np.array([K[0,0], K[1,1]]) +  K[0:2,2]
            # plt.scatter(xy_image_frame[0], xy_image_frame[1], marker = '+', s = 50, c = "blue")

            if valid:
                draw_grid(grid_image_frame)
                plt.savefig(save_path_grid)
                plt.close()
                position = estimated_position.position
                # plot_trajectory(position)
                trajectory.append(estimated_position)
                # plt.savefig(save_path_trajectory)
                # plt.close()
                with open(pickle_path_trajectory, 'wb') as handle:
                    pickle.dump(np.array(trajectory), handle)
                draw_fisheye_grid_image(input_directory_fished_image, grid_image_frame)
                plt.savefig(save_path_fisheye_grid)
                plt.close()

    plot_trajectory(pickle_path_trajectory, output_directory_trajectory)


def main():
    # pass
    path = 'data/defished_seq2/'
    filename = 'img0030.jpg'
    filepath = path + filename  
    I_rgb      = plt.imread(filepath)
    K, _ = load_camera_parameters()
    peak_rho, peak_theta, _ = compute_line_normalform(I_rgb)
    shift_rho, shift_theta = shift_rho_theta(peak_rho, peak_theta)
    rho, theta = shift_rho_theta(peak_rho, peak_theta)
    rho, theta = merge_related_lines(I_rgb, shift_rho, shift_theta)
    intersections = compute_intersection_points(rho, theta)

    intersections = select_intersections(intersections, I_rgb.shape, rho, theta)
    # print(intersections)  
    plot_intersection_image(I_rgb, rho, theta, intersections)
    valid, estimated_position, grid_image_frame = estimate_camera_position(intersections, K, POS_INV)
    draw_grid(grid_image_frame)
    plt.show()

if __name__ == "__main__":
    # main()
    base_dir = 'data/'
    input_dir = 'defished_seq2/'
    output_dir_grid = 'estimate_grid_seq2/'
    output_dir_trajectory = 'trajectory_seq2/'
    output_directory_fisheye_grid = 'fisheye_grid_seq2/'
    output_dir_intersection = 'intersection_seq2/'
    pickle_file_name = 'seq2.pickle'
    perform_line_detection_on_directory(
        base_dir + input_dir, 
        base_dir + pickle_file_name,
        base_dir + output_dir_grid,
        base_dir + output_dir_intersection,
        base_dir + output_directory_fisheye_grid,
        base_dir + output_dir_trajectory)
