import matplotlib.pyplot as plt 
import numpy as np
import os
import pickle 

from line_detection import compute_line_normalform, plot_base_image_and_lines, shift_rho_theta, plot_compare_image_lines, plot_intersection_image, load_camera_parameters
from line_merger import merge_related_lines
from line_intersection import compute_intersection_points, select_intersections
from homography_estimation import estimate_camera_position, compute_angle, plot_trajectory, POS_INV, draw_grid
from fisheye import draw_fisheye_grid_image

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
            # skips image with number smaller then given image number
            # image_number = int(file[3:7])
            # if image_number < 20:
            #     continue

            #setup all necessary pathes
            input_directory_fished_image = 'data/' + str(input_directory[-5:-1]) + '/' + file
            filepath = input_directory + file
            save_path_grid = output_directory_grid + file
            save_path_intersection = output_directory_intersection + file
            save_path_fisheye_grid = output_directory_fisheye_grid + file

            #read image, expects that it is already defished
            I_rgb      = plt.imread(filepath)
            #load the camera parameters
            K, _ = load_camera_parameters()
            #detects all edges in the image
            peak_rho, peak_theta, _ = compute_line_normalform(I_rgb)
            #convert all edges into certain interval
            rho, theta = shift_rho_theta(peak_rho, peak_theta)
            #merge similar lines
            rho, theta = merge_related_lines(I_rgb, rho, theta)
            #compute the intersections of all lines
            intersections = compute_intersection_points(rho, theta)
            #only use the intersections which satisfy the necessary conditions
            intersections = select_intersections(intersections, I_rgb.shape, rho, theta)
            #save image showing the intersections
            plot_intersection_image(I_rgb, rho, theta, intersections)
            plt.savefig(save_path_intersection)
            plt.close()
            
            #estimate the camera position and compute the grid in the image frame
            valid, estimated_position, grid_image_frame = estimate_camera_position(intersections, K, position)

            if valid:
                #plot and save an image showing the line intersections and the grid for a rectilinear image
                plot_intersection_image(I_rgb, rho, theta, intersections)
                draw_grid(grid_image_frame)
                plt.savefig(save_path_grid)
                plt.close()
                position = estimated_position.position
                with open(pickle_path_trajectory, 'wb') as handle:
                    pickle.dump(np.array(trajectory), handle)
                #draw the grid on the corresponding fisheye image
                draw_fisheye_grid_image(input_directory_fished_image, grid_image_frame)
                plt.savefig(save_path_fisheye_grid)
                plt.close()

    #plot and save the trajectory of the camera
    plot_trajectory(pickle_path_trajectory, output_directory_trajectory)


def on_single_image():
    # pass
    path = 'data/defished_seq3/'
    filename = 'img0010.jpg'
    filepath = path + filename  
    I_rgb      = plt.imread(filepath)
    K, _ = load_camera_parameters()
    peak_rho, peak_theta, _ = compute_line_normalform(I_rgb)
    shift_rho, shift_theta = shift_rho_theta(peak_rho, peak_theta)
    rho, theta = shift_rho_theta(peak_rho, peak_theta)
    rho, theta = merge_related_lines(I_rgb, shift_rho, shift_theta)
    intersections = compute_intersection_points(rho, theta)
    print(intersections)


    intersections = select_intersections(intersections, I_rgb.shape, rho, theta)
    # print(intersections) 
    
    plot_intersection_image(I_rgb, rho, theta, intersections)
    valid, estimated_position, grid_image_frame = estimate_camera_position(intersections, K, POS_INV)
    draw_grid(grid_image_frame)


    plt.show()

if __name__ == "__main__":
    # on_single_image()      
    base_dir = 'data/'
    input_dir = 'defished_seq3/'
    output_dir_grid = 'estimate_grid_seq3/'
    output_dir_trajectory = 'trajectory_seq3/'
    output_directory_fisheye_grid = 'fisheye_grid_seq3/'
    output_dir_intersection = 'intersection_seq3/'
    pickle_file_name = 'seq3_foo.pickle'
    perform_line_detection_on_directory(
        base_dir + input_dir, 
        base_dir + pickle_file_name,
        base_dir + output_dir_grid,
        base_dir + output_dir_intersection,
        base_dir + output_directory_fisheye_grid,
        base_dir + output_dir_trajectory)
