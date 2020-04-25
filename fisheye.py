import matplotlib.pyplot as plt
import numpy as np
import os
import pickle 

import cv2
from homography_estimation import draw_grid

def camera_to_fisheye(X, Y, Z, intrinsics):
    """
    X,Y,Z:      3D point in camera coordinates. Z-axis pointing forward,
                Y-axis pointing down, and X-axis point to the right.
    intrinsics: Fisheye intrinsics [f, cx, cy, k]
    """
    f,cx,cy,k = intrinsics
    theta = np.arctan2(np.sqrt(X*X + Y*Y), Z)
    phi = np.arctan2(Y, X)
    r = f*theta*(1 + k*theta*theta)
    u = cx + r*np.cos(phi)
    v = cy + r*np.sin(phi)
    return [u, v]

def camera_to_rectilinear(X, Y, Z, intrinsics):
    """
    projects camera coordinates using a rectilinear projection
    X,Y,Z:      3D point in camera coordinates. Z-axis pointing forward,
                Y-axis pointing down, and X-axis point to the right.
    intrinsics: Fisheye intrinsics [f, cx, cy, k]
    """
    f,cx,cy, _ = intrinsics
    u = cx + f * X/Z
    v = cy + f * Y/Z
    return [u,v]

def rectilinear_to_fisheye(u_rect, v_rect, K, D):
    """
    converts rectiliinear camera coordinates into fisheye camera coordinates
    u_rect, v_rect:     rectilinear 2d camera coordinates
    K, D:               camera intrinsics matrix and distortion parameters
    """
    f = K[0,0]
    cx = K[0,2]
    cy = K[1,2]
    k = D[0]
    phi = np.arctan((v_rect - cy)/(u_rect - cx))
    theta = np.arctan((u_rect - cx)/(f * np.cos(phi)))
    u_dst = cx + f * theta * (1 + k * theta**2) * np.cos(phi)
    v_dst = cy + f * theta * (1 + k * theta**2) * np.sin(phi)
    return u_dst, v_dst

def undistort_fisheye_image(img_src, K, D):
    """
        returns an undistorted fisheye image
    img_src: input image to be distorted
    K: camera matrix
    D: distortion matrix
    """
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K,
        D,
        np.eye(3),
        K,
        # (int(img_src.shape[1] * 1.8), int(img_src.shape[0] * 1.1)),
        (int(img_src.shape[1]), int(img_src.shape[0])),
        cv2.CV_16SC2
    )

    img_rect = cv2.remap(
        img_src,
        map1,
        map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )
    return img_rect


def draw_fisheye_grid(grid_image_frame, grid_color = 'yellow'):
    """
        draw a grid on the currently active plot using the values on the corners of the grid_image_frame
        grid_image_frame: numpy array (7x7x2), 2d grid coordinates in the image frame for a canonical 7x7 grid
    """
    def draw_interpolated_line(u_rect_start, v_rect_start, u_rect_finish, v_rect_finish, K, D, points = 4):
        """
            draws a line on a fisheye image by projecting a rectilinear line onto a fisheye image using interpolation
            u_rect_start, v_rect_start: start point in 2d image coordinates
            u_rect_finish, v_rect_finish: finish point 2d image coordinates
            K: camera intriniscs
            D: distortion parameters
            points: amount of interpolation points
        """
            interpolation = np.zeros((points,2))
            factor = points / (points - 1) #factor to smoothly connect all points
            for i in range(interpolation.shape[0]):
                interpolation[i, 0] = (u_rect_start * (points - i * factor) + u_rect_finish * i * factor) / points
                interpolation[i, 1] = (v_rect_start * (points - i * factor) + v_rect_finish * i * factor) / points

            for i in range(interpolation.shape[0]):
                u_dst, v_dst = rectilinear_to_fisheye(interpolation[i, 0], interpolation[i, 1], K, D)
                interpolation[i, 0] = u_dst
                interpolation[i, 1] = v_dst
            plt.plot(interpolation[:, 0], interpolation[:, 1], color = grid_color)


    K, D = load_camera_parameters()

    for i in range(7):
        for j in range(6):
            u_rect_top, v_rect_top = grid_image_frame[j][i] 
            u_rect_bot, v_rect_bot = grid_image_frame[j+1][i]
            draw_interpolated_line(u_rect_top, v_rect_top, u_rect_bot, v_rect_bot, K, D)

    for i in range(7):
        for j in range(6):
            u_rect_left, v_rect_left = grid_image_frame[i][j] 
            u_rect_right, v_rect_right = grid_image_frame[i][j+1]
            draw_interpolated_line(u_rect_left, v_rect_left, u_rect_right, v_rect_right, K, D)


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

def test_point_fisheye_distortion():
    """
        run a test to visualize point fisheye distortion
    """
    intrinsics = np.loadtxt('data/intrinsics.txt', comments='%')
    X = np.linspace(-2,+2,10)
    Y = np.linspace(-1,+1,5)
    uv_fish = []
    uv_rect = []
    xy_cv = []
    for X_i in X:
        for Y_i in Y:
            Z_i = 2.0
            uv_fish.append(camera_to_fisheye(X_i, Y_i, Z_i, intrinsics))
            uv_rect.append(camera_to_rectilinear(X_i, Y_i, Z_i, intrinsics))
            xy = np.array([X_i/Z_i, Y_i/Z_i])
            xy_cv.append(xy)

    xy_cv = np.array(xy_cv)
    if xy_cv.ndim == 2:
                xy_cv = np.expand_dims(xy_cv, 0)
    uv_cv_dist = cv2.fisheye.distortPoints(xy_cv, K, D)
    uv_cv_undist = cv2.fisheye.undistortPoints(uv_cv_dist, K, D)


    uv_fish = np.array(uv_fish)   
    uv_rect = np.array(uv_rect)   
    plt.scatter(uv_fish[:,0], uv_fish[:,1])
    plt.scatter(uv_rect[:,0], uv_rect[:,1])
    # plt.scatter(uv_cv[0,:,0], uv_cv[0,:,1])
    plt.axis('scaled')
    plt.xlim([0, 1280])
    plt.ylim([720, 0])
    plt.grid()
    plt.show()


def test_image_fisheye_distortion():
    """
        test the undistortion of a fish eye image for a given file input
    """
    img_src = cv2.imread('./data/seq1/img0110.jpg')
    K, D = load_camera_parameters()
    img_rect = undistort_fisheye_image(img_src, K, D)
    imgplot2 = plt.imshow(img_rect)
    plt.show()

def convert_fished_folder(input_directory_path, output_directory_path):
    """
    defishes all images in a folder and saves them to an output directory
    """
    K, D = load_camera_parameters()
    for file in os.listdir(input_directory_path):
        if file.endswith(".jpg"):
            print(file)
            img_src = cv2.imread(input_directory_path + "/" + file)
            img_rect = undistort_fisheye_image(img_src, K, D)
            cv2.imwrite(output_directory_path + "/" + file, img_rect)


def draw_fisheye_grid_image(filepath, grid_image_frame):
    """
        draws a grid on a fisheye image and plots the image
        filepath: path to fisheye image
        grid_image_frame: numpy array (7x7x2), 2d grid coordinates in the image frame for a canonical 7x7 grid
    """

    img = plt.imread(filepath)
    plt.imshow(img)
    # picklepath = 'data/img0100_grid_image_frame.pickle'
    # with open(picklepath, 'rb') as handle:
    #     grid_image_frame_rect = pickle.load(handle)
    # grid_image_frame_dst = compute_distorted_grid_from_undistorted(grid_image_frame_rect)
    # print(grid_image_frame_dst)
    # draw_grid(grid_image_frame_rect, grid_color='red')
    draw_fisheye_grid(grid_image_frame, grid_color='yellow')
    plt.xlim([0, img.shape[1]])
    plt.ylim([img.shape[0], 0])
    plt.title("Projection of grid onto fisheye image with two interpolation points")

if __name__ == '__main__':
    picklepath = 'data/img0100_grid_image_frame.pickle'
    filepath = 'data/seq1/img0100.jpg'
    with open(picklepath, 'rb') as handle:
        grid_image_frame = pickle.load(handle)

    draw_fisheye_grid_image(grid_image_frame, filepath)
    # convert_fished_folder('/home/sebastian/Studium/Robotic_Vision_TTK_4255/project/grid-detection/data/seq1', '/home/sebastian/Studium/Robotic_Vision_TTK_4255/project/grid-detection/data/defished_seq1_2')
# test_image_fisheye_distortion()
