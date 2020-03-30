import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from undistort_fisheye import undistort_image
import scipy.optimize as op
import pickle
import cv2

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
        (img_src.shape[1], img_src.shape[0]),
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

test_image_fisheye_distortion()