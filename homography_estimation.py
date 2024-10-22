import matplotlib.pyplot as plt
import numpy as np
import pickle

LOSS_INV = 99999 #Invalid Loss 
MAX_GRID_ANGLE_DIFF = 18 #Maximum angle difference for grid to be considered valid
POS_INV = np.array((-100, -100, -100)) #Invalid Estimated Position

class EstimatedPosition(object):
    """
        class which contains data fiels for the estimated camera position
        position:       np.vector((3,)), contains 3d position
        xy:             np.array((4,2) contains world coordinates of associated canonical tile
        x_dir:          np.array(2,), x direction of canoncial grid
        y_dir:          np.array(2,), y direction of canoncial grid
        euler_angle:    np.array((roll, pitch, yaw)), euler angles for rotation of camera

    """
    
    def __init__(self, position, xy, euler_angle):
        self.position = position
        self.xy = xy
        x_dir1 = xy[1,:] - xy[0,:]
        x_dir2 = xy[3,:] - xy[2,:]
        y_dir1 = xy[2,:] - xy[0,:]
        y_dir2 = xy[3,:] - xy[1,:]
        self.x_dir = (x_dir1 + x_dir2) / 2
        self.y_dir = (y_dir1 + y_dir2) / 2
        self.euler_angle = euler_angle

def estimate_H(xy, XY):
    A = []
    n = XY.shape[0]
    for i in range(n):
        X,Y = XY[i]
        x,y = xy[i]
        A.append(np.array([X,Y,1, 0,0,0, -X*x, -Y*x, -x]))
        A.append(np.array([0,0,0, X,Y,1, -X*y, -Y*y, -y]))
    A = np.array(A)
    _, _,VT = np.linalg.svd(A)
    h = VT[8,:]
    H = np.reshape(h, [3,3])
    # Alternatively we can explicitly construct H
    # H = np.array([
    #     [h[0], h[1], h[2]],
    #     [h[3], h[4], h[5]],
    #     [h[6], h[7], h[8]]
    # ])
    return H

def decompose_H(H):
    """
        decompose homography matrix into rotation matrix and translation matrix
        H: np.array((4,4)), homograhpy matrix
    """
    H *= 1.0/np.linalg.norm(H[:,0])
    r1 = H[:,0]
    r2 = H[:,1]
    r3 = np.cross(r1, r2) # note: r1 x r2 = -r1 x -r2 = r3
    t  = H[:,2]
    R1 = np.array([r1, r2, r3]).T
    R2 = np.array([-r1, -r2, r3]).T
    T1 = np.eye(4)
    T2 = np.eye(4)
    T1[:3,:3] = R1
    T1[:3,3] = t
    T2[:3,:3] = R2
    T2[:3,3] = -t
    return T1, T2

def choose_solution(T1, T2):
    # In this case the plane origin should always be in front of the
    # camera, so we can test the sign of the z-translation component
    # to select the correct transformation.
    z1 = T1[2,3]
    z2 = T2[2,3]
    if z1 > z2:
        return T1
    else:
        return T2

def compute_angle(x, y):
    """
    x: numpy vector
    y: numpy vector
    returns the angle between the two vectors
    cos(phi) = (| x @ y.T|) / (|x| * |y|) 


    """
    
    phi = np.arccos(np.dot(x,y) / (np.linalg.norm(x) * np.linalg.norm(y)))
    return phi * 180 / np.pi

def compute_angle_360(x, y):
    """
    x: numpy vector
    y: numpy vector
    returns the angle between the two vectors
    cos(phi) = (| x @ y.T|) / (|x| * |y|) 


    """
    dot = x[0]*y[0] + x[1] * y[1]      # dot product
    det = x[0]*y[1] - y[0]* x[1]      # determinant
    angle = np.arctan2(det, dot)  # atan2(y, x) or atan2(sin, cos)
    return angle
    # phi = np.arccos(np.abs(np.dot(x,y)) / (np.linalg.norm(x) * np.linalg.norm(y)))
    # if np.sign(np.dot(x,y)) < 0:
    #     return phi * 180 / np.pi + 180
    # return phi * 180 / np.pi

  
def compute_loss(grid_image_frame, intersections, T, prev_pos):
    """
        grid_image_frame: numpy array (7x7x2)
        intersections: nimpy array (nx2)
        T: numpy array (4x4)
        prev_pos: numpy vector
        Computes the loss between the grid represented by the lines detected by computer vision and the grid computed with the camera homography.
        The loss is a number describing how well the cv grid is described by the homography grid. 
        It consists of the grid_loss which is the sum of the norm of the cv and homography grid and 
        the position_loss which describes the difference to the previous position 
    """
    grid_loss = 0
    position_loss = 0
    POSITION_LOSS_FACTOR = 1000
    for intersection in intersections:
        min_dist = np.min(np.linalg.norm(intersection - grid_image_frame, axis = 2))
        grid_loss = grid_loss + min_dist
    if not np.array_equal(prev_pos, POS_INV):
        position_loss = POSITION_LOSS_FACTOR * np.linalg.norm(np.linalg.norm(T[0:3, 3] - np.array(prev_pos)))
    return grid_loss + position_loss, grid_loss, position_loss

def draw_grid(grid_image_frame, grid_color = 'red'):
    """
        grid_image_frame: numpy array (7x7x2)

        draw a grid on the currently active plot using the values on the corners of the grid_image_frame
    """

    for i in range(7):
        u_top, v_top = grid_image_frame[0][i] 
        u_bot, v_bot = grid_image_frame[6][i]
        plt.plot([u_bot, u_top], [v_bot, v_top], color = grid_color)

    for i in range(7):
        u_left, v_left = grid_image_frame[i][0] 
        u_right, v_right = grid_image_frame[i][6] 
        plt.plot([u_left, u_right], [v_left, v_right], [0, 0], color = grid_color)
    
    line = plt.plot([u_left, u_right], [v_left, v_right], [0, 0], color = grid_color)
    

def compute_camera_transformation(xy, XY):
    """
        estimate the transformation matrix from grid corners in one image with the associated real world grid
        xy: np.array((4,2)) coordinates of grid corners in image
        XY = np.array([(0,0),(1,0),(0,1),(1,1)])  associated grid. 
        The point described by the first row of the xy array corresponds to (0,0), second to (1,0), third to (0,1) and fourth to (1,1)  
    """
    H = estimate_H(xy, XY)
    T1,T2 = decompose_H(H)
    T = choose_solution(T1, T2)
    return T

def check_rectilinear_xy(xy):
    """
        check if the grid corresponding to the chosen grid corners is approximately rectilinear
        xy: np.array((4,2)) coordinates of grid corners in image

    """

    x_dir1 = xy[1,:] - xy[0,:]
    x_dir2 = xy[3,:] - xy[2,:]
    y_dir1 = xy[2,:] - xy[0,:]
    y_dir2 = xy[3,:] - xy[1,:]

    if compute_angle(x_dir1, x_dir2) > 2 * MAX_GRID_ANGLE_DIFF or compute_angle(y_dir1, y_dir2) > 2 * MAX_GRID_ANGLE_DIFF:
        return False

    if np.abs(compute_angle(x_dir1, y_dir1) - 90) > MAX_GRID_ANGLE_DIFF and np.abs(compute_angle(x_dir1, y_dir1) - 180) > MAX_GRID_ANGLE_DIFF:
        return False

    if np.abs(compute_angle(x_dir2, y_dir2) - 90) > MAX_GRID_ANGLE_DIFF and np.abs(compute_angle(x_dir2, y_dir2) - 180) > MAX_GRID_ANGLE_DIFF:
        return False
    return True

def ransac(intersections, K, XY, prev_pos):
    """
        perform a ransac loop to estimate the corner points of a tile which correspond to a 1mx1m canoncial grid 
        https://en.wikipedia.org/wiki/Random_sample_consensus
        intersections: np.array(N,)
    """
    grid_object_frame = compute_grid_object_frame()
    result = []
    for _ in range(20000):
        # create random indices corresponding to the row index chosen from the intersection 
        rand_ind = (np.random.rand(4) * intersections.shape[0]).astype(int)
        if np.unique(rand_ind).size < 4:
            continue
        # choose the intersections which correspond to the random indices
        uv = np.array(intersections[rand_ind, :])
        # convert the image coordinates into real word coordinates
        xy = (uv - K[0:2,2])/np.array([K[0,0], K[1,1]])

        # check if the grid satisfies the condition that it is approximately rectilinear
        valid = check_rectilinear_xy(xy)
        if not valid:
            continue
        # If the chosen intersections are already part of the result they don't need to be considered
        if len(result) > 2 and any(np.equal(np.stack(np.array(result)[:,1]),rand_ind).all(1)):
            continue

        #compute the camera transformation matrix
        T = compute_camera_transformation(xy, XY)
        #the z coordinate of the camera must be higher then 1.3m and lower then 2m
        if T[2][3] < 1.3 or T[2][3] > 2:
            continue 

        #compute the grid in the image frame
        grid_image_frame = compute_grid_image_frame(grid_object_frame, T, K)
        #compute the loss for the grid. The loss is an indicator for the quality of the grid
        loss, grid_loss, pos_loss = compute_loss(grid_image_frame, intersections, T, prev_pos)

        #append the associated grid in the list of results
        result.append((loss, rand_ind, grid_loss, pos_loss))
    result = np.array(result)
    # no valid results were found, return a flag that the computation was unsucessfull
    if result.size == 0:
        return False, 0

    # return the result with the lowest total los
    min_los_ind = np.argmin(result[:,0])

    return True, result[min_los_ind, 1]

def compute_grid_object_frame():
    """
        returns a np.array((7,7,2)) corresponding to the grid in the object frame
    """
    grid_object_frame = np.zeros((7,7,2))
    for i in range(7):
        for j in range(7):
            grid_object_frame[j][i] = (i - 3, j - 3)
    return grid_object_frame

def compute_grid_image_frame(grid_object_frame, T, K):
    """
        computes and returns a np.array((7,7,2)) corresponding to the grid in the image frame
        grid_object_frame: np.array((7,7,2)) corresponding to the grid in the object frame
        T: camera transformatinon matrix
        K: camera intrinsics matrix

    """

    grid_object_frame = np.reshape(grid_object_frame, (-1, 2))
    R = T[0:3,0:3]
    t = np.reshape(T[0:3,3], (3, 1))
    grid_object_frame = np.concatenate((grid_object_frame, np.zeros((grid_object_frame.shape[0], 1))), axis = 1)
    grid_camera_frame = np.dot(R, grid_object_frame.T).T + t.T
    x_camera_frame = np.reshape(grid_camera_frame[:, 0] / grid_camera_frame[:, 2], (grid_object_frame.shape[0], 1))
    y_camera_frame = np.reshape(grid_camera_frame[:, 1] / grid_camera_frame[:, 2], (grid_object_frame.shape[0], 1))
    xy_camera_frame = np.concatenate((x_camera_frame, y_camera_frame), axis = 1)
    grid_image_frame = xy_camera_frame * np.array([K[0,0], K[1,1]]) +  K[0:2,2]
    grid_image_frame = np.reshape(grid_image_frame, (7,7,2))

    return grid_image_frame

def estimate_camera_position(intersections, K, prev_pos):
    """
        estimates the camera position and the corner points of the detected grid
        intersections: np.array((N,2)) intersections of lines
        K: camera intrinsics
        prev_pos: previous camera position
    """
    XY = np.array([(0,0),(1,0),(0,1),(1,1)])
    grid_object_frame = compute_grid_object_frame()

    valid, min_los_ind = ransac(intersections, K, XY, prev_pos)
    if not valid: 
        return False, EstimatedPosition(POS_INV, np.zeros((4,2)), np.zeros((3,1))), np.zeros((7,7))
    uv = np.array(intersections[min_los_ind, :])
    xy = (uv - K[0:2,2])/np.array([K[0,0], K[1,1]])
    T = compute_camera_transformation(xy, XY)
    grid_image_frame = compute_grid_image_frame(grid_object_frame, T, K)

    roll = np.arctan2(T[2][0], T[2][1]) 
    pitch = np.arccos(T[2][2]) 
    yaw = np.arctan2(T[0][2], T[1][2]) 
    euler_angle = np.array((roll, pitch, yaw))
    position = np.array((T[0][3], T[1][3], T[2][3]))

    estimated_position = EstimatedPosition(position, xy, euler_angle)
    return True, estimated_position, grid_image_frame
