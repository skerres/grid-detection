import matplotlib.pyplot as plt
import numpy as np

LOSS_INV = 99999
MAX_GRID_ANGLE_DIFF = 8
POS_INV = (-100, -100, -100)

def estimate_H(xy, XY):
    A = []
    n = XY.shape[0]
    for i in range(n):
        X,Y = XY[i]
        x,y = xy[i]
        A.append(np.array([X,Y,1, 0,0,0, -X*x, -Y*x, -x]))
        A.append(np.array([0,0,0, X,Y,1, -X*y, -Y*y, -y]))
    A = np.array(A)
    U,s,VT = np.linalg.svd(A)
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

  
def compute_loss(grid_image_frame, intersections, x_dir, y_dir, T, prev_pos):
    grid_loss = 0
    position_loss = 0
    POSITION_LOSS_FACTOR = 10
    # if np.abs(compute_angle(x_dir, y_dir) - 90) > MAX_GRID_ANGLE_DIFF:
    #     return LOSS_INV
    for intersection in intersections:
        min_dist = np.min(np.linalg.norm(intersection - grid_image_frame, axis = 2))
        grid_loss = grid_loss + min_dist
    if not prev_pos == POS_INV:
        position_loss = POSITION_LOSS_FACTOR * np.linalg.norm(np.linalg.norm(T[0:3, 3] - np.array(prev_pos)))
    return grid_loss + position_loss, grid_loss, position_loss

def draw_grid(grid_image_frame):

    for i in range(7):
        u_top, v_top = grid_image_frame[0][i] 
        u_bot, v_bot = grid_image_frame[6][i]
        plt.plot([u_bot, u_top], [v_bot, v_top], color = 'red')

    for i in range(7):
        u_left, v_left = grid_image_frame[i][0] 
        u_right, v_right = grid_image_frame[i][6] 
        plt.plot([u_left, u_right], [v_left, v_right], color = 'red')


def plot_trajectory(trajectory):
    plt.figure(figsize=[6,8])
    plt.scatter(trajectory[0], trajectory[1])
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])

def compute_camera_transformation(xy, XY):
    H = estimate_H(xy, XY)
    T1,T2 = decompose_H(H)
    T = choose_solution(T1, T2)
    return T

def compute_grid_image_frame(uv, K, grid_object_frame):
    xy = (uv - K[0:2,2])/np.array([K[0,0], K[1,1]])
    x_dir1 = xy[1,:] - xy[0,:]
    x_dir2 = xy[3,:] - xy[2,:]
    y_dir1 = xy[2,:] - xy[0,:]
    y_dir2 = xy[3,:] - xy[1,:]
    # print("compute_angle(x_dir1, x_dir2) " + str(compute_angle(x_dir1, x_dir2)))
    # print("compute_angle(y_dir1, y_dir2) " + str(compute_angle(y_dir1, y_dir2)))
    # print("compute_angle(x_dir1, y_dir1) " + str(compute_angle(x_dir1, y_dir1)))
    # print("compute_angle(x_dir2, y_dir2) " + str(compute_angle(x_dir2, y_dir2)))
    if compute_angle(x_dir1, x_dir2) > 2 * MAX_GRID_ANGLE_DIFF and compute_angle(y_dir1, y_dir2) > 2 * MAX_GRID_ANGLE_DIFF:
        return False, 0, 0, 0, 0

    # print(compute_angle(x_dir1, y_dir1))
    # print()
    if np.abs(compute_angle(x_dir1, y_dir1) - 90) > MAX_GRID_ANGLE_DIFF and np.abs(compute_angle(x_dir1, y_dir1) - 180) > MAX_GRID_ANGLE_DIFF:
        return False, 0, 0, 0, 0

    if np.abs(compute_angle(x_dir2, y_dir2) - 90) > MAX_GRID_ANGLE_DIFF and np.abs(compute_angle(x_dir2, y_dir2) - 180) > MAX_GRID_ANGLE_DIFF:
        return False, 0, 0, 0, 0
    # print(compute_angle(x_dir1, y_dir1))
    # print(compute_angle(x_dir2, y_dir2))
    # print()

    x_dir = ((xy[1,:] - xy[0,:]) + (xy[3,:] - xy[2,:])) / 2
    y_dir = ((xy[2,:] - xy[0,:]) + (xy[3,:] - xy[1,:])) / 2
    
    # compute the coordinates of the grid intersection lines in camera frame
    grid_camera_frame = grid_object_frame * np.array((x_dir, y_dir))
    grid_camera_frame = np.sum(grid_camera_frame, axis = 2)

    # compute the coordinates of the grid intersection lines in image frame (pixel coordinates)
    grid_image_frame = grid_camera_frame * np.array([K[0,0], K[1,1]]) + uv[0,:]
    return True, grid_image_frame, x_dir, y_dir, xy

def ransac(intersections, K, XY, prev_pos):
    i = 0
    grid_object_frame, grid_object_frame_1x2 = compute_grid_object_frame()
    result = []
    for _ in range(10000):
        rand_ind = (np.random.rand(4) * intersections.shape[0]).astype(int)
        if np.unique(rand_ind).size < 4:
            continue
        uv = np.array(intersections[rand_ind, :])
        valid, grid_image_frame, x_dir, y_dir, xy = compute_grid_image_frame(uv, K, grid_object_frame)
        if not valid:
            continue
        i = i +1
        T = compute_camera_transformation(xy, XY)
        loss, grid_loss, pos_loss = compute_loss(grid_image_frame, intersections, x_dir, y_dir, T, prev_pos)
        result.append((loss, rand_ind, grid_loss, pos_loss))
    result = np.array(result)
    # print(result)
    if result.size == 0:
        return False, 0
    min_los_ind = np.argmin(result[:,0])
    # print(result)
    # print("Final Loss: "+ str(result[min_los_ind, 0]) + " " + str(result[min_los_ind, 2:4]))
    # print("Result " + str(result[min_los_ind, 1]))
    return True, result[min_los_ind, 1]

def compute_grid_object_frame():
    grid_object_frame_2x2 = np.zeros((7,7,2,2))
    grid_object_frame_1x2 = np.zeros((7,7,2))
    for i in range(7):
        for j in range(7):
            grid_object_frame_2x2[j][i] = ((i - 3, i - 3), (j - 3, j - 3))
            grid_object_frame_1x2[j][i] = (i - 3, j - 3)
    return grid_object_frame_2x2, grid_object_frame_1x2

def object_frame_to_pixel_coordinate(XY, R, t, K):
    XYZ = np.concatenate((XY, np.zeros((XY.shape[0], 1))), axis = 1)
    # print("XYZ " + str(XYZ))
    # print("XYZ.shape " + str(XYZ.shape))
    XYZ_camera = np.dot(R, XYZ.T).T + t.T
    # print("XYZ_camera " + str(XYZ_camera))
    # print("XYZ_camera.shape " + str(XYZ_camera.shape))
    x = np.reshape(XYZ_camera[:, 0] / XYZ_camera[:, 2], (XY.shape[0], 1))
    y = np.reshape(XYZ_camera[:, 1] / XYZ_camera[:, 2], (XY.shape[0], 1))
    # print("x " + str(x))
    # print("y " + str(y))
    xy = np.concatenate((x, y), axis = 1)
    uv = xy * np.array([K[0,0], K[1,1]]) +  K[0:2,2]
    # print("uv: " + str(uv))
    return uv

def estimate_camera_position(intersections, K, prev_pos):
    XY = np.array([(0,0),(1,0),(0,1),(1,1)])
    grid_object_frame, grid_object_frame_1x2 = compute_grid_object_frame()
    # intersections = intersections[[8,:], [1,:], [6,:], [0,:]]
    # uv = intersections[np.array((8,1,6,0))]
    # print(uv)
    # valid, grid_image_frame, x_dir, y_dir, _ = compute_grid_image_frame(uv, K, grid_object_frame)
    # print(grid_image_frame)
    # draw_grid(grid_image_frame)
    # return
    valid, min_los_ind = ransac(intersections, K, XY, prev_pos)
    if not valid: 
        return False, POS_INV
    uv = np.array(intersections[min_los_ind, :])
    grid_object_frame, grid_object_frame_1x2 = compute_grid_object_frame()
    # valid, grid_image_frame, _, _, xy, T = compute_grid_image_frame(uv, K, grid_object_frame, prev_position)
    valid, grid_image_frame, _, _, xy = compute_grid_image_frame(uv, K, grid_object_frame)
    if valid:
        T = compute_camera_transformation(xy, XY)
        # print(grid_object_frame_1x2)
        # print(np.reshape(grid_object_frame_1x2, (-1, 2)))
        uv2 = object_frame_to_pixel_coordinate(np.reshape(grid_object_frame_1x2, (-1, 2)), T[0:3,0:3], np.reshape(T[0:3,3], (3, 1)), K)
        uv2 = np.reshape(uv2, (7,7,2))
        position = ((T[0][3], T[1][3], T[2][3]))
        # print(uv)
       
        # print("trajectory[0] " + str(position[0]))
        # print("trajectory[1] " + str(position[1]))
        # print("trajectory[2] " + str(T[2][3]))
        # print("yaw / psi: " + str(-np.arctan2(T[0][2], T[1][2])))
        # print("pitch / theta: " + str(np.arccos(T[2][2])))
        # print("roll / phi: " + str(-np.arctan2(T[2][0], T[2][1])))
        draw_grid(uv2)
        return True, position
    return False, POS_INV
