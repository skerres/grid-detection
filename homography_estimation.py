import matplotlib.pyplot as plt
import numpy as np

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
    
def compute_grid_object_frame():
    grid_object_frame = np.zeros((7,7,2,2))
    for i in range(7):
        for j in range(7):
            grid_object_frame[j][i] = ((i - 3, i - 3), (j - 3, j - 3))
    return grid_object_frame

def compute_grid_image_frame(uv, K, grid_object_frame):
    xy = (uv - K[0:2,2])/np.array([K[0,0], K[1,1]])
    x_dir = ((xy[1,:] - xy[0,:]) + (xy[3,:] - xy[2,:])) / 2
    y_dir = ((xy[2,:] - xy[0,:]) + (xy[3,:] - xy[1,:])) / 2

    
    # compute the coordinates of the grid intersection lines in camera frame
    grid_camera_frame = grid_object_frame * np.array((x_dir, y_dir))
    grid_camera_frame = np.sum(grid_camera_frame, axis = 2)

    # compute the coordinates of the grid intersection lines in image frame (pixel coordinates)
    grid_image_frame = grid_camera_frame * np.array([K[0,0], K[1,1]]) + uv[0,:]
    return grid_image_frame, x_dir, y_dir, xy
  
def compute_loss(grid_image_frame, intersections, x_dir , y_dir):
    loss = 0
    LOSS_INV = 99999
    max_diff_to_orthogonal_angle = 88
    max_angle_fac = np.cos(max_diff_to_orthogonal_angle * np.pi / 180)
    if np.abs(np.dot(x_dir, y_dir)) > max_angle_fac * np.prod(np.linalg.norm((x_dir, y_dir), axis = 1)):
        return LOSS_INV
    for intersection in intersections:
        # print("intersection " + str(intersection))
        # print("min dist " + str(np.min(np.linalg.norm(intersection - grid_image_frame, axis = 2))))
        # print()
        min_dist = np.min(np.linalg.norm(intersection - grid_image_frame, axis = 2))
        loss = loss + min_dist
    return loss

def ransac(intersections, K):
    grid_object_frame = compute_grid_object_frame()
    result = []
    for _ in range(10000):
        rand_ind = (np.random.rand(4) * intersections.shape[0]).astype(int)
        if np.unique(rand_ind).size < 4:
            continue
        uv = np.array(intersections[rand_ind, :])
        grid_image_frame, x_dir, y_dir, _ = compute_grid_image_frame(uv, K, grid_object_frame)
        loss = compute_loss(grid_image_frame, intersections, x_dir, y_dir)
        result.append((loss, rand_ind))
    result = np.array(result)
    min_los_ind = np.argmin(result[:,0])
    return result[min_los_ind, 1:5]


def estimate_camera_position(intersections, K):
    XY = np.array([(0,0),(1,0),(0,1),(1,1)])
    min_los_ind = ransac(intersections, K)
    uv = np.array(intersections[min_los_ind[0], :])
    grid_object_frame = compute_grid_object_frame()
    grid_image_frame, _, _, xy = compute_grid_image_frame(uv, K, grid_object_frame)

    # print(intersections)
    # print(XY)
    # Convert pixel coordinates to normalized image coordinates
    # print("uv[0,:] " + str(uv[0,:]))
    # print("uv[1,:] " + str(uv[1,:]))
    # print("uv[2,:] " + str(uv[2,:]))
    # print("uv[3,:] " + str(uv[3,:]))
    # print(loss)
    # print(uv)
    # print(xy)
    H = estimate_H(xy, XY)
    T1,T2 = decompose_H(H)
    T = choose_solution(T1, T2)
    draw_grid(grid_image_frame)
    # print(K.shape)
    # print(H.shape)
    # print(H)
    # print(T)
    # print(XY.shape)
    # print(xy.shape)

def draw_grid(grid_image_frame):
   # image_coords = (grid_coords) * np.array([K[0,0], K[1,1]]) + K[0:2,2]
    # print("[0][0] " + str(image_coords[0][0]))
    # print("[6][0] " + str(image_coords[6][0]))
    # print("[0][6] " + str(image_coords[0][6]))#

    for i in range(7):
        u_top, v_top = grid_image_frame[0][i] 
        u_bot, v_bot = grid_image_frame[6][i]
        # print("(u_top, v_top) " + str(image_coords[0][i])) 
        # print("(u_bot, v_bot)" + str(image_coords[6][i]))
        plt.plot([u_bot, u_top], [v_bot, v_top], color = 'red')
    # plt.plot(image_coords[6][0], image_coords[0][0], color = 'red')
    # plt.plot(image_coords[6][1], image_coords[0][1], color = 'red')
    # plt.plot(image_coords[6][2], image_coords[0][2], color = 'red')
    # plt.plot(image_coords[6][3], image_coords[0][3], color = 'red')
    # plt.plot(image_coords[6][4], image_coords[0][4], color = 'red')

    for i in range(7):
        u_left, v_left = grid_image_frame[i][0] 
        u_right, v_right = grid_image_frame[i][6] 
        plt.plot([u_left, u_right], [v_left, v_right], color = 'red')


