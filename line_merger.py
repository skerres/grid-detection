import numpy as np
# from line_detection_helper import compute_endpoints
# class Point:
#     """ Point class represents and manipulates x,y coords. """

#     def __init__(self):
#         """ Create a new point at the origin """
#         self.x = 0
#         self.y = 0

#     def dist(self, p):
#         "compute the distance to other Point"
#         return np.sqrt(np.power((p.x - self.x), 2) + np.power((p.y - self.y), 2))

RHO_INV = 0
THETA_INV = -100

def compute_endpoints(rho1, theta1, x_min, x_max, y_min, y_max):
    if(np.abs(theta1) > np.pi * 45/180 and np.abs(theta1) < np.pi * 135/180 or (np.abs(theta1) > np.pi * 225/180 and np.abs(theta1) < np.pi * 315/180)):
        x1 = x_min
        y1 = (rho1 - x1 * np.cos(theta1))/np.sin(theta1)

        x2 = x_max
        y2 = (rho1 - x2 * np.cos(theta1))/np.sin(theta1)
    else: 
        y1 = y_min
        x1 = (rho1 - y1 * np.sin(theta1))/np.cos(theta1)

        y2 = y_max
        x2 = (rho1 - y2 * np.sin(theta1))/np.cos(theta1)

    return x1, x2, y1, y2

def merge_related_lines(img, rho, theta, threshold = 100):
    """
    Merge similiar lines. The lines are in normalform.
    """
    x_min = 0
    y_min = 0
    x_max = img.shape[1]
    y_max = img.shape[0]
    rho = np.reshape(rho, (rho.shape[0], 1))
    theta = np.reshape(theta, (theta.shape[0], 1))
    lines = np.concatenate((rho, theta), axis = 1)

    for current in lines:
        rho1 = current[0]
        theta1 = current[1]
        x1, x2, y1, y2 = compute_endpoints(rho1, theta1, x_min, x_max, y_min, y_max)
        p1_curr = [x1, y1]
        p2_curr = [x2, y2]

    for current in lines: 
        if(current[0] == RHO_INV and current[1] == THETA_INV):
            continue
        rho1 = current[0]
        theta1 = current[1]
        x1, x2, y1, y2 = compute_endpoints(rho1, theta1, x_min, x_max, y_min, y_max)
        p1_curr = [x1, y1]
        p2_curr = [x2, y2]
        for compare in lines:
            if np.array_equal(compare, current) or (compare[0] == RHO_INV and compare[1] == THETA_INV):
                continue
            rho1 = compare[0]
            theta1 = compare[1]
            x1, x2, y1, y2 = compute_endpoints(rho1, theta1, x_min, x_max, y_min, y_max)
            p1_comp = np.array((x1, y1))
            p2_comp = np.array((x2, y2))
            if(np.linalg.norm(p1_curr - p1_comp) < threshold and np.linalg.norm(p2_curr - p2_comp) < threshold):
                #Average rho
                current[0] = (current[0] + compare[0]) / 2 

                #If the difference between the thetas is about 2 * pi or pi, 
                #one theta must be shifted by 2*pi to keep the direction of the line
                diff = np.max((current[1] - compare[1], compare[1] - current[1]))
                if(np.abs(diff - 2 * np.pi) < 0.2 ):
                    current[1] = current[1] + 2 * np.pi                     
                elif(np.abs(diff - np.pi) < 0.2):
                    current[1] = current[1] + np.pi                     

                current[1] = (current[1] + compare[1]) / 2 
                compare[0] = RHO_INV
                compare[1] = THETA_INV
    rho = lines[:, 0]
    theta = lines[:, 1]
    doubled_lines_ind = np.where(np.logical_and(rho == RHO_INV, theta == THETA_INV))
    rho = np.delete(rho, doubled_lines_ind)            
    theta = np.delete(theta, doubled_lines_ind) 
    return rho, theta           
