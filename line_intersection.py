import numpy as np 
import matplotlib.pyplot as plt

def group_lines(rho, theta):
    group_count = 10
    interval_size = np.pi / group_count
    groups = [None] * 10
    for rho1, theta1 in zip(rho,theta):
        # print(theta1)
        if np.abs(theta1 - np.pi * 2) < 0.0001:
            theta_clamped = theta1 - np.pi - 0.0001
        elif theta1 > 2 * np.pi:
            theta_clamped = theta1 -  2 * np.pi
            theta1 = theta1 - 2 * np.pi
        elif theta1 > np.pi:
            theta_clamped = theta1 - np.pi
        else:
            theta_clamped = theta1
        group_ind = (int(theta_clamped/interval_size))
        # print((theta_clamped/interval_size))
        # print(group_ind)
        if groups[group_ind] is None:
            groups[group_ind] = np.array((rho1, theta1))
        else:
            groups[group_ind] = np.append(groups[group_ind], np.array((rho1, theta1)))
            
    for i in range(len(groups)):
        group = groups[i]
        if group is not None:
            groups[i] = np.reshape(group, (-1, 2))

    groups = (np.array(groups))
    groups_only_entries = []
    for i in range(len(groups)):
        if groups[i] is not None:
            groups_only_entries.append(groups[i])
    return groups_only_entries


def compute_intersection_points(rho, theta, shape):
    """Finds the intersections between groups of lines."""

    lines = group_lines(rho, theta)
    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i+1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(intersection(line1, line2)) 
    intersections = np.array(intersections)
    intersections = np.reshape(intersections, (-1, 2))

    u_min = 0
    u_max = shape[1]
    v_min = 0
    v_max = shape[0]
    u_bounds = np.logical_and(intersections[:,0] > u_min, intersections[:,0] < u_max)
    v_bounds = np.logical_and(intersections[:,1] > v_min, intersections[:,1] < v_max)
    intersections = intersections[np.where(np.logical_and(u_bounds, v_bounds))]

    return intersections

def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1= line1[0]
    theta1 = line1[1]
    rho2 = line2[0]
    theta2 = line2[1]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = (np.round(x0, 1)), (np.round(y0, 1))
    if x0 == 332:
        print("Line 1: " + str(line1))
        print("Line 2: " + str(line2))
        print("A: " + str(A))
        print("b: " + str(b))
    return [[x0, y0]]

