import numpy as np 

def group_lines(rho, theta, group_count = 10):
    """
        group lines that in each group the angle between the lines are equally spaced.
        rho, theta: line in normalform
        group_count: amount of groups

    """
    interval_size = np.pi / group_count
    groups = [None] * group_count
    for i, (rho1, theta1) in enumerate(zip(rho, theta)):
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
        if groups[group_ind] is None:
            groups[group_ind] = np.array((rho1, theta1, i))
        else:
            groups[group_ind] = np.append(groups[group_ind], np.array((rho1, theta1, i)))
        
    for i in range(len(groups)):
        group = groups[i]
        if group is not None:
            groups[i] = np.reshape(group, (-1, 3))

    groups_only_entries = []
    for i in range(len(groups)):
        if groups[i] is not None:
            groups_only_entries.append(groups[i])
    return groups_only_entries

def compute_intersection_points(rho, theta):
    """
    Finds the intersections between different group of lines.
    Lines are grouped according to their angle. If the slope of the lines are sufficiently different, they are put in different groups.
    As no lines in different groups can be parallel, they must intersect. 
    rho, theta: lines in normalform
    """

    lines = group_lines(rho, theta)
    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i+1:]:
            for line1 in group:
                for line2 in next_group:
                    x0, y0, index1, index2 = intersection(line1, line2)
                    intersections.append((x0, y0, index1, index2)) 
                    # intersections.append((x0, y0))
    intersections = np.array(intersections)
    intersections = np.reshape(intersections, (-1, 4))
    # intersections = np.reshape(intersections, (-1, 2))
    return intersections

def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1 = line1[0]
    theta1 = line1[1]
    index1 = line1[2]

    rho2 = line2[0]
    theta2 = line2[1]
    index2 = line2[2]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0 = np.round(x0, 1)
    y0 = np.round(y0, 1)
    return x0, y0, index1, index2

def select_intersections(intersections, shape, rho, theta):
    """
        select the intersection points which are considered helpful for the computation of the grid.
        They must several conditions:
        - They must be within the image
        - The angle between the lines should be within an interval close to a rectilinear angle (np.pi / 2 or np.pi * 3 / 2),
        the parameter INTERVAL_DEG defines what is the middle of the interval in relation to a rectilinear angle 
        TOL_DEG describes how big the tolerance around this interval is, that the angle is still considered valid. 
    """
    
    TOL_DEG = 7
    INTERVAL_DEG = 8.6
    u_min = 0
    u_max = shape[1]
    v_min = 0
    v_max = shape[0]
    u_bounds = np.logical_and(intersections[:, 0] > u_min, intersections[:, 0] < u_max)
    v_bounds = np.logical_and(intersections[:, 1] > v_min, intersections[:, 1] < v_max)
    intersections = intersections[np.where(np.logical_and(u_bounds, v_bounds))]
    intersection_within_tol = []
    # return intersections
    for intersect in intersections:
        tol = TOL_DEG * np.pi / 180
        interval_deg = INTERVAL_DEG * np.pi / 180
        i = int(intersect[2])
        j = int(intersect[3])

        if np.abs(np.abs(theta[i] - theta[j]) - np.pi/2 - interval_deg) < tol or \
                np.abs(np.abs(theta[i] - theta[j]) - np.pi/2 + interval_deg) < tol or \
                np.abs(np.abs(theta[i] - theta[j]) - 3 * np.pi/2 - interval_deg) < tol or \
                np.abs(np.abs(theta[i] - theta[j]) - 3 * np.pi/2 + interval_deg) < tol:
            intersection_within_tol.append(True)
        else:
            intersection_within_tol.append(False)

    valid_intersections = intersections[intersection_within_tol]
    valid_intersections = valid_intersections[:, 0:2]
    valid_intersections = np.vstack(valid_intersections)
    valid_intersections = np.reshape(valid_intersections, (-1, 2))
    return valid_intersections[:, 0:2].astype(float)

