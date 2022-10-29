from math import *
import string
import numpy as np
import random
random.seed(5)


points = [
    (1,1), # 0    1º
    (-1,-1), # 1  5º
    (1,-1), # 2   7º
    (-1,1), # 3   3º
    (-1,0), # 4   4º
    (0,-1), # 5   6º
    (1,0), # 6    8º
    (0,1) # 7     2º
]

# SOLUTION = [0,7,3,4,1,5,2,6]

centroid= (0,0)

def findangles(centroid, list_of_points, focus_point):
    # for not it used the first point, but we can make it to use the left,
    # just need to search it and "remove it" from the list
    angles = []

    for point in list_of_points:
        vec_a = (centroid[0] - list_of_points[focus_point][0],centroid[1] - list_of_points[focus_point][1] )
        vec_b = (centroid[0] - point[0],centroid[1] - point[1])

        diff_angle = degrees(atan2(vec_b[1], vec_b[0])) - degrees(atan2(vec_a[1], vec_a[0]))

        if  diff_angle < 0:
            diff_angle += 360              

        angles.append( diff_angle )

    return angles

def counter_clock_wise_heuristic(list_of_points, maximum_number):
    # list_of_points has to contain WareHouse as well

    # Point that will be our focus
    focus_point = random.randrange(maximum_number)

    # Calculate centroid to have 2 common points everytime
    centroid_x = centroid_y = 0

    for index, point in enumerate(list_of_points):
        if index == 0:
            continue
        centroid_x += point[0]
        centroid_y += point[1]

    centroid = (centroid_x/(len(list_of_points)-1),centroid_y/(len(list_of_points)-1) )
    # List of points no including the warehouse
    angles = findangles(centroid, list_of_points, focus_point)
    # Order it
    angles = np.array(angles)

    angles_sorted = np.argsort(angles)

    # Return our selection
    return angles_sorted


print(counter_clock_wise_heuristic(points, 8))