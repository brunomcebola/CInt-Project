from ctypes import pointer
import os
import random
import argparse
from turtle import st
import pandas as pd
import matplotlib.pyplot as plt

from deap import base
from deap import tools
from deap import creator
from math import *

import numpy as np

# CONSTANTS
CXPB = 0.5
LOAD = 1000
MUTPB = 0.2
STALL = inf
EVALS = 10000
COSTUMERS = 10
POPULATION = 40

# FUNCTIONS
def parseCmdLine():
    parser = argparse.ArgumentParser(
        prog="Distribute Orders",
        description="Using genetic algorithms to determine the best route to deliver products to costumers",
        add_help=True,
    )

    parser.add_argument("dist_file", type=str, help="File containing the matrix with the distances between costumers")
    parser.add_argument("coord_file", type=str, help="File with the coordinates of the costumers and the warehouse")
    parser.add_argument("orders_file", type=str, help="File with the orders of the costumers")

    parser.add_argument("--mut-prob", "-mp", type=int, dest="mutation_prob", default=MUTPB)
    parser.add_argument("--pop-size", "-p", type=int, dest="population_size", default=POPULATION)
    parser.add_argument("--cross-prob", "-cp", type=int, dest="crossover_prob", default=CXPB)
    parser.add_argument("--max-evals", "-e", type=int, dest="max_evals", default=EVALS)
    parser.add_argument("--nb-costumers", "-c", type=int, dest="nb_costumers", default=COSTUMERS)
    parser.add_argument("--max-stall", "-s", type=int, dest="max_stall", default=STALL)
    parser.add_argument("--max-load", "-l", type=int, dest="max_load", default=LOAD)

    parser.add_argument("--seed", type=int, dest="seed", default=0)
    parser.add_argument("--verbose", dest="verbose", action="store_true", default=False)
    parser.add_argument("--iterate", dest="iterate", action="store_true", default=False)

    args = parser.parse_args()

    return args.__dict__


def readCSV(file_name):
    return pd.read_csv(file_name)


def getRoute(route, orders_matrix, max_load):
    load = 0
    final_route = [-1]

    for location in route:
        load += int(orders_matrix.loc[orders_matrix["Customer"] == location + 1]["Orders"])

        if load > max_load:
            final_route.append(-1)
            load = 0

        final_route.append(location)

    final_route.append(-1)

    return final_route


def getDistance(route, dist_matrix):
    distance = 0

    for prev_place, current_place in zip(route[:-1], route[1:]):
        distance += dist_matrix[prev_place + 1][current_place + 1]

    return distance


def evalRoute(individual, orders_matrix, dist_matrix, max_load):

    route = getRoute(individual, orders_matrix, max_load)

    distance = getDistance(route, dist_matrix)

    return (distance,)


def algorithm(toolbox, population_size, mutation_prob, crossover_prob, max_stall, max_evals, seed, verbose=False):
    random.seed(seed)

    pop = toolbox.population(n=population_size)  # type: ignore

    if verbose:
        print("-- Generation 0 --")

    fitnesses = list(map(toolbox.evaluate, pop))  # type: ignore
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    fits = [ind.fitness.values[0] for ind in pop]

    g = 0
    stall = 0
    min_path = inf

    while stall < max_stall and g < floor((max_evals / population_size) - 1):
        if verbose:
            print("-- Generation %i --" % (g + 1))

        offspring = list(map(toolbox.clone, toolbox.select(pop, population_size)))  # type: ignore

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < crossover_prob:
                toolbox.mate(child1, child2)  # type: ignore

                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutation_prob:
                toolbox.mutate(mutant)  # type: ignore
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)  # type: ignore
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring

        fits = [ind.fitness.values[0] for ind in pop]

        if min(fits) < min_path:
            min_path = min(fits)
            stall = 0
        else:
            stall = stall + 1

        if verbose:
            print("dist %d" % min_path)

        g += 1

    if verbose:
        print("-- End of evolution --")

    return pop, fits


def getStats(fits, pop_size):

    mean = sum(fits) / pop_size
    sum2 = sum(x * x for x in fits)
    std = abs(sum2 / pop_size - mean**2) ** 0.5

    return max(fits), min(fits), mean, std


def plotRoute(route, coord_matrix):
    X = []
    Y = []
    L = []

    for i in range(len(route)):
        route[i] += 1

    for location in route:
        X.append(int(coord_matrix["X"][location]))
        Y.append(int(coord_matrix["Y"][location]))
        L.append(location)

    print("Route: %s" % route)

    plt.plot(X, Y, "-o")

    for i, txt in enumerate(L):
        plt.annotate(txt, (X[i], Y[i]))

    plt.show()
# Heuristic

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


# PARSE CMD LINE ARGUMENTS
params = parseCmdLine()

# READ DATA FROM CSV
if not os.path.exists(params["dist_file"]):
    raise SystemExit("[DISTANCES] File '%s' does not exist" % params["dist_file"])

if not os.path.exists(params["coord_file"]):
    raise SystemExit("[COORDINATES] File '%s' does not exist" % params["coord_file"])

if not os.path.exists(params["orders_file"]):
    raise SystemExit("[ORDERS] File '%s' does not exist" % params["orders_file"])

coord_matrix = readCSV(params["coord_file"])
orders_matrix = readCSV(params["orders_file"])
dist_matrix = readCSV(params["dist_file"])

dist_matrix = dist_matrix.iloc[:, 1:]
dist_matrix.columns = dist_matrix.columns.astype(int)

# REGISTER TOOLBOX
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)  # type: ignore

toolbox = base.Toolbox()

toolbox.register("location", random.sample, range(params["nb_costumers"]), params["nb_costumers"])

toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.location)  # type: ignore

toolbox.register("population", tools.initRepeat, list, toolbox.individual)  # type: ignore

toolbox.register("mate", tools.cxPartialyMatched)

toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)

toolbox.register("select", tools.selTournament, tournsize=2)

toolbox.register("evaluate", evalRoute, orders_matrix=orders_matrix, dist_matrix=dist_matrix, max_load=params["max_load"])

# ALGORITHM
if params["iterate"]:
    best_route = []
    best_fit = inf
    avg_max = 0
    avg_min = 0
    avg_mean = 0
    avg_std = 0

    for seed in range(30):
        pop, fits = algorithm(
            toolbox,
            params["population_size"],
            params["mutation_prob"],
            params["crossover_prob"],
            params["max_stall"],
            params["max_evals"],
            seed,
            params["verbose"],
        )

        route = getRoute(tools.selBest(pop, 1)[0], orders_matrix, params["max_load"])

        max_fit, min_fit, mean, std = getStats(fits, params["population_size"])

        # TODO
        if params["verbose"]:
            print("Min: %d" % avg_min)
            print("Avg max: %d" % avg_max)
            print("Avg std: %d" % avg_std)
            print("Avg mean: %d" % avg_mean)
            print()

        if min_fit < best_fit:
            best_fit = min_fit
            best_route = route

        avg_max += max_fit
        avg_min += min_fit
        avg_std += std
        avg_mean += mean

    avg_max /= 30
    avg_min /= 30
    avg_std /= 30
    avg_mean /= 30

    print("Avg min: %d" % avg_min)
    print("Avg max: %d" % avg_max)
    print("Avg std: %d" % avg_std)
    print("Avg mean: %d" % avg_mean)

    plotRoute(best_route, coord_matrix)
else:
    pop, fits = algorithm(
        toolbox,
        params["population_size"],
        params["mutation_prob"],
        params["crossover_prob"],
        params["max_stall"],
        params["max_evals"],
        params["seed"],
        params["verbose"],
    )

    route = getRoute(tools.selBest(pop, 1)[0], orders_matrix, params["max_load"])

    max_fit, min_fit, mean, std = getStats(fits, params["population_size"])

    print("Min: %d" % min(fits))
    print("Max: %d" % max(fits))
    print("Std: %d" % std)
    print("Mean: %d" % mean)

    plotRoute(route, coord_matrix)
