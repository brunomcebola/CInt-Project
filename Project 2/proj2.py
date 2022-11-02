import enum
import sys
import random
import itertools
from tabnanny import verbose
import matplotlib.pyplot as plt

from math import degrees, atan2, inf, dist

from deap.benchmarks.tools import hypervolume

from deap import base
from deap import tools
from deap import creator, algorithms
from copy import deepcopy

import aux_functions as aux
import numpy as np


def findangles(centroid, list_of_points, focus_point):
    # for not it used the first point, but we can make it to use the left,
    # just need to search it and "remove it" from the list
    angles = []

    for point in list_of_points:
        vec_a = (centroid[0] - list_of_points[focus_point][0], centroid[1] - list_of_points[focus_point][1])
        vec_b = (centroid[0] - point[0], centroid[1] - point[1])

        diff_angle = degrees(atan2(vec_b[1], vec_b[0])) - degrees(atan2(vec_a[1], vec_a[0]))

        if diff_angle < 0:
            diff_angle += 360

        angles.append(diff_angle)

    return np.array(angles)


def plotTrends(trends):
    for trend in trends:
        plt.plot(list(range(len(trend))), trend)

    plt.show()


def plotRoute(route, coordinates):
    route = deepcopy(route)

    X = []
    Y = []
    L = []

    for i in range(len(route)):
        route[i] += 1

    for location in route:
        X.append(int(coordinates[location][0]))
        Y.append(int(coordinates[location][1]))
        L.append(location)

    plt.plot(X, Y, "-o")

    for i, txt in enumerate(L):
        plt.annotate(txt, (X[i], Y[i]))

    plt.show()


class Algorithm:
    configuration = {
        "iterations": 30,
        #
        "pop_size": 40,
        "nb_costumers": 50,
        #
        "mut_prob": 0.2,
        "cross_prob": 0.5,
        #
        "max_load": 1000,
        "max_stall": inf,
        "max_evals": 10000,
        #
        "heuristic": False,
        "multi_objective": False,
    }

    # aux functions

    # Heuristic using angles
    def heuristic(self) -> list:
        # Point that will be our focus

        # focus_point = random.randrange(self.configuration["nb_costumers"])
        closest_point_dist = inf
        closest_point_id = 0

        for point_id in range(self.configuration["nb_costumers"]):
            if dist(self.coordinates[0], self.coordinates[point_id + 1]) < closest_point_dist:
                closest_point_id = point_id
                closest_point_dist = dist(self.coordinates[0], self.coordinates[point_id + 1])

        focus_point = closest_point_id

        # Calculate centroid to have 2 common points everytime
        centroid_x = centroid_y = 0

        for point in self.coordinates[1 : (self.configuration["nb_costumers"] + 1)]:
            centroid_x += point[0]
            centroid_y += point[1]

        centroid = (
            centroid_x / self.configuration["nb_costumers"],
            centroid_y / self.configuration["nb_costumers"],
        )


        # List of points no including the warehouse
        angles = findangles(centroid, self.coordinates[1 : (self.configuration["nb_costumers"] + 1)], focus_point)

        # Order it
        individual = np.argsort(angles)

        # Return our selection
        return list(individual)

    def getRoute(self, individual) -> list:
        load = 0
        route = [-1]

        for location in individual:
            load += int(self.orders[location + 1])

            if load > self.configuration["max_load"]:
                route.append(-1)
                load = 0

            route.append(location)

        route.append(-1)

        return route

    def getDistance(self, route) -> int:
        distance = 0

        for prev_place, current_place in zip(route[:-1], route[1:]):
            distance += self.distances[prev_place + 1][current_place + 1]

        return distance

    def getCost(self, route):
        cost = 0

        capacity = self.configuration["max_load"]

        for prev_place, current_place in zip(route[:-1], route[1:]):

            if capacity < self.orders[current_place + 1]:
                capacity = self.configuration["max_load"]

            cost += distances[prev_place + 1][current_place + 1] * capacity

            capacity -= self.orders[current_place + 1]

        return cost

    def evalSingle(self, individual):

        route = self.getRoute(individual)

        distance = self.getDistance(route)

        return (distance,)

    def evalRouteMulti(self, individual):

        route = self.getRoute(individual)

        distance = self.getDistance(route)

        cost = self.getCost(route)

        return (distance, cost)

    # setter functions

    def set_configuration(self, config: dict) -> None:
        for key in config:
            if key in self.configuration:
                self.configuration[key] = config[key]

    def set_orders(self, orders: list) -> None:
        self.orders = orders

    def set_distances(self, distances: list) -> None:
        self.distances = distances

    def set_coordinates(self, coordinates: list) -> None:
        self.coordinates = coordinates

    def set_toolbox(self) -> None:
        if self.configuration["heuristic"] == True:
            self.toolbox.register("location", self.heuristic)
        else:
            self.toolbox.register(
                "location", random.sample, range(self.configuration["nb_costumers"]), self.configuration["nb_costumers"]
            )

        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.location)  # type: ignore
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)  # type: ignore
        
        # This just makes some points go missing (???) 
        # self.toolbox.register("mate", tools.cxOnePoint)

        # This reaches a result a bit worse, BUT, the evolution is much better
        # self.toolbox.register("mate", tools.cxUniformPartialyMatched, indpb=0.01)
        
        # ORIGINAL
        self.toolbox.register("mate", tools.cxPartialyMatched)
        
        self.toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
        
        if self.configuration["multi_objective"] == True:
            self.toolbox.register("select", tools.selNSGA2)  # params: individuals, k (number of individuals to select)
            self.toolbox.register("evaluate", self.evalRouteMulti)
        else:
            self.toolbox.register("select", tools.selTournament, tournsize=2)
            self.toolbox.register("evaluate", self.evalSingle)

    # init class

    def __init__(self, config: dict, orders: list, distances: list, coordinates: list) -> None:      
        self.toolbox = base.Toolbox()

        if config["multi_objective"] == True:
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0))
        else:
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)  # type: ignore

        self.set_configuration(config)

        self.set_orders(orders)
        self.set_distances(distances)
        self.set_coordinates(coordinates)

        self.set_toolbox()

    # run algorithm

    def runSingle(self) -> tuple:
        global_min_fit = inf

        global_min_route = []
        global_min_trend = []
        global_avg_trend = []
        global_stats = {}

        # test algorithm for different seeds
        for seed in range(self.configuration["iterations"]):
            min_fit = inf
            min_route = []
            min_trend = []
            avg_trend = []
            stats = {}

            random.seed(seed)

            # generate initial population
            pop = self.toolbox.population(n=self.configuration["pop_size"])  # type: ignore

            # evaluate population
            fitnesses = list(map(self.toolbox.evaluate, pop))  # type: ignore
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit

            # store best info
            fits = [ind.fitness.values[0] for ind in pop]

            min_fit = min(fits)
            min_route = tools.selBest(pop, 1)[0]

            mean = sum(fits) / len(fits)
            std = abs((sum(x * x for x in fits)) / self.configuration["pop_size"] - mean**2) ** 0.5
            stats = {"min": min(fits), "max": max(fits), "mean": mean, "std": std}

            min_trend.append(min(fits))
            avg_trend.append(sum(fits) / len(fits))

            # offspring
            g = 0
            stall = 0
            while stall < self.configuration["max_stall"] and g < (
                self.configuration["max_evals"] - self.configuration["pop_size"]
            ):
                # generate offspring
                offspring = list(map(self.toolbox.clone, self.toolbox.select(pop, self.configuration["pop_size"])))  # type: ignore

                # mate
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < self.configuration["cross_prob"]:
                        self.toolbox.mate(child1, child2)  # type: ignore

                        del child1.fitness.values
                        del child2.fitness.values

                # mutate
                for mutant in offspring:
                    if random.random() < self.configuration["mut_prob"]:
                        self.toolbox.mutate(mutant)  # type: ignore
                        del mutant.fitness.values

                # evaluate offspring
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = map(self.toolbox.evaluate, invalid_ind)  # type: ignore
                g += len(invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                # replace population by offspring
                pop[:] = offspring

                # store best info
                fits = [ind.fitness.values[0] for ind in pop]

                if min(fits) < min_fit:
                    min_fit = min(fits)
                    min_route = tools.selBest(pop, 1)[0]

                    mean = sum(fits) / len(fits)
                    std = abs((sum(x * x for x in fits)) / self.configuration["pop_size"] - mean**2) ** 0.5
                    stats = {"min": min_fit, "max": max(fits), "mean": mean, "std": std}

                    stall = 0
                else:
                    stall += 1

                min_trend.append(min(fits))
                avg_trend.append(sum(fits) / len(fits))

            # stores best info for all seeds
            if min_fit < global_min_fit:
                global_min_fit = min_fit
                global_min_route = min_route
                global_min_trend = min_trend
                global_avg_trend = avg_trend
                global_stats = stats

        global_min_route = self.getRoute(global_min_route)

        return (global_min_route, global_min_trend, global_avg_trend, global_stats)

    def runMulti(self) -> tuple:
        
        # Needed variables
        mu = 40
        cxpb=0.7
        mutpb=0.3
        ngen=500

        # test algorithm for different seeds
        # for seed in range(self.configuration["iterations"]):
            
        random.seed(35)

        # generate initial population
        pop = self.toolbox.population(n=self.configuration["pop_size"])  # type: ignore

        hof = tools.ParetoFront()

        stats = tools.Statistics(lambda ind: ind.fitness.values)

        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)

        # Algorithm
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals', 'avg', 'std', 'min', 'hyper']

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

         # This is just to assign the crowding distance to the individuals
        # no actual selection is done
        pop = self.toolbox.select(pop, len(pop))

        if hof is not None:
            hof.update(pop)

        record = stats.compile(pop) if stats is not None else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record , hyper= hypervolume(pop) )

        total_eval = 0

        # Begin the generational process
        for gen in range(1, ngen + 1):

            # Vary the population
            offspring = tools.selTournamentDCD(pop, len(pop))
            offspring = [self.toolbox.clone(ind) for ind in offspring]
            offspring = algorithms.varAnd(pop, self.toolbox, cxpb, mutpb)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)

            total_eval += len(invalid_ind)

            # check if 10k evals reached
            if total_eval >= self.configuration["max_evals"]:
               break

            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Update the hall of fame with the generated individuals
            if hof is not None:
                hof.update(offspring)

            # Select the next generation population
            pop[:] = self.toolbox.select(pop + offspring, mu)

            # Update the statistics with the new population
            record = stats.compile(pop) if stats is not None else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record, hyper= hypervolume(pop))

        #pop, logbook = algorithm.myEaMuPlusLambda(pop, _stats=stats, _halloffame=hof)
        
        return pop, logbook, hof

if __name__ == "__main__":
    # get config from yaml
    config = aux.read_yaml(sys.argv[1])

    if "fill_orders" in config.keys():
        aux.validate_config(config, ["dist_file", "coord_file"], sys.argv[1])
    else:
        aux.validate_config(config, ["orders_file", "dist_file", "coord_file"], sys.argv[1])

    # get data from csv
    if "fill_orders" in config.keys():
        orders = [config["orders"] for i in range(config["nb_costumers"])]
        orders.insert(0, 0)
    else:
        orders = list(itertools.chain(*aux.readCSV(config["orders_file"])))

    print(orders)

    distances = aux.readCSV(config["dist_file"])
    coordinates = aux.readCSV(config["coord_file"])

    if config["multi_objective"] == True:
        algorithm = Algorithm(config, orders, distances, coordinates)

        pop, logbook, hof = algorithm.runMulti()

        
        pareto_front = []

        for points in hof:
            pareto_front.append(algorithm.evalRouteMulti(points))
        print("NUMBER OF CUSTOMERS: " + str(config['nb_costumers']))
        print()
        print("Pareto Front:" + str(pareto_front))
        print()
        print("Min Cost:" + str(pareto_front[-1]))
        print("Min Dist:" + str(pareto_front[0]))

        # Pareto Front
        maxCost = pareto_front[0][1]
        maxDist = pareto_front[-1][0]

        costNorm = []
        distNorm = []

        for index, point in enumerate(pareto_front):
            distNorm.append(pareto_front[index][0] / maxDist)
            costNorm.append(pareto_front[index][1] / maxCost)

        # Hyper Volume
        hypervolume = []

        for index in range(logbook[-1]['gen'] + 1):
            hypervolume.append(logbook[index]['hyper'])      

        hypervolume_array = np.array(hypervolume)

        max_hyper = np.max(hypervolume)  
        for index, hyper in enumerate(hypervolume_array):
            hypervolume_array[index] = hypervolume_array[index] / max_hyper 

        plt.plot(list(range(len(hypervolume_array))), hypervolume_array)

        plt.show()

        plt.plot(costNorm,distNorm, 'o')

        plt.show()
        

    else:
        algorithm = Algorithm(config, orders, distances, coordinates)

        min_route, min_trend, avg_trend, stats = algorithm.runSingle()

        print(stats)

        plotRoute(min_route, coordinates)

        plotTrends([min_trend])
