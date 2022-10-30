import sys
import random
import itertools
import matplotlib.pyplot as plt

from math import inf
from deap import base
from deap import tools
from deap import creator
from copy import deepcopy

import aux_functions as aux


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
        "max_stall": 20,  # TODO: mudar para inf
        "max_evals": 10000,
    }

    @classmethod
    def getRoute(cls, individual, orders, max_load) -> list:
        load = 0
        route = [-1]

        for location in individual:
            load += int(orders[location + 1])

            if load > max_load:
                route.append(-1)
                load = 0

            route.append(location)

        route.append(-1)

        return route

    @classmethod
    def getDistance(cls, route, distances) -> int:
        distance = 0

        for prev_place, current_place in zip(route[:-1], route[1:]):
            distance += distances[prev_place + 1][current_place + 1]

        return distance

    @classmethod
    def evalRoute(cls, individual, orders, distances, max_load) -> tuple:

        route = Algorithm.getRoute(individual, orders, max_load)

        distance = Algorithm.getDistance(route, distances)

        return (distance,)

    def __init__(self, config: dict, orders: list = [], distances: list = [], coordinates: list = []) -> None:
        self.stats = []
        self.route = []
        self.min_trends = []
        self.avg_trends = []

        self.toolbox = base.Toolbox()

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)  # type: ignore

        self.set_configuration(config)

        self.set_orders(orders)
        self.set_distances(distances)
        self.set_coordinates(coordinates)

        self.register_toolbox()

    def set_configuration(self, config: dict = {}) -> None:
        for key in config:
            if key in self.configuration:
                self.configuration[key] = config[key]

    def set_orders(self, orders: list = []) -> None:
        self.orders = orders

    def set_coordinates(self, coordinates: list = []) -> None:
        self.coordinates = coordinates

    def set_distances(self, distances: list = []) -> None:
        self.distances = distances

    def register_toolbox(self) -> None:
        self.toolbox.register(
            "location", random.sample, range(self.configuration["nb_costumers"]), self.configuration["nb_costumers"]
        )
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.location)  # type: ignore
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)  # type: ignore

        self.toolbox.register("mate", tools.cxPartialyMatched)
        self.toolbox.register("select", tools.selTournament, tournsize=2)
        self.toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
        self.toolbox.register(
            "evaluate",
            Algorithm.evalRoute,
            orders=self.orders,
            distances=self.distances,
            max_load=self.configuration["max_load"],
        )

    def run(self) -> None:
        global_min_fit = inf

        global_fits = []
        global_min_route = []
        global_min_trend = []
        global_avg_trend = []

        # test algorithm for different seeds
        for seed in range(self.configuration["iterations"]):
            min_fit = inf
            min_route = []
            min_trend = []
            avg_trend = []

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

            global_fits.append(min_fit)

        # stores best info ever on algorithm itself
        self.route = global_min_route
        self.min_trends.append(global_min_trend)
        self.avg_trends.append(global_avg_trend)

        # stores stats
        mean = sum(global_fits) / len(global_fits)
        std = abs((sum(x * x for x in global_fits)) / self.configuration["pop_size"] - mean**2) ** 0.5
        self.stats.append({"min": global_min_fit, "mean": mean, "std": std})

    def plotRoute(self) -> None:
        route = deepcopy(self.route)

        route = Algorithm.getRoute(route, self.orders, self.configuration["max_load"])

        X = []
        Y = []
        L = []

        for i in range(len(route)):
            route[i] += 1

        for location in route:
            X.append(int(self.coordinates[location][0]))
            Y.append(int(self.coordinates[location][1]))
            L.append(location)

        plt.plot(X, Y, "-o")

        for i, txt in enumerate(L):
            plt.annotate(txt, (X[i], Y[i]))

        plt.show()

    def plotMinTrends(self) -> None:
        for trend in self.min_trends:
            plt.plot(list(range(len(trend))), trend)

        plt.show()

    def plotAvgTrends(self) -> None:
        for trend in self.avg_trends:
            plt.plot(list(range(len(trend))), trend)

        plt.show()

    def printStats(self) -> None:
        for stat in self.stats:
            print(stat)
            print("Min: %s" % stat["min"])
            print("Mean: %s" % stat["mean"])
            print("Std: %s" % stat["std"])

    def reset_trends(self) -> None:
        self.min_trends = []
        self.avg_trends = []

    def reset_stats(self) -> None:
        self.stats = []


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

    distances = aux.readCSV(config["dist_file"])
    coordinates = aux.readCSV(config["coord_file"])

    algorithm = Algorithm(config, orders, distances, coordinates)

    algorithm.run()

    algorithm.printStats()

    algorithm.plotRoute()

    algorithm.plotAvgTrends()
    
    algorithm.plotMinTrends()
