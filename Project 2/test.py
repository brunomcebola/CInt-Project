from cmath import inf
from math import floor
import random
from tkinter import Place

from deap import base
from deap import creator
from deap import tools


# CONSTANTS
INIT_POP = 40
OFFSPRING_SIZE = 40
NB_PLACES = 10
MAX_LOAD = 1000
MAX_EVALS = 10000
MAX_STALL = 10
CXPB = 0.5
MUTPB = 0.2

# FUNCTIONS
def getPlaceList(nb_places: int):
    options = list(range(1, nb_places + 1))

    random.shuffle(options)

    return options


def evalRoute(individual):

    list_of_routes = []
    route = []
    capacity = 0

    flag = False

    for place in individual:

        distance_of_route = individual #TODO here comes the data from the matrix | NOT WHAT IT IS NOW
        
        capacity += distance_of_route

        if capacity <= 1000:
            route.append(place)
        else:
            list_of_routes.append(route)
            route.clear()
            route.append(place)
            capacity = distance_of_route

    # Calculate the distances made
    total_distance = 0

    for route in list_of_routes:
        for index, place in enumerate(route):
            if index == 0:
                distance = individual #TODO distance from warehouse to place
                total_distance += distance
            elif index == len(route) + 1:
                distance = individual #TODO distance from place to warehouse
                total_distance += distance
            else:
                distance = individual #TODO distance from place before to here
                total_distance += distance
                
    return (total_distance,)


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)  # type: ignore

toolbox = base.Toolbox()

toolbox.register("place", getPlaceList, NB_PLACES)

toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.place)  # type: ignore

toolbox.register("population", tools.initRepeat, list, toolbox.individual)  # type: ignore

toolbox.register("evaluate", evalRoute)

toolbox.register("crossover", tools.cxOrdered)

toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)

toolbox.register("select", tools.selTournament, tournsize=2)

# ALGORITHM
pop = toolbox.population(n=INIT_POP)  # type: ignore

print("-- Generation 0 --")

fitnesses = list(map(toolbox.evaluate, pop))  # type: ignore
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

fits = [ind.fitness.values[0] for ind in pop]

g = 0
stall = 0
min_path = inf

while stall < MAX_STALL and g < floor((MAX_EVALS - INIT_POP) / OFFSPRING_SIZE):
    print("-- Generation %i --" % g)

    offspring = list(map(toolbox.clone, toolbox.select(pop, OFFSPRING_SIZE)))  # type: ignore

    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CXPB:
            toolbox.crossover(child1, child2)  # type: ignore

            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < MUTPB:
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

    g = g + OFFSPRING_SIZE

print("-- End of (successful) evolution --")

best_ind = tools.selBest(pop, 1)[0]
print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
