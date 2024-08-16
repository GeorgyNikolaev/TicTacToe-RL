import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


from deap import base, creator, tools, algorithms

import random
import matplotlib.pyplot as plt
import numpy as np

import MyModel
import TicTacToe_game


polygone_size = 3
my_layers = [16, polygone_size]

# Гипер-параметры.
low = -1.0
up = 1.0
eta = 50

population_size = 200
max_generation = 20
p_crossover = 0.9
p_mutation = 0.1
hall_of_fame_size = 5

hof = tools.HallOfFame(hall_of_fame_size)

award = 3
game = TicTacToe_game.TicTacToe(polygone_size=polygone_size, award=award)
model = MyModel.NNetwork(polygone_size ** 2, *my_layers)
length_chrom = model.get_total_weight()

random_seed = 42
random.seed(random_seed)


creator.create("FitnessMin", base.Fitness, weights=(1.0, ))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()


def random_init(l, u):
    return np.float32(random.uniform(l, u))


toolbox.register("randomWeights", random_init, low, up)
toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.randomWeights, length_chrom)
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

population = toolbox.populationCreator(n=population_size)


def getScore(individual):
    # print('reset')
    # print(individual)
    # print(model.model.get_weights())
    model.set_weight(np.float32(individual))
    # print(model.model.get_weights())

    observation = game.reset()
    totalReward = 0

    done = False
    while not done:
        # print(observation)
        p_action = model(np.reshape(observation, newshape=(1, 1, polygone_size ** 2)))
        p_action = np.reshape(p_action, (2, 3))
        # p_action = [[x / max(p_action[0]) for x in p_action[0]],
        #             [x / max(p_action[1]) for x in p_action[1]]]
        # print(p_action)
        action = [np.where(p_action[0] == max(p_action[0]))[0][0],
                  np.where(p_action[1] == max(p_action[1]))[0][0]]
        observation, reward, done = game.step(action=action)
        totalReward += reward

    return totalReward,


toolbox.register("evaluate", getScore)
toolbox.register("select", tools.selTournament, tournsize=20)
toolbox.register("mate", tools.cxBlend, alpha=eta)
# toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=low, up=up, eta=eta)
# toolbox.register("mate", tools.cxSimulatedBinaryBounded, indpb=1.0 / length_chrom)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=1.0 / length_chrom)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("max", np.max)
stats.register("avg", np.mean)

population, logbook = algorithms.eaSimple(population, toolbox,
                                                 cxpb=p_crossover,
                                                 mutpb=p_mutation,
                                                 ngen=max_generation,
                                                 halloffame=hof,
                                                 stats=stats,
                                                 verbose=True)

maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")

# with open('brains.json', 'w') as fw:
#     json.dump(str(hof.items), fw)

plt.ioff()
plt.show()

plt.plot(maxFitnessValues, color='red')
plt.plot(meanFitnessValues, color='green')
plt.show()


for i in range(hall_of_fame_size * 2):
    observation = game.reset()
    if i >= hall_of_fame_size:
        print("best")
        brain = np.float32(hof.items[i-hall_of_fame_size])
        model.set_weight(brain)
    done = False
    print('_'*30)

    while not done:
        p_action = model(np.reshape(observation, newshape=(1, 1, polygone_size ** 2)))
        p_action = np.reshape(p_action, (2, 3))
        p_action = [[x / max(p_action[0]) for x in p_action[0]],
                    [x / max(p_action[1]) for x in p_action[1]]]
        # print(p_action)
        action = [np.where(p_action[0] == max(p_action[0]))[0][0],
                  np.where(p_action[1] == max(p_action[1]))[0][0]]
        observation, reward, done = game.step(action=action)
        print(observation)
        print('-'*10)
