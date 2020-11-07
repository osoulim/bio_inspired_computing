import random
import itertools

from deap.tools import cxOrdered
from tqdm import tqdm

POPULATION_SIZE = 200
GENERATIONS = 1000
CX_THRESHOLD = 0.7
MUT_THRESHOLD = 0.05

GRAPH_SIZE = 0
GRAPH = []


class Individual(list):
    fitness = 0


def read_data(filename):
    global GRAPH_SIZE, GRAPH
    file = open(filename, 'r')
    GRAPH_SIZE = int(file.readline())
    for line in file:
        GRAPH.append(list(map(int, line.split())))


def evaluate(individual):
    global GRAPH_SIZE, GRAPH
    res = 0
    for i in range(GRAPH_SIZE):
        u, v = individual[i], individual[(i + 1) % GRAPH_SIZE]
        res += GRAPH[u][v]
    return res


def mutate(individual):
    a, b = random.randint(0, GRAPH_SIZE-1), random.randint(0, GRAPH_SIZE-1)
    individual[a], individual[b] = individual[b], individual[a]


def generate_individual():
    global GRAPH_SIZE
    sorted_list = [_ for _ in range(GRAPH_SIZE)]
    random.shuffle(sorted_list)
    res = Individual()
    res.extend(sorted_list)
    return res


def generate_population(size=POPULATION_SIZE):
    population = []
    for x in range(size):
        ind = generate_individual()
        population.append(ind)
    return population


def select(population, amount):
    population.sort(key=lambda x: x.fitness)
    res = []
    for individual in population:
        if individual not in res:
            res.append(individual)
        if len(res) >= amount // 2:
            break
    for i in range(amount - len(res)):
        res.append(min(random.choices(population, k=3), key=lambda x: x.fitness))
    return res


if __name__ == "__main__":
    read_data('./Data2.txt')
    population = generate_population()
    for _ in tqdm(range(GENERATIONS)):
        # Crossover
        for p1, p2 in itertools.product(population, population):
            if random.random() < CX_THRESHOLD:
                c1, c2 = cxOrdered(p1, p2)
                population.append(c1)
                population.append(c2)

        # Mutation
        for p in population:
            if random.random() < MUT_THRESHOLD:
                mutate(p)

        # Evaluate
        for p in population:
            p.fitness = evaluate(p)

        # best = min(population, key=lambda x: x.fitness)
        # print(_, best.fitness, best)

        # Select new generation
        population = select(population, POPULATION_SIZE)

    answer = min(population, key=lambda x: x.fitness)
    print(answer.fitness, answer)
