import random
import itertools

from deap.tools import cxOrdered
from tqdm import tqdm

POPULATION_SIZE = 100
GENERATIONS = 400
CX_THRESHOLD = 0.3
MUT_THRESHOLD = 0.1

CHESS_SIZE = 20


class Individual(list):
    fitness = 0


def evaluate(individual):
    global CHESS_SIZE
    return sum(1 if abs(individual[i] - individual[j]) == abs(i - j) else 0
               for i in range(CHESS_SIZE) for j in range(i + 1, CHESS_SIZE))


def mutate(individual):
    a, b = random.randint(0, CHESS_SIZE - 1), random.randint(0, CHESS_SIZE - 1)
    individual[a], individual[b] = individual[b], individual[a]


def generate_individual():
    global CHESS_SIZE
    sorted_list = [_ for _ in range(CHESS_SIZE)]
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

        best = min(population, key=lambda x: x.fitness)
        if best.fitness == 0:
            break

        # Select new generation
        population = select(population, POPULATION_SIZE)

    answer = min(population, key=lambda x: x.fitness)
    print(answer.fitness, answer)
