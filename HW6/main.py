from random import randint, shuffle, random
from math import exp, pow
from copy import deepcopy

data = open("data.txt", "r")
lines = data.readlines()
n = len(lines) + 1
vertices = [i for i in range(0, n)]
adj = [[0 for i in range(n)] for j in range(n)]
for index, line in enumerate(lines):
    for idx, x in enumerate(line.split()):
        adj[index][idx + index + 1] = adj[index + idx + 1][index] = float(x)


def fitness(path):
    weight = 0
    for v in range(0, n - 1):
        weight += adj[path[v]][path[v + 1]]
    weight += adj[path[n - 1]][path[0]]
    return weight


global_best = 8000000
global_path = []
for iteration in range(0, 4):
    x = deepcopy(vertices)
    shuffle(x)
    T = 2400
    while T > 0:
        c = 0
        while c < 24+(2400-T)/100:
            q = randint(2, len(x)-1)
            p = randint(1, q)
            neighbor = deepcopy(x)
            neighbor[p:q+1] = neighbor[q:p-1:-1]
            dif = fitness(neighbor) - fitness(x)
            if dif < 0:
                dif = 0
            if random() < pow(exp(1), -dif/T):
                x = neighbor
                c += 1
        T -= 1
    if fitness(x) < global_best:
        global_path = x
        global_best = fitness(x)

print(global_best, global_path)
