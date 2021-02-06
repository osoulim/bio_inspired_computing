from math import sin, cos, sqrt, pi, exp
from random import uniform
from copy import deepcopy


def f1(x, y):
    return -abs(sin(x) * cos(y) * exp(abs(1-sqrt(x*x + y*y)/pi)))


def f2(x, y):
    numer = cos(sin(abs(x*x - y*y)))
    denom = 1 + 0.0001*(x*x + y*y)
    return 0.5 + (numer*numer - 0.5)/(denom*denom)


functions = [f1, f2]
bounds = [10, 100]
particles = 500
iterations = 1500
phi_q = 0.5
phi_p = 0.2
omega = 0.1


for index, func in enumerate(functions):
    bound = bounds[index]
    p_vector = [[uniform(-bound, bound+1), uniform(-bound, bound+1)] for i in range(particles)]
    particle_best_pos = deepcopy(p_vector)
    particle_best_val = [func(pos[0], pos[1]) for pos in particle_best_pos]
    swarm_best_val = max(particle_best_val)
    swarm_best_pos = particle_best_pos[particle_best_val.index(swarm_best_val)]
    v_vector = [[uniform(-2*bound, 2*bound+1), uniform(-2*bound, 2*bound+1)] for i in range(particles)]

    for iteration in range(iterations):
        for par in range(particles):
            rp = uniform(0, 1)
            rq = uniform(0, 1)
            for dim in range(2):
                v_vector[par][dim] = omega * v_vector[par][dim]
                v_vector[par][dim] += phi_p * rp * (particle_best_pos[par][dim] - p_vector[par][dim])
                v_vector[par][dim] += phi_q * rq * (swarm_best_pos[dim] - p_vector[par][dim])
            for dim in range(2):
                p_vector[par][dim] = max(-bound, min(bound, p_vector[par][dim] + v_vector[par][dim]))
            value = func(p_vector[par][0], p_vector[par][1])
            if value < particle_best_val[par]:
                particle_best_val[par] = value
                particle_best_pos[par] = deepcopy(p_vector[par])
            if value < swarm_best_val:
                swarm_best_val = value
                swarm_best_pos = deepcopy(p_vector[par])

    print("min for function " + str(index + 1) + ":")
    print(swarm_best_pos)
    print(swarm_best_val)
    print("------------")

