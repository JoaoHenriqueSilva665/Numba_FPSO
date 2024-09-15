import numpy as np
from matplotlib import pyplot as plt
from numba import njit, jit


def DesignOfPressureVessel(X):
    x1, x2, x3, x4 = X
    return ((0.6224 * x1 * x3 * x4) +
            (1.7781 * x2 * x3 ** 2) +
            (3.1661 * x1 ** 2 * x4) +
            (19.84 * x1 ** 2 * x3))

def adjust_to_integer_multiples(value, multiple=0.0625):
    """Ajusta o valor para o múltiplo inteiro mais próximo de `multiple`."""
    x1, x2, x3, x4 = value
    # tolerance =
    # print(f"x1: {x1}")
    # print(f"x2: {x2}\n")
    adjust_x1 = np.round(x1 / multiple) * multiple
    adjust_x2 = np.round(x2 / multiple) * multiple

    # print(f"ajustado x1: {adjust_x1}")
    # print(f"ajustado x2: {adjust_x2}\n")

    return adjust_x1, adjust_x2


def check_integer_multiples(x):
    """Verifica se T_s e T_h são múltiplos inteiros de `multiple`."""
    multiple = 0.0625
    x1, x2, x3, x4 = x
    return (x1 % multiple == 0) and (x2 % multiple == 0)


def constraints_DesignOfPressureVessel(X):
    x1, x2, x3, x4 = X

    try:
        c1 = -x1 + 0.0193 * x3
        c2 = -x2 + 0.00954 * x3
        c3 = -np.pi * x3 ** 2 * x4 - 43 * np.pi * x3 ** 3 + 1296000
        c4 = x4 - 240

        if (c1 <= 0 and c2 <= 0 and c3 <= 0 and c4 <= 0 and check_integer_multiples(X)):
            return DesignOfPressureVessel(X)
        else:
            return np.inf
    except (ZeroDivisionError, OverflowError):
        return np.inf


def generate_valid_particle():
    while True:
        # x1, x2 = random_multiples()
        particle = np.random.uniform(bound_min, bound_max)
        x1, x2 = adjust_to_integer_multiples(particle)

        particle[0] = x1
        particle[1] = x2

        if constraints_DesignOfPressureVessel(particle) < np.inf:
            return particle


def adjust_particle(particle, iteration_limit=100):
    for _ in range(iteration_limit):
        # Ajustar a partícula aleatoriamente para tentar evitar 'inf'
        # x1, x2 = adjust_to_integer_multiples(particle)
        x1, x2, x3, x4 = particle
        adjustment = np.random.uniform(-0.001, 0.001, size=dim)

        candidate_x3 = x3 + adjustment[2]
        candidate_x4 = x4 + adjustment[3]

        # Verificar se a nova partícula é válida
        # candidate[0] = x1
        # candidate[1] = x2
        candidate = [x1, x2, candidate_x3, candidate_x4]
        if constraints_DesignOfPressureVessel(candidate) < np.inf:
            return candidate

    # Se não encontrar uma partícula válida após várias tentativas, gerar uma nova partícula
    return generate_valid_particle()


def Pass_velocity(vMin_bound, vMax_bound, Particles_dim):
    all_iterations_velocity = []
    all_iterations_position = []

    for _ in range(3):
        # Gerando a matriz de velocidades para esta iteração
        velocity_i = np.random.uniform(vMin_bound, vMax_bound, (Particles_dim))
        position_i = np.array([generate_valid_particle() for _ in range(Particles)])

        # Armazenando a matriz de velocidades atual
        all_iterations_velocity.append(velocity_i)
        all_iterations_position.append(position_i)

    # Convertendo a lista para um array NumPy tridimensional
    all_iterations_velocity = np.array(all_iterations_velocity)
    all_iterations_position = np.array(all_iterations_position)
    return all_iterations_velocity, all_iterations_position

def FPSO(Particles_dim, alpha_values, beta_values, Function):
    #velocity_i = np.zeros(Particles_dim)
    # num_particles, dim = Particles_dim
    #velocity_i = generate_velocity(vMin_bound, vMax_bound, num_particles)
    velocity_i = np.random.uniform(vMin_bound, vMax_bound, Particles_dim)
    # position_i = generate_positions(bound_min, bound_max, Particles)
    position_i = np.array([generate_valid_particle() for _ in range(Particles)])

    cost_i = np.array([Function(particle) for particle in position_i])
    pBest = np.copy(position_i)  # Copia de Position
    pBest_cost = np.copy(cost_i)  # Copia de Cost

    gBest = pBest[np.argmin(pBest_cost)]
    gBest_cost = np.min(cost_i)

    BestCost = np.zeros(max_iter)
    velocity_pass, position_pass = Pass_velocity(vMin_bound, vMax_bound, Particles_dim)

    for it in range(max_iter):

        rand = np.random.rand()
        zr = 4 * rand * (1 - rand)
        w = ((0.9 - 0.4) * (max_iter - it) / max_iter) + 0.4 * zr
        # w = 0.9 - 0.4 * (it / max_iter)
        c1, c2 = 1, 1

        for i in range(Particles):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)

            velocity_i[i] = (
                    (w + alpha_values[it] - 1) * velocity_i[i]
                    + c1 * r1 * (pBest[i] - position_i[i])
                    + c2 * r2 * (gBest - position_i[i])
                    + ((1 / 2) * alpha_values[it] * (1 - alpha_values[it]) * velocity_pass[0, i])
                    + ((1 / 6) * alpha_values[it] * (1 - alpha_values[it]) * (2 - alpha_values[it]) * velocity_pass[1, i])
                    + ((1 / 24) * alpha_values[it] * (1 - alpha_values[it]) * (2 - alpha_values[it]) * (3 - alpha_values[it]) * velocity_pass[2, i])
            )
            # print("##### verificação de velocidade #####")
            velocity_i[i] = np.clip(velocity_i[i],
                                    vMin_bound,
                                    vMax_bound)
            position_i[i] = (
                    beta_values[it] * position_i[i]
                    + velocity_i[i]
                    + ((1 / 2) * beta_values[it] * (1 - beta_values[it]) * (position_pass[0, i]))
                    + ((1 / 6) * beta_values[it] * (1 - beta_values[it]) * (2 - beta_values[it]) * (position_pass[1, i]))
                    + ((1 / 24) * beta_values[it] * (1 - beta_values[it]) * (2 - beta_values[it]) * (3 - beta_values[it]) * (position_pass[2, i]))
            )
            # print("##### verificação de Posição #####")
            # print("##### verificação da possibilidade de existir d, D e N viável #####")
            # position_i[i] = adjust_to_nearest_multiple(position_i[i])
            if (constraints_DesignOfPressureVessel(position_i[i]) == np.inf):
                # print("entrou aqui:")
                # print(constraints_03(position_i[i]))
                position_i[i] = adjust_particle(position_i[i])
                # position_i[i] = generate_valid_particle()

            # print("##### verificação de Posição #####")

            position_i[i] = np.clip(position_i[i], bound_min,
                                    bound_max)

        # Armazenamento do melhor custo encontrado na iteração atual
        # BestCost[it] = gBest_cost

        # Cálculo dos valores de fitness para todas as partículas na posição atual'
        elocity_pass = np.roll(velocity_pass, shift=1, axis=0)
        velocity_pass[0] = velocity_i

        position_pass = np.roll(position_pass, shift=1, axis=0)
        position_pass[0] = position_i

        fitness_values = np.array([Function(particle) for particle in position_i])
        # print(f"O resultado da função objetivo!!!: {fitness_values}")

        # Verificação das partículas que melhoraram sua posição
        improved_index = np.where(fitness_values < pBest_cost)[0]

        # Atualização de pBest para partículas que encontraram uma melhor solução
        pBest[improved_index] = position_i[improved_index]

        pBest_cost[improved_index] = fitness_values[improved_index]

        # Verificação se a melhor solução global foi encontrada
        min_fitness_value = np.min(fitness_values)
        # print(f"O minimo!: {min_fitness_value}")

        if min_fitness_value < gBest_cost:
            gBest_cost = min_fitness_value
            gBest = pBest[np.argmin(fitness_values)]

        # Armazenamento do melhor custo encontrado na iteração atual
        BestCost[it] = gBest_cost

    return gBest, gBest_cost, BestCost


def run_pso_multiple_times(Particles_dim, Function, it=25):
    # FPSO
    gBest_list = np.zeros((it, dim))
    gBest_cost_list = np.zeros(it)
    best_costs = np.zeros((it, max_iter))

    for i in range(it):
        vector_gBest, vector_gBest_cost, vector_BestCost = FPSO(Particles_dim, alpha_values, beta_values, Function)

        # FPSO
        gBest_list[i] = vector_gBest
        gBest_cost_list[i] = vector_gBest_cost
        best_costs[i] = vector_BestCost

    # FPSO
    mean_best_cost = np.mean(best_costs, axis=0)

    print(f"Mínimo do bando (FPSO): {np.min(gBest_cost_list)}")
    print(f"Maximo do bando (FPSO): {np.max(gBest_cost_list)}")
    print(f"Desvio Padrão (FPSO): {np.std(gBest_cost_list)}")
    print(f"Media da função (FPSO): {np.mean(gBest_cost_list)}\n")

    print(f"O menor parametro para o FPSO:{gBest_list[np.argmin(gBest_cost_list)]}")
    print(f"O maior parametro para o FPSO:{gBest_list[np.argmax(gBest_cost_list)]}")

    return mean_best_cost


max_iter = 150
Particles = 400
dim = 4
Particles_dim = (Particles, dim)
bound_min = [0.1, 0.1, 10, 10]
bound_max = [99, 99, 200, 200]

vMin_bound = [-0.1 * (bound_max[0] - bound_min[0]),
              -0.1 * (bound_max[1] - bound_min[1]),
              -0.1 * (bound_max[2] - bound_min[2]),
              -0.1 * (bound_max[3] - bound_min[3])]
vMax_bound = [0.1 * (bound_max[0] - bound_min[0]),
              0.1 * (bound_max[1] - bound_min[1]),
              0.1 * (bound_max[2] - bound_min[2]),
              0.1 * (bound_max[3] - bound_min[3])]

alpha_values = 0.1 + (1.2 * (np.arange(max_iter) / max_iter))
beta_values = 0.1 + (1.2 * (np.arange(max_iter) / max_iter))

mean_best_cost = run_pso_multiple_times(Particles_dim, constraints_DesignOfPressureVessel)
print(mean_best_cost)
