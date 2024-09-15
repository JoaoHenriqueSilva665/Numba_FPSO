import numpy as np
from matplotlib import pyplot as plt
from numba import njit, jit


@njit
def OptimumDesignofHollowShaft(X):
    d0, k = X
    return (0.3259 * (d0 ** 2) * (1 - k ** 2))


@njit
def constraints_OptimumDesignofHollowShaft(X):
    d0, k = X

    restric_01 = (d0 ** 4) * (1 - k ** 4)
    restric_02 = (d0 ** 3) * ((1 - k) ** 2.5)

    if restric_01 >= 1736.93 and restric_02 <= 2.5165:
        return np.array([OptimumDesignofHollowShaft(X)])
    else:
        return np.array([np.inf])


@njit
def generate_valid_particle(bound_min, bound_max):
    dim = len(bound_min)
    particle = np.empty(dim, dtype=np.float64)

    while True:
        # Gera cada elemento da partícula dentro dos limites especificados
        for i in range(dim):
            particle[i] = np.random.uniform(bound_min[i], bound_max[i])

        # Verifica a partícula usando a função de restrição
        if constraints_OptimumDesignofHollowShaft(particle)[0] < np.inf:
            return particle


@njit
def adjust_particle(particle, bound_min, bound_max, iteration_limit=100):
    dim = len(particle)

    for _ in range(iteration_limit):
        # Ajustar a partícula aleatoriamente para tentar evitar 'inf'
        adjustment = np.random.uniform(-0.01, 0.01, size=dim)
        candidate = particle + adjustment
        candidate = np.clip(candidate, bound_min, bound_max)

        # Verificar se a nova partícula é válida
        if constraints_OptimumDesignofHollowShaft(candidate)[0] < np.inf:
            return candidate

    # Se não encontrar uma partícula válida após várias tentativas, gerar uma nova partícula
    return generate_valid_particle(bound_min, bound_max)


@njit
def Pass_velocity(vMin_bound, vMax_bound, Particles_dim):
    iterations = 3
    Particles, dim = Particles_dim
    # Inicializa os arrays para armazenar todas as iterações
    all_iterations_velocity = np.empty((iterations, Particles, dim), dtype=np.float64)
    all_iterations_position = np.empty((iterations, Particles, dim), dtype=np.float64)

    for k in range(iterations):
        # Preenche a matriz de velocidades com valores aleatórios
        for i in range(Particles):
            for j in range(dim):
                all_iterations_velocity[k, i, j] = np.random.uniform(vMin_bound[j], vMax_bound[j])

        # Preenche a matriz de posições com partículas válidas
        for i in range(Particles):
            all_iterations_position[k, i, :] = generate_valid_particle(bound_min, bound_max)

    return all_iterations_velocity, all_iterations_position


"""def Pass_velocity(vMin_bound,vMax_bound, Particles_dim):
    all_iterations_velocity = []
    all_iterations_position = []

    for _ in range(3):
        # Gerando a matriz de velocidades para esta iteração
        velocity_i = np.random.uniform(vMin_bound, vMax_bound, (Particles_dim))
        position_i = np.random.uniform(bound_min, bound_max, (Particles_dim))

        # Armazenando a matriz de velocidades atual
        all_iterations_velocity.append(velocity_i)
        all_iterations_position.append(position_i)

    # Convertendo a lista para um array NumPy tridimensional
    all_iterations_velocity = np.array(all_iterations_velocity)
    all_iterations_position = np.array(all_iterations_position)
    return all_iterations_velocity, all_iterations_position"""


def FPSO(Particles_dim, alpha_values, beta_values, Function):
    # velocity_i = np.zeros(Particles_dim)
    # num_particles, dim = Particles_dim
    # velocity_i = generate_velocity(vMin_bound, vMax_bound, num_particles)
    velocity_i = np.random.uniform(vMin_bound, vMax_bound, Particles_dim)
    # position_i = generate_positions(bound_min, bound_max, Particles)
    position_i = np.array([generate_valid_particle(bound_min, bound_max) for _ in range(Particles)])

    cost_i = np.array([Function(particle) for particle in position_i])
    pBest = np.copy(position_i)  # Copia de Position
    pBest_cost = np.copy(cost_i)  # Copia de Cost

    gBest = pBest[np.argmin(pBest_cost)]
    gBest_cost = np.min(cost_i)

    BestCost = np.zeros(max_iter)
    velocity_pass, position_pass = Pass_velocity(vMin_bound, vMax_bound, Particles_dim)

    for it in range(max_iter):
        # w = 0.9 - 0.4 * (it / max_iter)

        rand = np.random.rand()
        zr = 4 * rand * (1 - rand)
        w = ((0.9 - 0.4) * (max_iter - it) / max_iter) + 0.4 * zr
        #w = 0.9 - 0.4 * (it / max_iter)
        c1, c2 = 1, 1

        for i in range(Particles):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)

            velocity_i[i] = (
                    (w + alpha_values[it] - 1) * velocity_i[i]
                    + c1 * r1 * (pBest[i] - position_i[i])
                    + c2 * r2 * (gBest - position_i[i])
                    + ((1 / 2) * alpha_values[it] * (1 - alpha_values[it]) * velocity_pass[0, i])
                    + ((1 / 6) * alpha_values[it] * (1 - alpha_values[it]) * (
                    2 - alpha_values[it]) * velocity_pass[1, i])
                    + ((1 / 24) * alpha_values[it] * (1 - alpha_values[it]) * (
                    2 - alpha_values[it]) * (3 - alpha_values[it]) * velocity_pass[2, i])
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
            if (constraints_OptimumDesignofHollowShaft(position_i[i]) == np.inf):
                # print("entrou aqui:")
                # print(constraints_03(position_i[i]))
                position_i[i] = adjust_particle(position_i[i], bound_min, bound_max)
                # position_i[i] = generate_valid_particle()

            # print("##### verificação de Posição #####")

            position_i[i] = np.clip(position_i[i], bound_min,
                                    bound_max)

        # Armazenamento do melhor custo encontrado na iteração atual
        # BestCost[it] = gBest_cost

        # Cálculo dos valores de fitness para todas as partículas na posição atual

        velocity_pass = np.roll(velocity_pass, shift=1, axis=0)
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


def PSO(Particles_dim, Function):
    # velocity_i = np.zeros(Particles_dim)
    # num_particles, dim = Particles_dim
    # velocity_i = generate_velocity(vMin_bound, vMax_bound, num_particles)
    velocity_i = np.random.uniform(vMin_bound, vMax_bound, Particles_dim)
    # position_i = generate_positions(bound_min, bound_max, Particles)
    position_i = np.array([generate_valid_particle(bound_min, bound_max) for _ in range(Particles)])

    cost_i = np.array([Function(particle) for particle in position_i])
    pBest = np.copy(position_i)  # Copia de Position
    pBest_cost = np.copy(cost_i)  # Copia de Cost

    gBest = pBest[np.argmin(pBest_cost)]
    gBest_cost = np.min(cost_i)

    BestCost = np.zeros(max_iter)

    for it in range(max_iter):
        # w = 0.9 - 0.4 * (it / max_iter)

        w = 1
        # w = 0.9 - 0.4 * (it / max_iter)
        c1, c2 = 1, 1

        for i in range(Particles):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)

            velocity_i[i] = (
                    (w) * velocity_i[i]
                    + c1 * r1 * (pBest[i] - position_i[i])
                    + c2 * r2 * (gBest - position_i[i]))
            # print("##### verificação de velocidade #####")
            velocity_i[i] = np.clip(velocity_i[i],
                                    vMin_bound,
                                    vMax_bound)
            position_i[i] = (position_i[i] + velocity_i[i])
            # print("##### verificação de Posição #####")
            # print("##### verificação da possibilidade de existir d, D e N viável #####")
            # position_i[i] = adjust_to_nearest_multiple(position_i[i])
            if (constraints_OptimumDesignofHollowShaft(position_i[i]) == np.inf):
                # print("entrou aqui:")
                # print(constraints_03(position_i[i]))
                position_i[i] = adjust_particle(position_i[i], bound_min, bound_max)
                # position_i[i] = generate_valid_particle()

            # print("##### verificação de Posição #####")

            position_i[i] = np.clip(position_i[i], bound_min,
                                    bound_max)

        # Armazenamento do melhor custo encontrado na iteração atual
        # BestCost[it] = gBest_cost

        # Cálculo dos valores de fitness para todas as partículas na posição atuali

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
        #vector_gBest, vector_gBest_cost, vector_BestCost = FPSO(Particles_dim, alpha_values, beta_values, Function)
        vector_gBest, vector_gBest_cost, vector_BestCost = PSO(Particles_dim, Function)

        # FPSO
        gBest_list[i] = vector_gBest
        gBest_cost_list[i] = vector_gBest_cost
        best_costs[i] = vector_BestCost

    # FPSO
    mean_best_cost = np.mean(best_costs, axis=0)

    print(f"Mínimo do bando (PSO): {np.min(gBest_cost_list)}")
    print(f"Maximo do bando (PSO): {np.max(gBest_cost_list)}")
    print(f"Desvio Padrão (PSO): {np.std(gBest_cost_list)}")
    print(f"Media da função (PSO): {np.mean(gBest_cost_list)}\n")

    print(f"O menor parametro para o PSO:{gBest_list[np.argmin(gBest_cost_list)]}")
    print(f"O maior parametro para o PSO:{gBest_list[np.argmax(gBest_cost_list)]}")

    return mean_best_cost


max_iter = 150
Particles = 200
dim = 2
Particles_dim = (Particles, dim)
bound_min = np.array([7, 0.7])
bound_max = np.array([25, 0.97])

vMin_bound = np.array([-0.1 * (bound_max[0] - bound_min[0]),
                       -0.1 * (bound_max[1] - bound_min[1])])
vMax_bound = np.array([0.1 * (bound_max[0] - bound_min[0]),
                       0.1 * (bound_max[1] - bound_min[1])])

alpha_values = 0.1 + (1.2 * (np.arange(max_iter) / max_iter))
beta_values = 0.1 + (1.2 * (np.arange(max_iter) / max_iter))

mean_best_cost = run_pso_multiple_times(Particles_dim, constraints_OptimumDesignofHollowShaft)
print(mean_best_cost)
