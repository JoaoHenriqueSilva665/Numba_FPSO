import numpy as np
from matplotlib import pyplot as plt
from numba import njit, jit

@njit
def DesignOfTensionCompressionSpring(X):
    d, D, N = X
    return (N + 2) * (D * d ** 2)


@njit
def constraints_DesignOfTensionCompressionSpring(X):
    d, D, N = X

    restric_01 = (D ** 3 * N) / (71785 * d ** 4)
    restric_02 = ((4 * D ** 2 - d * D) / (12566 * (D * d ** 3 - d ** 4))) + (1 / (5108 * d ** 2))
    restric_03 = (140.45 * d) / (D ** 2 * N)
    restric_04 = (D + d) / 1.5

    if restric_01 >= 1 and restric_02 <= 1 and restric_03 >= 1 and restric_04 <= 1:
        return np.array([DesignOfTensionCompressionSpring(X)])
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
        if constraints_DesignOfTensionCompressionSpring(particle)[0] < np.inf:
            return particle


@njit
def adjust_particle(particle, bound_min, bound_max, iteration_limit=100):
    dim = len(particle)

    for _ in range(iteration_limit):
        # Ajustar a partícula aleatoriamente para tentar evitar 'inf'
        adjustment = np.random.uniform(-0.0001, 0.0001, size=dim)
        candidate = particle + adjustment
        candidate = np.clip(candidate, bound_min, bound_max)

        # Verificar se a nova partícula é válida
        if constraints_DesignOfTensionCompressionSpring(candidate)[0] < np.inf:
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


"""def Pass_velocity(vMin_bound, vMax_bound, Particles_dim):
    all_iterations_velocity = []
    all_iterations_position = []

    for _ in range(3):
        # Gerando a matriz de velocidades para esta iteração
        velocity_i = np.random.uniform(vMin_bound, vMax_bound, (Particles_dim))
        position_i = np.array([generate_valid_particle(bound_min, bound_max) for _ in range(Particles)])

        # Armazenando a matriz de velocidades atual
        all_iterations_velocity.append(velocity_i)
        all_iterations_position.append(position_i)

    # Convertendo a lista para um array NumPy tridimensional
    all_iterations_velocity = np.array(all_iterations_velocity)
    all_iterations_position = np.array(all_iterations_position)
    return all_iterations_velocity, all_iterations_position

"""


@njit
def generate_velocity(vMin_bound, vMax_bound, num_particles):
    num_dimensions = len(vMin_bound)
    velocity_matrix = np.empty((num_particles, num_dimensions))

    for i in range(num_particles):
        for j in range(num_dimensions):
            velocity_matrix[i, j] = vMin_bound[j] + (vMax_bound[j] - vMin_bound[j]) * np.random.random()

    return velocity_matrix


@njit
def generate_positions(bound_min, bound_max, num_particles):
    num_dimensions = len(bound_min)
    position_matrix = np.empty((num_particles, num_dimensions))
    for i in range(num_particles):
        position_matrix[i] = generate_valid_particle(bound_min, bound_max)
    return position_matrix


@njit
def compute_costs(position_matrix):
    num_particles = position_matrix.shape[0]
    costs = np.empty(num_particles)
    for i in range(num_particles):
        costs[i] = DesignOfTensionCompressionSpring(position_matrix[i])
    return costs


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
    # print(f"A posição responsável pelo minimo: \n{gBest} \n")
    gBest_cost = np.min(cost_i)
    # print(f"O minimo: \n{gBest_cost} \n")

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

            # print(f"(ANTES VELOCITY) em i={i}:\n {velocity_i} ")
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
            # print(f"(DEPOIS VELOCITY) em i={i}:\n {velocity_i}\n")
            # print("##### verificação de velocidade #####")
            velocity_i[i] = np.clip(velocity_i[i],
                                    vMin_bound,
                                    vMax_bound)
            # print(f"###### Depois da verificação de velocidade:\n {velocity_i}\n")

            # print(f"(ANTES POSITION) em i={i}:\n {position_i}")
            position_i[i] = (
                    beta_values[it] * position_i[i]
                    + velocity_i[i]
                    + ((1 / 2) * beta_values[it] * (1 - beta_values[it]) * (position_pass[0, i]))
                    + ((1 / 6) * beta_values[it] * (1 - beta_values[it]) * (2 - beta_values[it]) * (position_pass[1, i]))
                    + ((1 / 24) * beta_values[it] * (1 - beta_values[it]) * (2 - beta_values[it]) * (3 - beta_values[it]) * (position_pass[2, i]))
            )
            # print(f"(DEPOIS POSITION) em i={i}:\n {position_i}\n")
            # print("##### verificação de Posição #####")
            # print("##### verificação da possibilidade de existir d, D e N viável #####")
            # position_i[i] = adjust_to_nearest_multiple(position_i[i])
            if (constraints_DesignOfTensionCompressionSpring(position_i[i]) == np.inf):
                # print("entrou aqui:")
                # print(constraints_03(position_i[i]))
                position_i[i] = adjust_particle(position_i[i], bound_min, bound_max)
                # position_i[i] = generate_valid_particle()

            # print(f"##### Depois da validação: \n{position_i}\n #####")
            # print("##### verificação de Posição #####")

            position_i[i] = np.clip(position_i[i], bound_min,
                                    bound_max)

            # print(f"###### Depois da verificação:\n {position_i}\n")

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
        improved_index = np.where(fitness_values < pBest_cost)
        # print(f"Onde a condição de pbest > fitness_values:\n {improved_index}\n")

        # print(f"comparação:\n antes -> \n{pBest}\n")
        # Atualização de pBest para partículas que encontraram uma melhor solução
        pBest[improved_index] = position_i[improved_index]
        # print(f"comparação:\n depois -> \n{pBest}\n ")

        # print(f"comparação:\n antes -> \n{pBest_cost}\n")
        pBest_cost[improved_index] = fitness_values[improved_index]
        # print(f"comparação:\n depois -> \n{pBest_cost}\n ")

        # Verificação se a melhor solução global foi encontrada
        min_fitness_value = np.min(fitness_values)
        # print(f"O minimo!: {min_fitness_value}")

        # print(f"O otimo global até agora.. {it}: \n{gBest_cost}\n")
        # print(f"A melhor particula global até agora..: \n{gBest}\n")
        # print(f"A melhor particula local até agora..: \n{pBest}\n")
        # print(f"array de posições até agora..: \n{position_i}\n")
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
    # print(f"A posição responsável pelo minimo: \n{gBest} \n")
    gBest_cost = np.min(cost_i)
    # print(f"O minimo: \n{gBest_cost} \n")

    BestCost = np.zeros(max_iter)
    for it in range(max_iter):

        w = 1
        c1, c2 = 1, 1

        for i in range(Particles):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)

            # print(f"(ANTES VELOCITY) em i={i}:\n {velocity_i} ")
            velocity_i[i] = (
                    (w) * velocity_i[i]
                    + c1 * r1 * (pBest[i] - position_i[i])
                    + c2 * r2 * (gBest - position_i[i]))
            # print(f"(DEPOIS VELOCITY) em i={i}:\n {velocity_i}\n")
            # print("##### verificação de velocidade #####")
            velocity_i[i] = np.clip(velocity_i[i], vMin_bound, vMax_bound)
            # print(f"###### Depois da verificação de velocidade:\n {velocity_i}\n")

            # print(f"(ANTES POSITION) em i={i}:\n {position_i}")
            position_i[i] = (position_i[i] + velocity_i[i])
            # print(f"(DEPOIS POSITION) em i={i}:\n {position_i}\n")
            # print("##### verificação de Posição #####")
            # print("##### verificação da possibilidade de existir d, D e N viável #####")
            # position_i[i] = adjust_to_nearest_multiple(position_i[i])
            if (constraints_DesignOfTensionCompressionSpring(position_i[i]) == np.inf):
                # print("entrou aqui:")
                # print(constraints_03(position_i[i]))
                position_i[i] = adjust_particle(position_i[i], bound_min, bound_max)
                # position_i[i] = generate_valid_particle()

            # print(f"##### Depois da validação: \n{position_i}\n #####")
            # print("##### verificação de Posição #####")

            position_i[i] = np.clip(position_i[i], bound_min,
                                    bound_max)

            # print(f"###### Depois da verificação:\n {position_i}\n")

        # Armazenamento do melhor custo encontrado na iteração atual
        # BestCost[it] = gBest_cost

        fitness_values = np.array([Function(particle) for particle in position_i])
        # print(f"O resultado da função objetivo!!!: {fitness_values}")

        # Verificação das partículas que melhoraram sua posição
        improved_index = np.where(fitness_values < pBest_cost)
        # print(f"Onde a condição de pbest > fitness_values:\n {improved_index}\n")

        # print(f"comparação:\n antes -> \n{pBest}\n")
        # Atualização de pBest para partículas que encontraram uma melhor solução
        pBest[improved_index] = position_i[improved_index]
        # print(f"comparação:\n depois -> \n{pBest}\n ")

        # print(f"comparação:\n antes -> \n{pBest_cost}\n")
        pBest_cost[improved_index] = fitness_values[improved_index]
        # print(f"comparação:\n depois -> \n{pBest_cost}\n ")

        # Verificação se a melhor solução global foi encontrada
        min_fitness_value = np.min(fitness_values)
        # print(f"O minimo!: {min_fitness_value}")

        # print(f"O otimo global até agora.. {it}: \n{gBest_cost}\n")
        # print(f"A melhor particula global até agora..: \n{gBest}\n")
        # print(f"A melhor particula local até agora..: \n{pBest}\n")
        # print(f"array de posições até agora..: \n{position_i}\n")
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

    # PSO
    gBest_list_classic = np.zeros((it, dim))
    gBest_cost_list_classic = np.zeros(it)
    best_costs_classic = np.zeros((it, max_iter))

    for i in range(it):
        vector_gBest, vector_gBest_cost, vector_BestCost = FPSO(Particles_dim,
                                                                alpha_values,
                                                                beta_values,
                                                                Function)
        vector_gBest_classic, vector_gBest_cost_classic, vector_BestCost_classic = PSO(Particles_dim, Function)

        # FPSO
        gBest_list[i] = vector_gBest
        gBest_cost_list[i] = vector_gBest_cost
        best_costs[i] = vector_BestCost

        # PSO
        gBest_list_classic[i] = vector_gBest_classic
        gBest_cost_list_classic[i] = vector_gBest_cost_classic
        best_costs_classic[i] = vector_BestCost_classic

    # FPSO
    mean_best_cost = np.mean(best_costs, axis=0)
    # PSO
    mean_best_cost_classic = np.mean(best_costs_classic, axis=0)

    print(f"Mínimo do bando (FPSO): {np.min(gBest_cost_list)}")
    print(f"Maximo do bando (FPSO): {np.max(gBest_cost_list)}")
    print(f"Desvio Padrão (FPSO): {np.std(gBest_cost_list)}")
    print(f"Media da função (FPSO): {np.mean(gBest_cost_list)}\n")

    print(f"Mínimo do bando (PSO): {np.min(gBest_cost_list_classic)}")
    print(f"Maximo do bando (PSO): {np.max(gBest_cost_list_classic)}")
    print(f"Desvio Padrão (PSO): {np.std(gBest_cost_list_classic)}")
    print(f"Media da função (PSO): {np.mean(gBest_cost_list_classic)}\n")

    print(f"O menor parametro para o FPSO:{gBest_list[np.argmin(gBest_cost_list)]}")
    print(f"O maior parametro para o FPSO:{gBest_list[np.argmax(gBest_cost_list)]}")
    print(f"O menor parametro para o PSO:{gBest_list_classic[np.argmin(gBest_cost_list_classic)]}")
    print(f"O maior parametro para o PSO:{gBest_list_classic[np.argmax(gBest_cost_list_classic)]}")

    return (mean_best_cost, mean_best_cost_classic)


def Plot_graphic(mean_best_cost, mean_best_cost_classic):
    plt.rcParams['legend.fontsize'] = 14
    plt.plot(mean_best_cost_classic, color="k", linestyle="solid", label="PSO",
             linewidth=1.5)
    plt.plot(mean_best_cost, color="k", linestyle="dashdot", label="FPSO",
             linewidth=1.5)

    plt.tick_params(axis='x', direction='in')
    plt.tick_params(axis='y', direction='in')

    plt.xlabel("Número de Iteração", fontsize=12)
    plt.xlim(0, max_iter)

    plt.ylabel("Valor da Função Objetivo", fontsize=12)
    plt.yscale("log")

    plt.legend(framealpha=1, frameon=True)
    plt.tight_layout()
    plt.savefig("Imagens/DesignOfTensionCompressionSpring_.svg", format="svg")
    # plt.show()


max_iter = 100
Particles = 300
dim = 3
Particles_dim = (Particles, dim)
bound_min = np.array([0.05, 0.25, 2])
bound_max = np.array([2, 1.3, 15])
alpha_values = 0.1 + 1.2 * ((np.arange(max_iter) / max_iter))
beta_values = 0.1 + 1.2 * ((np.arange(max_iter) / max_iter))

vMin_bound = np.array([-0.1 * (bound_max[0] - bound_min[0]),
                       -0.1 * (bound_max[1] - bound_min[1]),
                       -0.1 * (bound_max[2] - bound_min[2])])

vMax_bound = np.array([0.1 * (bound_max[0] - bound_min[0]),
                       0.1 * (bound_max[1] - bound_min[1]),
                       0.1 * (bound_max[2] - bound_min[2])])

mean_best_cost, mean_best_cost_classic = run_pso_multiple_times(Particles_dim, constraints_DesignOfTensionCompressionSpring)
Plot_graphic(mean_best_cost, mean_best_cost_classic)
