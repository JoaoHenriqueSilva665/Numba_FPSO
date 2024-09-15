from numba import jit
import numpy as np


@jit(nopython=True)
def roll_velocity_pass(velocity_pass):
    n = velocity_pass.shape[0]
    shifted_velocity_pass = np.empty_like(velocity_pass)

    # Preenche o array deslocado
    shifted_velocity_pass[0] = velocity_pass[-1]
    shifted_velocity_pass[1:] = velocity_pass[:-1]

    return shifted_velocity_pass


# Exemplo de uso
velocity_pass = np.array([1, 2, 3, 4, 5])
shifted_velocity_pass = roll_velocity_pass(velocity_pass)
print(shifted_velocity_pass)
