# Pacote de funções 2

def triangular_inferior(a, b):
    import numpy as np
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)

    y = np.zeros(len(b))

    for i in range(len(b)):
        soma_termos= 0.0
        for j in range(i):
            soma_termos = soma_termos + a[i, j] * y[j]
        if a[i, i] == 0:
            print(f"Erro: O elemento L[{i},{i}] é zero.")

        y[i] = (b[i] - soma_termos) / a[i, i]

    return y

# def triangular_superior(a, b):