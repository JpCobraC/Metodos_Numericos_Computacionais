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

def triangular_superior(a, b):
    import numpy as np
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)

    n = len(b)
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        soma_termos = 0.0
        for j in range(i + 1, n):
            soma_termos = soma_termos + a[i, j] * x[j]
        
        if a[i, i] == 0:
            print(f"Erro: O elemento U[{i},{i}] é zero.")
            return None
        x[i] = (b[i] - soma_termos) / a[i, i]

    return x


def gauss_pivot_parcial(A_original, b_original):
    import numpy as np
    A = A_original.copy().astype(float)
    b = b_original.copy().astype(float)

    num_linhas = A.shape[0]
    num_colunas = A.shape[1]

    if num_linhas != len(b):
        return None

    matriz_aumentada = np.concatenate((A, b.reshape(-1, 1)), axis=1)

    for j in range(min(num_linhas, num_colunas)):
        linha_do_pivo = j
        max_val = abs(matriz_aumentada[j, j])

        for i in range(j + 1, num_linhas):
            if abs(matriz_aumentada[i, j]) > max_val:
                max_val = abs(matriz_aumentada[i, j])
                linha_do_pivo = i

        if linha_do_pivo != j:
            temp_linha = np.copy(matriz_aumentada[j, :])
            matriz_aumentada[j, :] = matriz_aumentada[linha_do_pivo, :]
            matriz_aumentada[linha_do_pivo, :] = temp_linha

        if matriz_aumentada[j, j] == 0:
            if num_linhas == num_colunas:
                return None
            else:
                continue

        for i in range(j + 1, num_linhas):
            fator = matriz_aumentada[i, j] / matriz_aumentada[j, j]
            matriz_aumentada[i, :] = matriz_aumentada[i, :] - fator * matriz_aumentada[j, :]

    if num_linhas != num_colunas:
        return None

    x = np.zeros(num_colunas)
    for i in range(num_colunas - 1, -1, -1):
        if matriz_aumentada[i, i] == 0:
            return None
        x[i] = (matriz_aumentada[i, num_colunas] - np.dot(matriz_aumentada[i, i+1:num_colunas], x[i+1:num_colunas])) / matriz_aumentada[i, i]

    return x