# Pacote de funções 2
# João Pedro Cobra Casici

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

def decomp_lu(A):
    import numpy as np
    n = A.shape[0]
    L = np.zeros_like(A, dtype=float)
    U = np.zeros_like(A, dtype=float)
    for i in range(n):
        L[i, i] = 1.0
        for j in range(i, n):
            U[i, j] = A[i, j] - sum(L[i, k] * U[k, j] for k in range(i))
        for j in range(i+1, n):
            if U[i, i] == 0:
                raise ValueError("Zero na diagonal de U. Não é possível decompor sem pivotamento.")
            L[j, i] = (A[j, i] - sum(L[j, k] * U[k, i] for k in range(i))) / U[i, i]
    return L, U

import numpy as np

def decomp_lu_pivot(A, b):
    n = A.shape[0]
    L = np.identity(n, dtype=float)
    U = A.astype(float).copy()
    p = np.arange(n)

    for k in range(n - 1):
        pivot_row_index = np.argmax(np.abs(U[k:, k])) + k
        
        if pivot_row_index != k:
            U[[k, pivot_row_index]] = U[[pivot_row_index, k]]
            p[[k, pivot_row_index]] = p[[pivot_row_index, k]]
            if k > 0:
                L[[k, pivot_row_index], :k] = L[[pivot_row_index, k], :k]
        
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]
            U[i, k] = 0
    
    b_permuted = b[p]

    y = np.zeros(n)
    for i in range(n):
        y[i] = b_permuted[i] - np.dot(L[i, :i], y[:i])

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
        
    return x

def lagrange_interpol(x, y, x_interp):
    import numpy as np
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    x_interp = np.array(x_interp, dtype=float)
    def L(k, x0):
        prod = 1.0
        for i in range(len(x)):
            if i != k:
                prod *= (x0 - x[i]) / (x[k] - x[i])
        return prod
    if x_interp.ndim == 0:
        return sum(y[k] * L(k, x_interp) for k in range(len(x)))
    else:
        return np.array([sum(y[k] * L(k, xi) for k in range(len(x))) for xi in x_interp])

def trapezio_simples(f, a, b):
    return (b - a) * (f(a) + f(b)) / 2

def trapezio_composto(f, a, b, n=100):
    import numpy as np

    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    return (h/2) * (y[0] + 2 * np.sum(y[1:-1]) + y[-1])

def gauss_sem_pivotamento(A, b):
    import numpy as np
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    Ab = np.hstack([A, b.reshape(-1, 1)])

    for k in range(n-1):
        if Ab[k, k] == 0:
            print(f"Zero na diagonal principal em linha {k}. Não é possível resolver sem pivotamento.")
            return None
        for i in range(k+1, n):
            m = Ab[i, k] / Ab[k, k]
            Ab[i, k:] = Ab[i, k:] - m * Ab[k, k:]

    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        if Ab[i, i] == 0:
            print(f"Zero na diagonal principal em linha {i}. Sistema impossível ou indeterminado.")
            return None
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:n])) / Ab[i, i]
    return x

def decomp_cholesky(A):
    import numpy as np
    n = A.shape[0]
    L = np.zeros_like(A)
    for i in range(n):
        for j in range(i+1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                L[i][j] = np.sqrt(A[i][i] - s)
            else:
                L[i][j] = (A[i][j] - s) / L[j][j]
    return L