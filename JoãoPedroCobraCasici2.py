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

def avaliar_polinomio(coef, x):
    import numpy as np
    resultado = 0
    grau = len(coef) - 1
    for i in range(len(coef)):
        resultado += coef[i] * x**(grau - i)
    return resultado

def polyfit_manual(x, y, grau):
    import numpy as np
    A = np.zeros((len(x), grau + 1))
    for i in range(len(x)):
        for j in range(grau + 1):
            A[i, j] = x[i]**(grau - j)

    A_T_A = A.T @ A 
    A_T_y = A.T @ y
    coef = np.linalg.solve(A_T_A, A_T_y)
    
    return coef

def melhor_grau(x, y, grau_max):
    import numpy as np
    melhor_rmse = float('inf')
    melhor_grau = 1
    melhor_coef = None
    
    for grau in range(1, grau_max + 1):
        coef = polyfit_manual(x, y, grau)
        
        y_pred = np.array([avaliar_polinomio(coef, xi) for xi in x])
        rmse = np.sqrt(np.mean((y - y_pred)**2))
        
        if rmse < melhor_rmse:
            melhor_rmse = rmse
            melhor_grau = grau
            melhor_coef = coef
            
    return melhor_grau, melhor_coef

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
    import numpy as np
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

def integral_num_simples(func, x_ini, x_fin):
  return ((func(x_ini) + func(x_fin)) / 2) * (x_fin - x_ini)

def integral_numerica(func, x_ini, x_fin, n_points=100_000):
    import numpy as np
    points = np.linspace(x_ini, x_fin, n_points)
    y = func(points)
    h = (x_fin - x_ini) / (n_points - 1)
    integral = h * (np.sum(y) - (y[0] + y[-1]) / 2)
    return integral

def antiderivada(x):
  import numpy as np
  return (np.e**x / 101) * (np.sin(10*x) - 10 * np.cos(10*x)) + 8 * x

def integral_trapezio_composta(func, x_ini, x_fin, n_points=1000):
    import numpy as np
    x = np.linspace(x_ini, x_fin, n_points)
    y = func(x)
    h = (x_fin - x_ini) / (n_points - 1)
    return h * (np.sum(y) - (y[0] + y[-1]) / 2)

def gerar_tabela_diferencas(x_i, y_i):
    n = len(x_i)

    tabela_diferencas = np.full((n, n), np.nan)
    tabela_diferencas[:, 0] = y_i

    for j in range(1, n):
        for i in range(n - j):
            numerador = tabela_diferencas[i+1, j-1] - tabela_diferencas[i, j-1]
            denominador = x_i[i+j] - x_i[i]
            tabela_diferencas[i, j] = numerador / denominador

    largura_i = 5
    largura_x = 10
    largura_dados = 12

    partes_header = [f"{'i':<{largura_i}}", f"{'x_i':<{largura_x}}"]
    partes_header.append(f"{'y_i':<{largura_dados}}")
    for k in range(1, n):
        partes_header.append(f"{f'D^{k} y_i':<{largura_dados}}")

    header_str = " | ".join(partes_header)

    separador_str = "-" * len(header_str)

    tabela_final_str = f"{header_str}\n{separador_str}\n"

    for i in range(n):
        partes_linha = [
            f"{i:<{largura_i}}",
            f"{x_i[i]:<{largura_x}.4f}"
        ]
        for j in range(n):
            valor = tabela_diferencas[i, j]
            if np.isnan(valor):
                partes_linha.append(f"{'':<{largura_dados}}")
            else:
                partes_linha.append(f"{valor:<{largura_dados}.4f}")

        tabela_final_str += " | ".join(partes_linha) + "\n"

    return tabela_final_str

def polinomio_lagrange(x_data, y_data, x_eval):
    import numpy as np
    
    try:
        x_d = np.asarray(x_data, dtype=float)
        y_d = np.asarray(y_data, dtype=float)
    except Exception as e:
        raise TypeError("x_data and y_data must be convertible to NumPy float arrays.") from e

    if x_d.ndim != 1 or y_d.ndim != 1:
        raise ValueError("x_data and y_data must be 1D arrays.")
    if len(x_d) != len(y_d):
        raise ValueError("x_data and y_data must have the same length.")
    if len(x_d) == 0:
        raise ValueError("Input arrays x_data and y_data cannot be empty.")
    if len(np.unique(x_d)) != len(x_d):
        raise ValueError("Points in x_data must be unique for Lagrange interpolation.")

    n_points = len(x_d)
    _is_scalar_eval = np.isscalar(x_eval)

    try:
        x_e_arr = np.atleast_1d(np.asarray(x_eval, dtype=float))
    except Exception as e:
        raise TypeError("x_eval must be a scalar or array-like type convertible to a NumPy float array.") from e

    if x_e_arr.ndim > 1:
        raise ValueError("If an array, x_eval must be 1D.")

    P_x_eval_arr = np.zeros_like(x_e_arr, dtype=float)

    for j in range(n_points):
        yj = y_d[j]
        xj = x_d[j]
        
        L_j_x = np.ones_like(x_e_arr, dtype=float)
        for i in range(n_points):
            if i == j:
                continue
            xi = x_d[i]
            L_j_x *= (x_e_arr - xi) / (xj - xi)

        P_x_eval_arr += yj * L_j_x

    if _is_scalar_eval:
        return P_x_eval_arr[0]
    else:
        return P_x_eval_arr

def diferenca_dividida(x, y, i, j):
    import numpy as np

    try:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
    except (TypeError, ValueError):
        raise TypeError("Inputs x and y must be array-like and numeric.")

    n = len(x)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D arrays.")
    if n != len(y):
        raise ValueError("x and y must have the same length.")
    if n == 0:
        raise ValueError("Input arrays cannot be empty.")
    if len(np.unique(x)) != len(x):
        raise ValueError("All x values must be unique for this method.")
    if not (0 <= j < n and 0 <= i < n - j):
        raise ValueError(f"Indices i={i}, j={j} are out of bounds for n={n}.")

    memo = {}

    def _calculate_diff(start_idx, order):
        if (start_idx, order) in memo:
            return memo[(start_idx, order)]

        if order == 0:
            return y[start_idx]

        numerator = (_calculate_diff(start_idx + 1, order - 1) -
                     _calculate_diff(start_idx, order - 1))
        
        denominator = x[start_idx + order] - x[start_idx]
        
        result = numerator / denominator
        
        memo[(start_idx, order)] = result
        return result

    return _calculate_diff(i, j)

def tabela_diferencas_divididas(x, y):
    import numpy as np

    try:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
    except (TypeError, ValueError):
        raise TypeError("Inputs x and y must be array-like and numeric.")

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D arrays.")
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")
    if len(x) == 0:
        raise ValueError("Input arrays cannot be empty.")
    if len(np.unique(x)) != len(x):
        raise ValueError("All x values must be unique for this method.")

    n = len(x)
    diff_table = np.full((n, n), np.nan)
    diff_table[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            numerator = diff_table[i + 1, j - 1] - diff_table[i, j - 1]
            denominator = x[i + j] - x[i]
            diff_table[i, j] = numerator / denominator

    index_width = 5
    x_width = 10
    data_width = 12

    header_parts = [
        f"{'i':<{index_width}}",
        f"{'x_i':<{x_width}}",
        f"{'y_i':<{data_width}}"
    ]
    for k in range(1, n):
        header_parts.append(f"{f'D^{k} y_i':<{data_width}}")

    header_str = " | ".join(header_parts)
    separator_str = "-" * len(header_str)
    
    table_string = f"{header_str}\n{separator_str}\n"

    for i in range(n):
        row_parts = [
            f"{i:<{index_width}}",
            f"{x[i]:<{x_width}.4f}"
        ]
        for j in range(n):
            value = diff_table[i, j]
            if np.isnan(value):
                row_parts.append(f"{'':<{data_width}}")
            else:
                row_parts.append(f"{value:<{data_width}.4f}")
        
        table_string += " | ".join(row_parts) + "\n"

    return table_string

def integracao_numerica_simples(func, x_ini, x_fin):
    return ((func(x_ini) + func(x_fin)) / 2) * (x_fin - x_ini)

def integracao_numerica(func, x_ini, x_fin, n_points=100_000):
    import numpy as np

    if n_points < 3:
        raise ValueError("n_points must be 3 or greater to have at least one interior point.")

    points = np.linspace(x_ini, x_fin, n_points)
    step_size = (x_fin - x_ini) / (n_points - 1)
    
    integral_sum = np.sum(func(points[1:-1]))
    
    return integral_sum * step_size

def plotar_comparacao_area_integracao(func, antiderivative, x_start, x_end, func_label="f(x)", integration_method=None, method_name="Numerical Method"):
    import numpy as np
    import matplotlib.pyplot as plt

    if integration_method is None:
        integration_method = lambda f, a, b: ((f(a) + f(b)) / 2) * (b - a)
        method_name = "Simple Trapezoid"

    exact_value = antiderivative(x_end) - antiderivative(x_start)
    approx_value = integration_method(func, x_start, x_end)

    print(f"Exact Integral Value: {exact_value:.6f}")
    print(f"Approximate Value ({method_name}): {approx_value:.6f}")
    print(f"Absolute Error: {abs(exact_value - approx_value):.6f}")

    plt.figure(figsize=(12, 7))

    plot_range_start = x_start - 0.2 * (x_end - x_start)
    plot_range_end = x_end + 0.2 * (x_end - x_start)
    x_curve = np.linspace(min(0, plot_range_start), plot_range_end, 500)
    y_curve = func(x_curve)

    plt.plot(x_curve, y_curve, 'k', linewidth=2.5, label=f'Curve of ${func_label}$')

    x_fill_exact = np.linspace(x_start, x_end, 200)
    y_fill_exact = func(x_fill_exact)
    plt.fill_between(
        x_fill_exact, y_fill_exact, color='blue', alpha=0.4,
        label=f'Exact Area ≈ {exact_value:.4f}'
    )

    trap_x_points = [x_start, x_end, x_end, x_start]
    trap_y_points = [func(x_start), func(x_end), 0, 0]
    plt.fill(
        trap_x_points, trap_y_points, edgecolor='red', facecolor='red', alpha=0.4,
        label=f'Approx. Area (Visual) = {approx_value:.4f}'
    )

    plt.title(f'Exact Integral vs. {method_name} Approximation')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.legend()
    plt.show()

def interpol_newton(x, y, x_val):
    import numpy as np

    try:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
    except (TypeError, ValueError):
        raise TypeError("Inputs x and y must be array-like and numeric.")

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D arrays.")
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")
    if len(x) == 0:
        raise ValueError("Input arrays cannot be empty.")
    if len(np.unique(x)) != len(x):
        raise ValueError("All x values must be unique for Newton's method.")

    n = len(x)
    divided_diffs = np.zeros((n, n))
    divided_diffs[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            numerator = divided_diffs[i + 1, j - 1] - divided_diffs[i, j - 1]
            denominator = x[i + j] - x[i]
            divided_diffs[i, j] = numerator / denominator

    coeffs = divided_diffs[0]
    
    result = coeffs[0]
    product_term = 1.0

    for j in range(1, n):
        product_term *= (x_val - x[j - 1])
        result += coeffs[j] * product_term

    return result

def subst_progressiva(L, C):
    import numpy as np

    try:
        L = np.asarray(L, dtype=float)
        C = np.asarray(C, dtype=float)
    except (TypeError, ValueError):
        raise TypeError("Inputs L and C must be array-like and numeric.")

    n = L.shape[0]
    if L.shape != (n, n) or C.shape != (n,):
        raise ValueError("Incompatible dimensions: L must be n-by-n and C must be n.")

    x = np.zeros(n, dtype=float)
    for i in range(n):
        if L[i, i] == 0:
            raise ValueError(f"Zero on diagonal at L[{i},{i}]. Matrix is singular.")

        dot_product = np.dot(L[i, :i], x[:i])

        x[i] = (C[i] - dot_product) / L[i, i]

    return x

def subist_regressiva(U, C):
    import numpy as np
    
    try:
        U = np.asarray(U, dtype=float)
        C = np.asarray(C, dtype=float)
    except (TypeError, ValueError):
        raise TypeError("Inputs U and C must be array-like and numeric.")

    n = U.shape[0]
    if U.shape != (n, n) or C.shape != (n,):
        raise ValueError("Incompatible dimensions: U must be n-by-n and C must be n.")

    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        if U[i, i] == 0:
            raise ValueError(f"Zero on diagonal at U[{i},{i}]. Matrix is singular.")

        dot_product = np.dot(U[i, i + 1:], x[i + 1:])

        x[i] = (C[i] - dot_product) / U[i, i]

    return x
