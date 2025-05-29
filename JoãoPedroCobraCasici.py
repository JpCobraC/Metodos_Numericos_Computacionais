def zero_do_computador():
    e = 1.0
    while (e + 1.0 > 1.0):
        e = e / 2.0
    return e


def seno_taylor(x, k_max):
    resultado = 0.0
    sinal = 1
    for k in range(k_max):
        numerador = x ** (2 * k + 1)
        denominador = fatorial(2 * k + 1)
        resultado += sinal * numerador / denominador
        sinal *= -1
    return resultado


def fatorial(n):
    resultado = 1
    for i in range(2, n + 1):
        resultado *= i
    return resultado


def bisseccao(f, a, b, tol=1e-6, max_iter=100):
    if f(a) * f(b) >= 0:
        raise ValueError("f(a) e f(b) devem ter sinais opostos.")
    for _ in range(max_iter):
        m = (a + b) / 2
        if abs(f(m)) < tol or (b - a) / 2 < tol:
            return m
        if f(a) * f(m) < 0:
            b = m
        else:
            a = m
    return (a + b) / 2


def secante(f, x0, x1, tol=1e-6, max_iter=100):
    for _ in range(max_iter):
        if f(x1) == f(x0):
            print("Divisão por zero detectada.")
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        if abs(x2 - x1) < tol:
            return x2
        x0, x1 = x1, x2
    return None


def falsa_posicao(f, a, b, tol=1e-6, max_iter=100):
    if f(a) * f(b) >= 0:
        print("f(a) e f(b) devem ter sinais opostos.")
    for _ in range(max_iter):
        c = (a * f(b) - b * f(a)) / (f(b) - f(a))
        if abs(f(c)) < tol:
            return c
        if f(c) * f(a) < 0:
            b = c
        else:
            a = c
    return None


def ajuste_linear(x, y):
    n = len(x)
    soma_x = sum(x)
    soma_y = sum(y)
    soma_xy = sum([xi * yi for xi, yi in zip(x, y)])
    soma_x2 = sum([xi * xi for xi in x])

    b1 = (n * soma_xy - soma_x * soma_y) / (n * soma_x2 - soma_x ** 2)
    b0 = (soma_y - b1 * soma_x) / n
    return b0, b1


def vandermonde(x, y):
    n = len(x)
    V = [[x[i] ** j for j in range(n)] for i in range(n)]
    return eliminacao_gaussiana(V, y)


def eliminacao_gaussiana(A, b):
    n = len(A)
    for k in range(n):
        for i in range(k + 1, n):
            if A[k][k] == 0:
                print("Divisão por zero detectada na matriz.")
            fator = A[i][k] / A[k][k]
            for j in range(k, n):
                A[i][j] -= fator * A[k][j]
            b[i] -= fator * b[k]
    x = [0] * n
    for i in range(n - 1, -1, -1):
        soma = sum([A[i][j] * x[j] for j in range(i + 1, n)])
        x[i] = (b[i] - soma) / A[i][i]
    return x


def avaliar_polinomio(x_val, coef):
    return sum(coef[i] * x_val ** i for i in range(len(coef)))


def plotar_interpolacao(x, y):
    import matplotlib.pyplot as plt
    import numpy as np

    coef = vandermonde(x, y)
    P = lambda xi: sum(coef[i] * xi**i for i in range(len(coef)))

    x_plot = np.linspace(min(x), max(x), 300)
    y_plot = [P(xi) for xi in x_plot]

    plt.figure()
    plt.plot(x_plot, y_plot, label="Polinômio Interpolador", color="green")
    plt.scatter(x, y, color='red', label="Pontos")
    plt.title("Interpolação Polinomial")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    plt.show()

    return coef

def avaliar_polinomio(x_val, coef):
    return sum(coef[i] * x_val ** i for i in range(len(coef)))

def reg_lin_mult(x1,x2,y,n):
    import numpy as np

    m1=np.array([[n,sum(x1),sum(x2)],[sum(x1),sum(x1)*sum(x1),sum(x2*x1)],[sum(x2),sum(x1*x2),sum(x2*x2)]])
    vetor1=np.array([sum(y),sum(x1*y),sum(x2*y)])
    b0,b1,b2=np.linalg.solve(m1,vetor1)
    print("Reta dos Mínimos Quadrados:",round(b0,4),"+ (",round(b1,4),")x1 + (",round(b2,4),")x2")
    u=b0+b1*x1+b2*x2
    R=np.sum((y-u)**2)
    r2= 1-(R/(np.sum(y**2)-((np.sum(y))**2)/n))
    print("Coeficiente de determinação para a reta dos mínimos quadrados:",r2)
    return None