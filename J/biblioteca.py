import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, Tuple, List

def PlotFunctionGraph(x1: float, x2: float, f: Callable[[np.ndarray], np.ndarray], points: int = 1000, height: int = 6, width: int = 10) -> None:
    """
    Plots the graph of a vectorized function f(x) over the interval [x1, x2].

    Generates a smooth curve by evaluating the function at evenly spaced points
    and displays it with coordinate axes and grid lines for reference.

    Args:
        x1 (float): Left boundary of the interval
        x2 (float): Right boundary of the interval
        f (Callable[[np.ndarray], np.ndarray]): Vectorized function to plot
        points (int, optional): Number of evaluation points. Defaults to 1000.
        height (int, optional): Plot height in inches. Defaults to 6.
        width (int, optional): Plot width in inches. Defaults to 10.

    Returns:
        None

    Raises:
        TypeError: If f is not a callable function
        ValueError: If x1 >= x2
    """

    # Input validation
    if not callable(f):
        raise TypeError("Argument 'f' must be a callable function")
    
    if x1 >= x2:
        raise ValueError("Invalid interval: x1 must be less than x2")

    # Generate data
    x = np.linspace(x1, x2, points)
    y = f(x)

    # Configure plot
    plt.figure(figsize=(width, height))
    plt.plot(x, y, label="f(x)", color='blue')
    plt.axhline(0, color='red', linewidth=0.8)
    plt.axvline(0, color='black', linewidth=0.8)
    plt.title('Graph of the Function f(x)')
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()

def Bisection(a: float, b: float, f: Callable[[float], float], tol: float = 1e-10) -> Tuple[float, List[float]]:
    """
    Finds a root of the function f in the interval [a, b] using the Bisection method.

    The method repeatedly divides the interval [a, b] in half and selects the subinterval
    where the function changes sign, continuing until the length of the interval is
    less than the specified tolerance.

    Args:
        a (float): Left endpoint of the initial interval.
        b (float): Right endpoint of the initial interval.
        f (Callable[[float], float]): Continuous function to find the root of.
        tol (float, optional): Tolerance for the stopping criterion (half the interval size). 
                               Defaults to 1e-10.

    Returns:
        Tuple[float, List[float]]: A tuple containing:
            - Approximate root (float)
            - List of errors (half the interval size) at each iteration (List[float])

    Raises:
        TypeError: If f is not a callable function.
        ValueError: If a >= b or if f(a) and f(b) have the same sign.
    """

    if not callable(f):
        raise TypeError("The argument 'f' must be a callable function")
    
    if a >= b:
        raise ValueError("Invalid interval: a must be less than b")
    
    fa = f(a)
    fb = f(b)

    if fa * fb >= 0:
        if fa == 0:
            return a, [] 
        if fb == 0:
            return b, []
        raise ValueError("f(a) and f(b) must have opposite signs to apply the bisection method, or one of the endpoints is already the root.")

    errors: List[float] = []
    
    while (b - a) / 2 > tol:
        midpoint = (a + b) / 2
        f_midpoint = f(midpoint)
        
        errors.append((b - a) / 2)
        
        if fa * f_midpoint < 0:
            b = midpoint
            fb = f_midpoint
        else:
            a = midpoint
            fa = f_midpoint
        
    root = (a + b) / 2
    return root, errors

def Secant(a: float, b: float, f: Callable[[float], float], error: float = 1e-10, max_iter: int = 1000) -> Tuple[float, List[float]]:
    """
    Finds a root of a function using the secant method.

    The secant method is an iterative numerical method for finding successively
    better approximations to the root (zero) of a real-valued function.  It
    approximates the root using a sequence of secant lines.

    Args:
        a (float): An initial guess for the root.
        b (float): Another initial guess for the root, different from `a`.
        f (Callable[[float], float]): The function for which the root is sought.
            Must be a callable that accepts a float and returns a float.
        error (float, optional): The desired accuracy (tolerance). The iteration
            stops when the absolute difference between successive approximations
            is less than this value. Defaults to 1e-10.
        max_iter (int, optional): The maximum number of iterations allowed.
            Defaults to 1000.

    Returns:
        tuple[float, list[float]]: A tuple containing:
            - The approximate root (float).
            - A list of absolute errors at each iteration (list[float]).

    Raises:
        ValueError: If `f(a)` and `f(b)` are equal, indicating a division by zero
            in the secant method formula.
    """
    
    if f(a) == f(b):
        raise ValueError("f(a) and f(b) cannot be equal in the secant method.")

    errors = []
    iterations = 0

    while abs(b - a) > error and iterations < max_iter:
        c = b - f(b) * (b - a) / (f(b) - f(a))

        errors.append(abs(c - b))

        a, b = b, c
        iterations += 1

    return b, errors

def FakePosition(a: float, b: float, f: Callable[[float], float], erro: float = 1e-10, max_iter: int = 1000) -> Tuple[float, List[float]]:
    """
    Finds a root of a function using the false position method.

    The false position method is an iterative numerical method for finding successively
    better approximations to the root (zero) of a real-valued function.  It
    approximates the root using a sequence of secant lines. It is similar to the Secant
    Method, but it guarantees that a root is bracketed.

    Args:
        a (float): Left endpoint of the initial interval.
        b (float): Right endpoint of the initial interval.
        f (Callable[[float], float]): The function for which the root is sought.
            Must be a callable that accepts a float and returns a float.
        erro (float, optional): The desired accuracy (tolerance). The iteration
            stops when the absolute difference between successive approximations
            is less than this value. Defaults to 1e-10.
        max_iter (int, optional): The maximum number of iterations allowed.
            Defaults to 1000.

    Returns:
        Tuple[float, list[float]]: A tuple containing:
            - The approximate root (float).
            - A list of absolute errors at each iteration (list[float]).

    Raises:
        ValueError: If f(a) * f(b) >= 0, indicating that the function does not have
            opposite signs at the endpoints of the interval.
    """

    if f(a) * f(b) >= 0:
        raise ValueError("f(a) * f(b) >= 0: o método da falsa posição requer sinais opostos em a e b.")

    erros = []
    iteracoes = 0
    c = a

    while iteracoes < max_iter:
        c_novo = b - f(b) * (b - a) / (f(b) - f(a))
        erros.append(abs(c_novo - c))

        c = c_novo
        f_c = f(c)

        if abs(f_c) < erro or abs(b - a) < erro:
            break

        if f(a) * f_c < 0:
            b = c
        else:
            a = c

        iteracoes += 1

    return c, erros

def LinearRegression(x, y) -> Tuple[float, float]:
    """
    Performs simple linear regression on the given x and y data points.

    This function calculates the coefficients of a linear regression line (y = b0 + b1*x)
    where:
        - b0 is the y-intercept
        - b1 is the slope of the line

    Args:
        x (array-like): 1D array of independent variable data points.
        y (array-like): 1D array of dependent variable data points.

    Returns:
        Tuple[float, float]: A tuple containing:
            - b0 (float): The y-intercept of the regression line.
            - b1 (float): The slope of the regression line.

    Raises:
        ValueError: If the lengths of x and y are not equal.
        TypeError: If x or y are not array-like structures.
        ValueError: If the denominator in the linear regression calculation is zero.
        ValueError: If x or y are not 1D arrays.
    """

    try:
        x = np.array(x)
        y = np.array(y)
    except Exception as e:
        raise TypeError("x and y must be array-like structures (e.g., lists, numpy arrays).") from e

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D arrays.")

    if len(x) != len(y):
        raise ValueError("x & y arrays must have the same length.")

    n = len(x)
    sum_y = np.sum(y)
    sum_x = np.sum(x)
    sum_xy = np.sum(x*y)
    sum_x2 = np.sum(x**2)

    denominator = n * sum_x2 - sum_x**2
    if denominator == 0:
        raise ValueError("Denominator in linear regression calculation is zero. Check your data.")

    b1 = (n * sum_xy - sum_x * sum_y) / denominator
    b0 = (sum_y - b1 * sum_x) / n

    return b0, b1

def PlotLinearRegression(x, y):
    """
    Plots a scatter plot of the input data and the best-fit linear regression line.

    This function computes the linear regression coefficients (b0 and b1) using the
    `linear_regression` function and plots both the original data points and the
    resulting regression line.

    Args:
        x (array-like): 1D array of independent variable data points.
        y (array-like): 1D array of dependent variable data points.

    Returns:
        None

    Raises:
        ValueError: If the lengths of x and y are not equal.
        TypeError: If x or y are not array-like structures.
        ValueError: If the denominator in the linear regression calculation is zero.
        ValueError: If x or y are not 1D arrays.
    """
    
    b0, b1 = linear_regression(x, y)
    x = np.asarray(x)

    x_reg = np.linspace(x.min(), x.max(), 100)
    y_reg = b0 + b1 * x_reg

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, color='blue', label='Data points')
    plt.plot(x_reg, y_reg, color='red', label=f'Regression line (y = {b0:.4f} + {b1:.4f}x)')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Regression')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
