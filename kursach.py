from ast import parse
import math
import sympy
import matplotlib.pyplot as plt
import numpy as np
import click
import time
from sympy.utilities.lambdify import lambdastr
from sympy.parsing.mathematica import parse_mathematica


class NumMethods:
    func = 1
    spaces = []
    answers = []
    tograph = []
    time = 0
    iterations_num = 0

    def __init__(self) -> None:
        pass

    def makeaplot(self, graph_type=None, start=-10, end=10):
        print(self.answers)
        print(self.func(0))

        fig, ax = plt.subplots()
        x = np.arange(start, end, 0.01)
        y = np.vectorize(self.func)
        ax.plot(x, y(x), color="red", label="исходная функция")
        ax.plot(x, np.zeros_like(x), color="black")
        ax.scatter(
            self.answers,
            np.zeros_like(self.answers),
            color="#363636",
            linewidths=5,
        )

        ax.grid(which="major")
        ax.spines["left"].set_position("zero")
        ax.spines["right"].set_color("none")
        ax.spines["bottom"].set_position("zero")
        ax.spines["top"].set_color("none")
        fig.set_size_inches(10, 6)

        if graph_type == "MPI":
            for root in self.answers:
                y = [start, end]
                x = np.ones_like(y) * root
                ax.plot(x, y, color="black", linestyle="dashed")
            ax.plot([start, end], [start, end], "y--", label="y=x")

            for element in self.tograph:
                xax = np.arange(start, end, 0.01)
                x = sympy.symbols("x")
                y = np.vectorize(lambda n: element.evalf(subs={x: n}))
                ax.scatter(
                    self.answers,
                    0,
                    color="#363636",
                    linewidths=5,
                )
                y = y(xax)
                ax.plot(xax, y, color="green", label=str(sympy.simplify(element)))

        # print(round(1.2*min(self.answers)) if round(1.2*min(self.answers)) < 0 else -1, round(1.2*max(self.answers)) if round(1.2*min(self.answers)) > 0 else 1)
        # xmin, xmax = ax.set_xlim(float(1.2*min(self.answers) if 1.2*min(self.answers) < 0 else -1), float(1.2*max(self.answers) if 1.2*min(self.answers) > 0 else 1))
        ax.set_xlim(start, end)
        ax.set_ylim(start, end)
        # ax.set_ylim(float(self.func(xmin)), float(self.func(xmax)))

        ax.legend()
        plt.show()

    def proiz(self, x, func):
        return (func(x + 0.001) - func(x)) / 0.001

    def lim(self, x):
        if x == sympy.oo:
            x = 1e3
        elif x == sympy.oo * -1:
            x = -1e3
        else:
            return x
        return x

    def answer_spaces_search(self, func, a, b, deepness=10, depth=0):
        # print("checking [", a, b, "]")
        c = (a + b) / 2
        print(func(a) * func(c))
        if func(a) * func(c) > 0 and depth < deepness:
            self.answer_spaces_search(func, a, c, deepness=deepness, depth=depth + 1)
        elif func(a) * func(c) <= 0 and depth < deepness:
            self.answer_spaces_search(func, a, c, deepness=deepness, depth=depth + 1)
        elif func(a) * func(c) <= 0 and depth >= deepness:
            self.spaces.append([a, c])
            print("appending [{}, {}]".format(a, c))
        else:
            None

        if func(b) * func(c) > 0 and depth < deepness:
            self.answer_spaces_search(func, c, b, deepness=deepness, depth=depth + 1)
        elif func(b) * func(c) <= 0 and depth < deepness:
            self.answer_spaces_search(func, c, b, deepness=deepness, depth=depth + 1)
        elif func(b) * func(c) <= 0 and depth >= deepness:
            self.spaces.append([c, b])
            # print("appending [{}, {}]".format(c, b))
        else:
            None

    def Half(self, a, b, eps=0.01, func="x"):
        self.answers = []
        x = sympy.symbols("x")
        alim, blim = sympy.limit(func, x, a), sympy.limit(func, x, a)
        func_callable = eval("lambda x: " + str(func))
        self.func = func_callable
        a, b = self.lim(a), self.lim(b)
        self.answer_spaces_search(func_callable, a, b)
        # print(self.spaces)
        # print(func_callable(-0.1), func_callable(0), func_callable(0.1))

        for element in self.spaces:
            exit_flag = False
            a, b = element[0], element[1]
            ans, prevans = self.func((a + b) / 2), -1 * self.func((a + b) / 2) - 1
            # print(ans, prevans)
            while abs(ans - prevans) >= eps:
                for element in [a, b, (a + b) / 2]:
                    if self.func(element) == 0:
                        self.answers.append(element)
                        exit_flag = True
                if exit_flag:
                    break
                if self.func(a) * self.func((a + b) / 2) < 0:
                    b = (a + b) / 2
                    prevans = ans
                    ans = self.func((a + b) / 2)
                elif self.func((a + b) / 2) * self.func(b) < 0:
                    a = (a + b) / 2
                    prevans = ans
                    ans = self.func((a + b) / 2)
            # print(
            #     "a is {}, b is {} and delta is {}, {} will be written".format(
            #         a, b, abs(ans - prevans), (a + b) / 2
            #     )
            # )
            self.answers.append(round((a + b) / 2, int(abs(math.log10(eps)))))
        # print(self.answers)
        return np.unique(self.answers)

    def MPI(self, start, end, eps, left, right, x):
        self.answers = []
        left, right = parse_mathematica(left), parse_mathematica(right)
        print(left, right)
        self.func = sympy.lambdify(x, left - right)
        eps, start, end = float(eps), float(start), float(end)
        self.answer_spaces_search(self.func, start, end, deepness=4)

        left = sympy.simplify(left + x - right)
        sympy_f = sympy.simplify(left - x)
        right = x

        for interval in self.spaces:
            print("searching in interval {}\n sympy_f = {}".format(interval, sympy_f))
            print(sympy.diff(sympy_f, x))

            try:
                max = sympy.maximum(
                    sympy.diff(sympy_f, x), x, sympy.Interval(interval[0], interval[1])
                )
            except NotImplementedError:
                dif = sympy.lambdify(x, sympy.diff(sympy_f, x))
                dif = np.vectorize(dif)
                max = np.max(dif(np.arange(interval[0], interval[1] + eps, eps)))

            print("maximum is ", max)
            if max != sympy.oo and max <= 1e20:
                lmbd = 1 / max
                new_left = x - lmbd * sympy_f
                self.tograph.append(new_left)
                x0 = interval[0] - 10
                xn = interval[0]
                while abs(xn - x0) > eps:
                    x0 = xn
                    xn = new_left.evalf(subs={x: x0})
                    print(x0, xn, abs(xn - x0))
                self.answers.append(round(xn, int(abs(math.log10(eps)))))
            else:
                pass
        return np.unique(self.answers)

    def Newton(self, start, end, eps, left, right, x):
        self.answers = []
        left, right = parse_mathematica(left), parse_mathematica(right)
        self.func = sympy.lambdify(x, left - right)
        eps, start, end = float(eps), float(start), float(end)
        self.answer_spaces_search(self.func, start, end, deepness=4)

        sympy_f = left - right
        print("sympy_f={}".format(sympy_f))

        for interval in self.spaces:
            print("searching in interval {}\n sympy_f = {}".format(interval, sympy_f))
            print(sympy.diff(sympy_f, x))
            try:
                max = sympy.maximum(
                    sympy.diff(sympy_f, x), x, sympy.Interval(interval[0], interval[1])
                )
            except NotImplementedError:
                dif = sympy.lambdify(x, sympy.diff(sympy_f, x))
                dif = np.vectorize(dif)
                max = np.max(dif(np.arange(interval[0], interval[1] + eps, eps)))
            if max != sympy.oo and max <= 1e20:
                new_left = x - sympy_f / sympy.diff(sympy_f)
                x0 = float(interval[0] - 10)
                xn = float(interval[0])
                while abs(self.func(float(xn)) - self.func(float(x0))) > eps:
                    x0 = xn
                    xn = new_left.evalf(subs={x: x0})
                self.answers.append(round(xn, int(abs(math.log10(eps)))))
            else:
                pass
        return np.unique(self.answers)

    def MatrixMPI(
        self, matrix, vec, eps
    ):  # сначала матрицу надо привести к нормальному виду
        start_time, iters = time.time(), 0
        for _ in range(len(matrix)):
            print(matrix)
            for i in range(len(matrix)):
                matcopy = np.array(list(map(abs, np.copy(matrix[i]))))
                print(np.max(matcopy), matcopy[i])
                if np.max(matcopy) == matcopy[i]:
                    # print("matcopy is {}, max is in element {}".format(matcopy, i))
                    pass
                else:
                    print("переставляю")
                    ind = np.where(matcopy == np.max(matcopy))[0]
                    if i not in ind and matrix[i, i] < matrix[ind, i]:
                        matrix[[i, ind[0]]] = matrix[[ind[0], i]]
                        vec[[i, ind[0]]] = vec[[ind[0], i]]
                    break
        for i in range(  # приведение матрицы к сходящейся
            len(matrix)
        ):  # i соответствует номеру строки и т. к. матрица квадратная-номеру столбца для искомого решения
            if matrix[i, i] >= 0:
                matrix[i], vec[i] = matrix[i] * -1, vec[i] * -1
            alpha = 1 / abs(matrix[i, i])
            matrix[i], vec[i] = matrix[i] * alpha, vec[i] * alpha
            Cmat = np.copy(matrix[i])
            Cmat[i] += 1
            if np.linalg.norm(Cmat) > 1:
                prevnorm, k = np.linalg.norm(Cmat), 1.1
                matrix[i], vec[i] = matrix[i] / k, vec[i] / k
                Cmat = np.copy(matrix[i])
                Cmat[i] += 1
                norm = np.linalg.norm(Cmat)
                while norm > 1:
                    prevnorm, k = np.linalg.norm(Cmat), 1.1
                    matrix[i], vec[i] = matrix[i] / k, vec[i] / k
                    Cmat = np.copy(matrix[i])
                    Cmat[i] += 1
                    norm = np.linalg.norm(Cmat)
                    k += 0.1
                    if norm > prevnorm:
                        return "error"
            print(matrix)
        # start of the iter process
        Cmat = matrix + np.eye(len(matrix))  # matrix of view X = CX - f
        x, prevx = np.zeros(len(matrix)), np.zeros(len(matrix))
        x = Cmat.dot(x) - vec
        cycletrigger = True
        while cycletrigger:
            iters += 1
            delta = []
            prevx = x
            x = Cmat.dot(x) - vec
            for i in range(len(x)):
                delta.append(abs(x[i] - prevx[i]))
            if max(delta) < eps:
                cycletrigger = False
        self.time = time.time() - start_time
        self.iterations_num = iters
        print(time.time(), start_time)
        return np.around(x, int(abs(math.log10(eps))))

    def MatrixZeid(
        self, matrix, vec, eps
    ):  # сначала матрицу надо привести к нормальному виду
        # Метод Зейделя отличается лишь тем, что в цикле вектор Х обновляется сразу с записью в него новой переменной. Он эффективнее,сли реализовывать МПИ построчно. Здесь он будет наоборот медленнее
        start_time, iters = time.time(), 0
        for _ in range(len(matrix)):
            print(matrix)
            for i in range(len(matrix)):
                matcopy = np.array(list(map(abs, np.copy(matrix[i]))))
                print(np.max(matcopy), matcopy[i])
                if np.max(matcopy) == matcopy[i]:
                    # print("matcopy is {}, max is in element {}".format(matcopy, i))
                    pass
                else:
                    print("переставляю")
                    ind = np.where(matcopy == np.max(matcopy))[0]
                    if i not in ind and matrix[i, i] > matrix[ind, i]:
                        matrix[[i, ind[0]]] = matrix[[ind[0], i]]
                        vec[[i, ind[0]]] = vec[[ind[0], i]]
                    break
        for i in range(  # приведение матрицы к сходящейся
            len(matrix)
        ):  # i соответствует номеру строки и т. к. матрица квадратная-номеру столбца для искомого решения
            if matrix[i, i] >= 0:
                matrix[i], vec[i] = matrix[i] * -1, vec[i] * -1
            alpha = 1 / abs(matrix[i, i])
            matrix[i], vec[i] = matrix[i] * alpha, vec[i] * alpha
            Cmat = np.copy(matrix[i])
            Cmat[i] += 1
            if np.linalg.norm(Cmat) > 1:
                prevnorm, k = np.linalg.norm(Cmat), 1.1
                matrix[i], vec[i] = matrix[i] / k, vec[i] / k
                Cmat = np.copy(matrix[i])
                Cmat[i] += 1
                norm = np.linalg.norm(Cmat)
                while norm > 1:
                    prevnorm, k = np.linalg.norm(Cmat), 1.1
                    matrix[i], vec[i] = matrix[i] / k, vec[i] / k
                    Cmat = np.copy(matrix[i])
                    Cmat[i] += 1
                    norm = np.linalg.norm(Cmat)
                    k += 0.1
                    if norm > prevnorm:
                        return "error"

        # start of the iter process
        Cmat = matrix + np.eye(len(matrix))  # matrix of view X = CX - f
        x, prevx = np.zeros(len(matrix)), np.zeros(len(matrix))
        x = Cmat.dot(x) - vec
        cycletrigger = True

        while cycletrigger:
            iters += 1
            delta = []
            prevx = np.copy(x)
            for i in range(len(matrix)):
                x[i] = (Cmat.dot(x) - vec)[i]
            for i in range(len(x)):
                delta.append(abs(x[i] - prevx[i]))
            if max(delta) < eps:
                cycletrigger = False
        self.time = time.time() - start_time
        self.iterations_num = iters
        return np.around(x, int(abs(math.log10(eps))))

    def NewtonMatrix(self, Funcs, vals, var_array, eps):
        start_time, iters = time.time(), 0
        Funcs = sympy.Matrix(
            [[parse_mathematica(fun)] for fun in Funcs]
        ) - sympy.Matrix([parse_mathematica(val) for val in vals])
        var_array = sympy.symbols(var_array)
        eps = float(eps)
        X = np.array(var_array)
        X0 = sympy.Matrix(np.ones_like(X))
        print(X0)
        J = Funcs.jacobian(X)
        norm = np.linalg.norm(
            np.array(J.evalf(subs={X[i]: X0[i] for i in range(len(X))})).astype(float)
        )
        norm_coef = norm / (1 - norm)
        exit_flag = True
        f = sympy.lambdify(var_array, J**-1 * Funcs, "numpy")
        print(J**-1 * Funcs)
        print((J**-1 * Funcs).evalf(subs={X[0]: 1, X[1]: 1}))
        print(f(1, 1))
        while exit_flag:
            iters += 1
            delta = []
            prevX = X0
            print(*list(map(float, X0)), X0)
            print(f(*list(map(float, X0))))
            X0 = X0 - f(*list(map(float, X0)))
            print(X0, "\n")
            for i in range(len(X0)):
                delta.append(abs(X0[i] - prevX[i]))
            if abs(norm_coef * max(delta)) <= eps:
                self.time = time.time() - start_time
                self.iterations_num = iters
                return np.array(X0).astype(np.float64).round(int(abs(math.log10(eps))))

    def MPInonlinear(self, Funcs, vals, var_array, eps):
        start_time, iters = time.time(), 0
        Funcs = sympy.Matrix(
            [[parse_mathematica(fun)] for fun in Funcs]
        ) - sympy.Matrix([parse_mathematica(val) for val in vals])
        var_array = sympy.symbols(var_array)
        eps = float(eps)
        X = np.array(var_array)
        X0 = sympy.Matrix(np.ones_like(X))
        y = sympy.symbols("y:" + str(len(Funcs)))
        Funcs_subser = []
        for i in range(len(Funcs)):
            Funcs_subser.append(
                X[i] + sum([y[j] * Funcs[j] for j in range(len(Funcs))])
            )
            Funcslist = [
                sympy.diff(Funcs_subser[i], var).evalf(
                    subs={X[k]: X0[k] for k in range(len(X))}
                )
                for var in X
            ]
            Funcs_subser[i] = Funcs_subser[i].evalf(subs=sympy.solve(Funcslist))
        Funcs = sympy.Matrix(Funcs_subser)

        f = sympy.lambdify(var_array, Funcs, "numpy")
        delta = [200 * eps]

        try:
            # while norm_coef*max(delta) >= eps:
            while max(delta) >= eps:
                delta.clear()
                prevX = X0
                X0 = f(*map(float, X0))
                for i in range(len(X0)):
                    delta.append(abs(X0[i] - prevX[i]))
                iters += 1
            self.time = time.time() - start_time
            self.iterations_num = iters
            return np.array(X0).astype(np.float64).round(int(abs(math.log10(eps))))
        except OverflowError:
            return "Система не решается методом простой итерации: она только что окончательно разошлась"

    def Square_inter(self, xy_table, coefs, right_eq):
        right_eq = parse_mathematica(right_eq)
        x, y = sympy.symbols("x y")
        X = np.array(sympy.symbols("x:" + str(len(xy_table[0]))))
        Y = np.array(sympy.symbols("y:" + str(len(xy_table[0]))))
        right = np.array([right_eq.subs(x, new_x) for new_x in X])
        Fx = sum(np.power(Y - right, 2))
        Fx = sympy.Matrix(
            [
                Fx.diff(coef).evalf(
                    subs={
                        np.concatenate((X, Y), axis=None)[i]: np.copy(
                            xy_table
                        ).flatten()[i]
                        for i in range(len(np.concatenate((X, Y), axis=None)))
                    }
                )
                for coef in coefs
            ]
        )
        Fx = sympy.solve(Fx)
        return ((Y - right)[0].evalf(subs=Fx).subs([(X[0], x), (Y[0], y)]) - y) * -1

    def Lagrange_inter(self, xy_table):
        xy_table = np.array(xy_table)
        x = sympy.symbols("x")
        y = sympy.symbols("y")
        Y = np.array(sympy.symbols("y:" + str(len(xy_table[1]))))
        X = np.array(sympy.symbols("x:" + str(len(xy_table[0]))))
        polynomes = []
        for i in range(len(Y)):
            uplist = math.prod(
                [x - xy_table[0, j] if j != i else 1 for j in range(len(xy_table[0]))]
            )
            downlist = math.prod(
                [
                    ((xy_table[0, i] - xy_table[0, j]) if j != i else 1)
                    for j in range(len(xy_table[0]))
                ]
            )
            polynomes.append(uplist / downlist)
        L = sum(xy_table[1] * polynomes)
        print('решение методом Лагранжа - {}'.format(sympy.simplify(L)))
        return sympy.simplify(L)

    def Newtons_iter(self, xy_table):
        xy_table = np.array(xy_table)
        x = sympy.symbols("x")
        dict = {xy_table[0, i]: xy_table[1, i] for i in range(len(xy_table[0]))}
        print([[xy_table[0, j] for j in range(i + 1)]for i in range(len(xy_table[0]))])
        f_vec = np.array(
            [
                f([xy_table[0, j] for j in range(i + 1)], dict, x)
                for i in range(len(xy_table[0]))
            ]
        )
        print(f_vec, "\n\n")
        X = np.array(
            [
                math.prod([x - xy_table[0, j] for j in range(i)]) if i != 0 else 1
                for i in range(len(f_vec))
            ]
        )
        print('решение методом Ньютона - {}'.format(sympy.simplify(sum(f_vec * X))))
        return sympy.simplify(sum(f_vec * X))

    def Integrate(
        self, function, der, other_symbols, other_values, interval, eps, method
    ):
        if method == "True":
            intg = sympy.integrate(function, (der, interval[0], interval[1]))
            intg = intg.subs({other_symbols: other_values[0]})
            print(intg, other_symbols, other_values)
            return intg
        print(function)
        try:
            function = sympy.lambdify((der, *other_symbols), function)
        except TypeError:
            function = sympy.lambdify((der, other_symbols), function)
        arr = np.arange(*interval, eps)
        prev_dot = arr[0]
        intg = 0.0

        if method == "squares":
            for dot in arr:
                step = abs(dot - prev_dot)
                intg += step * function(dot, *other_values)
                print(dot, function(dot, *other_values))
                prev_dot = dot
            return intg

        elif method == "trapezes":
            for dot in arr:
                step = dot - prev_dot
                intg += (
                    step
                    * (function(dot, *other_values) + function(prev_dot, *other_values))
                    / 2
                )
                prev_dot = dot
            return intg

        elif method == "Simpson":
            for dot in arr:
                step = dot - prev_dot
                intg += (
                    step
                    * (
                        function(dot, *other_values)
                        + function(prev_dot, *other_values)
                        + 4 * function((dot + prev_dot) / 2, *other_values)
                    )
                    / 6
                )
                prev_dot = dot
            return intg

    def Euler_dif(self, function, x, y, x0, y0, stop_x, h):
        X, Y = [x0], [y0]
        function = sympy.lambdify((x, y), function)
        while X[-1] < stop_x:
            Y.append(Y[-1] + h * function(X[-1], Y[-1]))
            X.append(X[-1] + h)
        return X, Y

    def Runge_4th(self, function, x, y, x0, y0, stop_x, h):
        X, Y = [x0], [y0]
        function = sympy.lambdify((x, y), function)
        while X[-1] < stop_x:
            k1 = function(X[-1], Y[-1])
            k2 = function(X[-1] + h / 2, Y[-1] + h / 2 * k1)
            k3 = function(X[-1] + h / 2, Y[-1] + h / 2 * k2)
            k4 = function(X[-1] + h, Y[-1] + h * k3)
            Y.append(Y[-1] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
            X.append(X[-1] + h)
        return X, Y

    def diff_analytic_solve(self, function, x, f, x0, y0, stop_x, h):
        y = sympy.symbols("y", cls=sympy.Function)
        function = function.subs({f: y(x)})
        C1 = sympy.symbols("C1")
        diff_eq = sympy.Eq(y(x).diff(x), function)
        solved_eq = sympy.dsolve(diff_eq)
        solved_eq = solved_eq.subs(
            {C1: sympy.solve(solved_eq.rhs - y0)[0][C1].subs({x: x0})}
        )
        return sympy.solve(solved_eq, y(x))[0]


def f(it, dict, x):
    if len(it) > 1:
        return (f(it[1:], dict, x) - f(it[:-1], dict, x)) / (it[-1] - it[0])
    else:
        return dict[it[0]]

# №№№№№№№№№№№№№№№№№№№№№№№№№№ ТУТ РЕШЕНИЕ ЗАДАНИЯ 4

# a = NumMethods()
# x = sympy.symbols("x:2")
# F = sympy.Matrix([[sympy.sin(x[0] - x[1]) - x[0] * x[1] + 1], [x[0] ** 2 - x[1] ** 2]])
# X = sympy.Matrix(x)
# Z = sympy.Matrix([0, 0.75])

# X0 = sympy.Matrix([1.15, 0.7])
# result = a.MPInonlinear(F, X, Z, X0, x, 0.000001)
# print("{} - is calculated by FPI method".format(result))

# №№№№№№№№№№№№№№№№№№№№№№№№№№ ТУТ РЕШЕНИЕ ЗАДАНИЯ 4

# №№№№№№№№№№№№№№№№№№№№№№№№№№ ТУТ РЕШЕНИЕ ЗАДАНИЯ 5
# method = NumMethods()
# x, y = sympy.symbols("x, y")
# a = sympy.symbols("a:3")
# coefficients = a
# table = np.array(
#     [
#         [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#         [178, 182, 190, 199, 200, 213, 220, 231, 235, 242],
#     ]
# )
# right_eq_side = a[2] * x**2 + a[1] * x + a[0]

# result = method.Square_inter(table, coefficients, right_eq_side)
# print(result, "solved by squares")

# result2 = method.Lagrange_inter(table)
# result3 = method.Newtons_iter(table)

# print(result2, "Решено Лагранжем")
# print(result3, "Solved by Newton")
# №№№№№№№№№№№№№№№№№№№№№№№№№№ ТУТ РЕШЕНИЕ ЗАДАНИЯ 5

# №№№№№№№№№№№№№№№№№№№№№№№№№№ ТУТ РЕШЕНИЕ ЗАДАНИЯ 6
# method = NumMethods()
# x, N = sympy.symbols("x, N")
# func = sympy.ln(N) / (sympy.sqrt(4 * x**2 + 3 * x + 0.4 * N))
# eps = 0.01
# interval = (1.2, 2.8 + eps)
# N_val = 2

# result1 = method.Integrate(func, x, [N], [N_val], (1.2, 2.8), "squares")
# print(result1)

# result2 = method.Integrate(func, x, [N], [N_val], (1.2, 2.8), "trapezes")
# print(result2)

# result3 = method.Integrate(func, x, [N], [N_val], (1.2, 2.8), "Simpson")
# print(result3)

# print(sympy.integrate(func, (x, 1.2, 2.8)).evalf(subs={N: N_val}))
# №№№№№№№№№№№№№№№№№№№№№№№№№№ ТУТ РЕШЕНИЕ ЗАДАНИЯ 6

# №№№№№№№№№№№№№№№№№№№№№№№№№№ ТУТ РЕШЕНИЕ ЗАДАНИЯ 7
# method = NumMethods()
# x, y = sympy.symbols('x y')
# # function = x*sympy.E**(-1*x**2) - 2*x*y
# function = 6 * x**2 + 5 * x * y
# h = 0.1
# stop = 1
# y0 = 0
# x0 = 0

# result = method.Euler_dif(function, x, y, x0, y0, stop, h)
# print(result[0], '\n', result[1])

# result = method.Runge_4th(function, x, y, x0, y0, stop, h)
# print(result[0], '\n', result[1])

# №№№№№№№№№№№№№№№№№№№№№№№№№№ ТУТ РЕШЕНИЕ ЗАДАНИЯ 7


# F = sympy.Matrix([[sympy.sin(x[0] - x[1]) - x[0] * x[1] + 1], [x[0] ** 2 - x[1] ** 2]])
# X0 = sympy.Matrix([1.15, 0.7])
# result2 = a.NewtonMatrix(F, X, Z, X0, x, 0.000001)
# print("{} - is calculated by Newton's method".format(result2))

# inp = input()
# function = eval("lambda x:" + inp)
# gfunc = eval("lambda x:" + inp + "-x")


# print(a.Half(a=sympy.oo * -1, b=sympy.oo, eps=0.001, func="2**x - 3*x-2"))]
# x = sympy.symbols("x")
# print(
#     a.Newton(
#         0.01,
#         left=sympy.tan(x / 2 - 1.2),
#         right=x**2 - 1,
#         makeaplot=True,
#         start=-10,
#         end=10,
#     )
# )

# M = np.array([[3, -1, -1], [1, 1, 2], [1, 6, -1]], dtype=np.float64)      РАБОТА С ЗАДАНИЕМ 4!
# vex = np.array([2, 3, 0], dtype=np.float64)
# print("given matrix M is\n{}\ngiven vector is {}".format(M, vex))
# start_time = time.time()
# print(a.MatrixMPI(M, vex, 0.0001, 3))
# print('time spent on that is ', time.time()-start_time)

# M = np.array([[1, 5, 1], [2, -1, -1], [1, -2, -1]], dtype=np.float64)
# vex = np.array([-7, 0, 2], dtype=np.float64)
# start_time = time.time()
# print(a.MatrixZeid(M, vex, 0.001, 3))
# print("time spent on that is ", time.time() - start_time)


# a.makeaplot()

# print(a.MPI(0, eps=0.001, func=gfunc))
# print(a.NewtonMethod(0, 0.001, function))
# print(a.NewtonMod1(0, 0.001, function))
