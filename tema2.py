import numpy as np


class prob:
    def __init__(self, A, b):
        self.A = A
        self.n = len(A)
        self.A2 = self.copy(A)
        self.b = b

        self.epsilon = 10 ** -16
        self.solution = [0 for i in range(self.n)]
        self.L = None
        self.U = None

    def copy(self, matrix):
        A2 = [[0 for i in range(self.n)] for j in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                A2[i][j] = matrix[i][j]
        return A2

    def LU(self):
        for p in range(self.n):
            for i in range(p, self.n):
                S = 0
                for k in range(p):
                    S += self.A[p][k] * self.A[k][i]
                self.A[p][i] -= S
            if abs(self.A[p][p]) < self.epsilon:
                print("nu se poate descompune")
                exit(-1)
            for i in range(p + 1, self.n):
                S = 0
                for k in range(p):
                    S += self.A[i][k] * self.A[k][p]
                if abs(self.A[p][p]) < self.epsilon:
                    print("impartire la 0")
                    exit(-2)
                self.A[i][p] = (self.A[i][p] - S) / self.A[p][p]

        self.get_L()
        self.get_U()

    def get_L(self):
        self.L = [[0 for i in range(self.n)] for j in range(self.n)]
        for i in range(1, self.n):
            for j in range(i):
                self.L[i][j] = self.A[i][j]
        for i in range(self.n):
            self.L[i][i] = 1
        return self.L

    def get_U(self):
        self.U = [[0 for i in range(self.n)] for j in range(self.n)]
        for i in range(self.n):
            for j in range(i, self.n):
                self.U[i][j] = self.A[i][j]
        return self.U

    def detA(self):
        result = 1
        for i in range(self.n):
            result *= self.A[i][i]
        return result

    def get_L_solution(self):
        x = [0 for i in range(self.n)]

        for i in range(self.n):
            # s = 0
            # for j in range(0, i):
            #     s += L[i][j] * x[j]
            # x[i] = b[i] - s
            x[i] = self.b[i] - sum([self.L[i][j] * x[j] for j in range(i)])
        return x

    def get_U_solution(self, y):
        x = [0 for i in range(self.n)]

        for i in reversed(range(self.n)):
            # s = 0
            # for j in range(i + 1, n):
            #     s += U[i][j] * x[j]
            # x[i] = (y[i] - s) / U[i][i]
            x[i] = (y[i] - sum([self.U[i][j] * x[j]
                                for j in range(i + 1, self.n)])) / self.U[i][i]

        return x

    def get_solution(self):
        return self.get_U_solution(self.get_L_solution())

    @staticmethod
    def multiply_lines(a, b):
        sum = 0
        for i in range(len(a)):
            sum += a[i] * b[i]
        return sum

    @staticmethod
    def minus(a, b):
        result = [0 for i in range(len(a))]
        for i in range(len(a)):
            result[i] = a[i] - b[i]
        return result

    def norma1(self):
        left = [0 for i in range(self.n)]
        for i in range(self.n):
            left[i] = prob.multiply_lines(self.A2[i], self.solution)
        return np.linalg.norm(prob.minus(left, self.b))

    def lib_solution(self):
        return np.linalg.solve(self.A2, self.b)

    def lib_inv(self):
        return np.linalg.inv(self.A2)

    def norma2(self):
        return np.linalg.norm(prob.minus(self.solution, self.lib_solution()))

    def norma3(self):
        right = [0 for i in range(self.n)]
        for i in range(self.n):
            right[i] = prob.multiply_lines(self.lib_inv()[i], self.b)
        return np.linalg.norm(prob.minus(self.solution, right))

    def results(self):
        self.LU()
        print("Determinantul lui A este: " + str(self.detA()) + "\n")

        self.solution = self.get_solution()
        print("Solutia sistemului este: " + str(self.solution) + "\n")

        norma1 = self.norma1()
        print("Norma Ainit * Xlu - b este: " + str(self.norma1()) + "\n")

        norma2 = self.norma2()
        print("Norma Xlu - Xlib este: " + str(norma2) + "\n")

        norma3 = self.norma3()
        print("Norma Xlu - Ainv * b este:" + str(norma3) + "\n")


if __name__ == "__main__":
    system_matrix = [
        [1, 1, 1],
        [5, 3, 2],
        [0, 1, -1]]
    free_array = [25, 0, 6]
    pr = prob(system_matrix, free_array)
    pr.results()
