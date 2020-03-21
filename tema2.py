import numpy as np
import copy



class Bonus:
    def __init__(self, A, b):
        self.A = copy.deepcopy(A)
        self.b = copy.deepcopy(b)
        self.n = len(A)
        self.epsilon = 10 ** -16

        self.L = list()
        for i in range(self.sumN(self.n)):
            self.L.append(0)

        self.U = list()
        for i in range(self.sumN(self.n)):
            self.U.append(0)

        self.solution = None

        self.LU()

    @staticmethod
    def sumN(x):
        val = int((x * (x + 1)) / 2)
        return val

    def l_index(self, lin, col):
        return self.sumN(lin) + col

    def u_index(self, lin, col):
        return self.sumN(self.n) - self.sumN(self.n - lin) + col - lin
        # [(0,0), (0,1), (0,2) , (1,1), (1,2), (2,2)] -> U
        #  1,2 -> 4  = 6 - 3 + 2 - 1
        #  0,2 -> 2 = 6 - 6 + 2

    def geta(self): return self.A


    def LU(self):

        for p in range(0, self.n):
            self.L[self.l_index(p, p)] = 1
            for i in range(p, self.n):
                u = copy.deepcopy(self.A[p][i])
                for k in range(0, p):
                    u = u - self.L[self.l_index(p, k)] * self.U[self.u_index(k, i)]
                self.U[self.u_index(p, i)] = copy.deepcopy(u)

            for i in range(p + 1, self.n):
                l = copy.deepcopy(self.A[i][p])
                for k in range(0, p):
                    l = l - self.L[self.l_index(i, k)] * self.U[self.u_index(k, p)]
                if abs(self.U[self.u_index(p, p)]) > self.epsilon:
                    l = l / self.U[self.u_index(p, p)]
                else:
                    print("impartire la 0")
                    exit(-3)
                self.L[self.l_index(i, p)] = copy.deepcopy(l)

    def solve(self):
        y = [0 for i in range(self.n)]
        x = [0 for i in range(self.n)]
        for i in range(self.n):
            y[i] = self.b[i]
            for j in range(i):
                y[i] -= self.L[self.l_index(i, j)] * y[j]

        for i in reversed(range(0, self.n)):
            x[i] = y[i]
            for j in range(i + 1, self.n):
                x[i] -= self.U[self.u_index(i, j)] * x[j]
            if abs(self.U[self.u_index(i, i)]) > self.epsilon:
                x[i] /= self.U[self.u_index(i, i)]
            else:
                print("impartire la 0")
                exit(-4)

        self.solution = x

    def get_solution(self):
        self.solve()
        return self.solution

    def print_l(self):
        for i in range(self.n):
            for j in range(i + 1):
                print(self.L[self.l_index(i, j)], end=" ")
            print()

    def print_u(self):
        for i in range(self.n):
            for j in range(i, self.n):
                print(self.U[self.u_index(i, j)], end=" ")
            print()





class Prob:
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
            x[i] = self.b[i] - sum([self.L[i][j] * x[j] for j in range(i)])
        return x

    def get_U_solution(self, y):
        x = [0 for i in range(self.n)]

        for i in reversed(range(self.n)):
            x[i] = (y[i] - sum([self.U[i][j] * x[j] for j in range(i + 1, self.n)])) / self.U[i][i]

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
            left[i] = Prob.multiply_lines(self.A2[i], self.solution)
        return np.linalg.norm(Prob.minus(left, self.b))

    def lib_solution(self):
        return np.linalg.solve(self.A2, self.b)

    def lib_inv(self):
        return np.linalg.inv(self.A2)

    def norma2(self):
        return np.linalg.norm(Prob.minus(self.solution, self.lib_solution()))

    def norma3(self):
        right = [0 for i in range(self.n)]
        for i in range(self.n):
            right[i] = Prob.multiply_lines(self.lib_inv()[i], self.b)
        return np.linalg.norm(Prob.minus(self.solution, right))

    def matrix_minus(self, A, B):
        C = [[0 for i in range(self.n)] for j in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                C[i][j] = A[i][j] - B[i][j]

        return C

    def inversa(self):
        Ainv = [[0 for i in range(self.n)] for j in range(self.n)]

        for k in range(self.n):
            e = [0 for i in range(self.n)]
            y = [0 for i in range(self.n)]
            x = [0 for i in range(self.n)]
            e[k] = 1

            for i in range(self.n):
                y[i] = e[i]
                y[i] -= sum([self.A[i][j] * y[j] for j in range(i)])

            for i in reversed(range(self.n)):
                x[i] = y[i]
                x[i] -= sum([self.A[i][j] * x[j] for j in range(i + 1, self.n)])
                x[i] /= self.A[i][i]

            for i in range(self.n):
                Ainv[i][k] = x[i]
        return Ainv

    def norma4(self):
        Ainv = self.inversa()
        newmatrix = self.matrix_minus(Ainv, self.lib_inv())
        max = 0
        for j in range(self.n):
            max += abs(newmatrix[0][j])
        for i in range(1, self.n):
            current = 0
            for j in range(self.n):
                current += abs(newmatrix[i][j])
            if current > max: max = current

        return max

    def results(self):
        self.LU()
        print("Determinantul lui A este: " + str(self.detA()) + "\n")

        self.solution = self.get_solution()
        print("Solutia sistemului este: " + str(self.solution) + "\n")

        print("Solutia sistemului calculata cu numpy este: " + str(self.lib_solution()) + "\n")

        norma1 = self.norma1()
        print("Norma Ainit * Xlu - b este: " + str(norma1) + "\n")

        norma2 = self.norma2()
        print("Norma Xlu - Xlib este: " + str(norma2) + "\n")

        norma3 = self.norma3()
        print("Norma Xlu - Ainv * b este: " + str(norma3) + "\n")

        Ainv = self.inversa()
        print("Inversa lui A este: " + str(Ainv) + "\n")

        norma4 = self.norma4()
        print("Norma AinvLU - AinvLib este: " + str(norma4) + '\n')


if __name__ == "__main__":
    system_matrix = [
        [2.5, 2, 2],
        [5, 6, 5],
        [5, 6, 6.5]]
    free_array = [2, 2, 2]

    system_matrix2 = [
        [1, 1, 1],
        [5, 3, 2],
        [0, 1, -1]
    ]
    free_array2 = [25, 0, 6]

    pr = Prob(copy.deepcopy(system_matrix), copy.deepcopy(free_array))
    pr.results()

    bonus = Bonus(system_matrix, free_array)

    print("L de la bonus este: ", end="")
    print(str(bonus.L))
    bonus.print_l()
    print("U de la bonus este: ", end="")
    print(str(bonus.U))
    bonus.print_u()
    print("Solutia sistemului de la bonus este: ", end = "")
    print(bonus.get_solution())




