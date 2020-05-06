import random
from Matrix import Matrix
import numpy as np
class prob:
    n = None
    p = None
    KMAX = 100000
    EPSILON = 10 ** -10

    def __init__(self):
        prob.n = int(input("n=\n"))
        prob.p = int(input("p=\n"))

        self.generated = self.generate_matrix()
        self.generated = Matrix.erase_null_values(self.generated)
        self.generated = Matrix.sortMatrix(self.generated)
        self.read, self.transpose = Matrix.getMatrixAndTranspose("../resurse_tema5/a_500.txt")


    def solve(self):
        if prob.p == prob.n :
            if prob.p < 500:
                print("p si n trebuie sa fie mai mari ca 500. Programul se inchide...")
                exit(0)
            gen_v, gen_lmd = prob.get_values(self, self.generated)
            print("Valoarea proprie de modul maxim a matricei generate este: " + str(gen_lmd))
            print("Vectorul propriu asociat matricei generate este: " + str(gen_v))
            if Matrix.compare(self.read, self.transpose):
                read_v, read_lmd = prob.get_values(self, self.read)
                print("Valoarea proprie de modul maxim a matricei din fisier este: " + str(read_lmd))
                print("Vectorul propriu asociat matricei din fisier este: " + str(read_v))
            else:
                print("Matricea din fisier nu este simetrica.")
        else:
            matrix = prob.generate_classic_matrix(prob.p, prob.n)
            free = [random.uniform(-10, 10) for _ in range(prob.p)]
            singular_values = prob.get_singular_values(matrix)[1].tolist()
            print("Valorile singulare sunt: " + str(singular_values))
            print("Rangul matricei este: " + str(prob.get_matrix_rank(singular_values)))
            print("Rangul matricei calculat cu numpy este: " + str(np.linalg.matrix_rank(matrix)))
            print("Numarul de conditionare al matricei este: " + str(prob.get_matrix_condition(singular_values)))
            print("Numarul de conditionare calculat cu numpy este: " + str(np.linalg.cond(matrix)))
            pseudoInverse = prob.get_matrix_MoorePenrose(matrix)
            # print("Pseudoinversa matricei este:\n" + str(pseudoInverse))
            # print(np.linalg.pinv(matrix))
            xI = prob.multiply_classic_matrix_vector(pseudoInverse, free)
            # xI2 = prob.compute_xI(matrix, free)
            # print("xI2 = " + str(xI2))
            print("Solutia sistemului Ax = b este: " + str(xI))
            Ax = prob.multiply_classic_matrix_vector(matrix, xI)
            # print("A * Xi = " + str(Ax))
            # print("free = " + str(free))
            print("Norma intre pseudoinversa generata si cea din biblioteca: " + str(prob.get_norm_for_matrices(pseudoInverse, np.linalg.pinv(matrix).tolist())))
            print("Norma b - Ax este: " + str(prob.euclidean_norm(free, Ax)))

            inverse2 = prob.compute_smallest_square_inverse(matrix)
            print("Norma intre inverse: " + str(prob.get_norm_for_matrices(pseudoInverse, inverse2)))


    def generate_vector_of_norm1(self):
        random.seed()
        x = list()
        v = [0 for _ in range(self.n)]
        for _ in range(self.n):
            number = random.randint(1, 100)
            x.append(number)
        x_norm = sum([x[i]**2 for i in range(len(x))])**(1/2)
        for i in range(self.n):
            v[i] = (1.0 / x_norm) * x[i]

        return v


    def generate_matrix(self):
        A = dict()
        random.seed()
        for i in range(self.n):
            loop = True
            while loop:
                lin = random.randint(0, self.n - 1)
                col = random.randint(0, self.n - 1)
                nr = round(random.uniform(0, 100), 2)
                elt = A.get(lin, "none")
                if elt == "none":
                    A[lin] = [[nr, col]]
                    A[col] = [[nr, lin]]
                    loop = False
                else:
                    index = [i for i in range(len(elt)) if elt[i][1] == col]
                    if not index:
                        if lin != col:
                            A[lin].append([nr, col])
                            elt = A.get(col, "none")
                            if elt == "none":
                                A[col] = [[nr, lin]]
                            else:
                                A[col].append([nr, lin])
                        else:
                            A[lin].append([nr, col])
                        loop = False

        return A

    @staticmethod
    def generate_classic_matrix(linCount, colCount):
        return [[round(random.uniform(-10, 10), 2) for _ in range(colCount)] for _ in range(linCount)]

    @staticmethod
    def multiply_matrix_vector(matrix, vector):
        result = [0 for _ in range(prob.n)]
        for i in range(prob.n):
            if i in matrix.keys():
                sum = 0
                for poz in range(len(matrix[i])):
                    col = matrix[i][poz][1]

                    sum += matrix[i][poz][0] * vector[col]

                result[i] = sum
            else:
                result[i] = 0
        return result

    @staticmethod
    def scalar_product(v1, v2):
        return sum([v1[i] * v2[i] for i in range(len(v1))])

    @staticmethod
    def compute_v(w):
        v = list()
        w_norm = sum([w[i]**2 for i in range(len(w))])**(1/2)
        for i in range(len(w)):
            v.append((1.0 / w_norm) * w[i])
        return v

    @staticmethod
    def scale_vector(vector, scalar):
        return [(vector[i] * scalar) for i in range(len(vector))]

    @staticmethod
    def euclidean_norm(v1, v2):
        diff = [v1[i] - v2[i] for i in range(len(v1))]
        return sum([diff[i]**2 for i in range(len(diff))])**(1/2)

    def power_method(self, matrix):
        v = prob.generate_vector_of_norm1(self)
        w = prob.multiply_matrix_vector(matrix, v)
        lmd = prob.scalar_product(w, v)
        k = 0
        looping = True
        while looping:
            looping = False
            v = prob.compute_v(w)
            w = prob.multiply_matrix_vector(matrix, v)
            lmd = prob.scalar_product(w, v)
            k += 1
            if k > prob.KMAX:
                looping = False
                return "None"
            norm = prob.euclidean_norm(w, prob.scale_vector(v, lmd))
            neps = prob.n * prob.EPSILON
            if norm > neps:
                looping = True
        print("Found values after " + str(k) + " iterations.")
        return v, lmd

    def get_values(self, matrix):
        print("Trying to get values with Kmax = " + str(self.KMAX) + " and epsilon = " + str(self.EPSILON) + "...")
        tries = 1
        result = self.power_method(matrix)
        while result == "None" and tries < 5:
            self.KMAX += 100000
            self.EPSILON *= 10
            tries +=1
            print("Trying to get values with Kmax = " + str(self.KMAX) + " and epsilon = " + str(self.EPSILON) + "...")
            result = self.power_method(matrix)
        if tries == 5:
            print("Could not get values. Exiting...")
            exit(0)
        return result

    @staticmethod
    def get_singular_values(matrix):
        return np.linalg.svd(matrix)

    @staticmethod
    def get_matrix_rank(singular_values):
        rank = 0
        for value in singular_values:
            if value > 0:
                rank += 1
        return rank


    @staticmethod
    def get_matrix_condition(singular_values):
        return max(singular_values) / min(singular_values)

    @staticmethod
    def compute_sI(singular_values, p, n):
        sI=[[0 for _ in range(p)] for _ in range(n)]
        positive_values = [value for value in singular_values if value > 0]
        for i in range(len(positive_values)):
            sI[i][i] = 1.0 / positive_values[i]
        return sI

    @staticmethod
    def multiply_matrixes(a, b):
        if len(a[0]) != len(b):
            print("Nu se poate face inmultirea intre matrici")
            exit(-1)
        result = [[0 for _ in range(len(b[0]))] for _ in range(len(a))]
        for i in range(len(a)):
            for j in range(len(b[0])):
                sum = 0
                for k in range(len(b)):
                    sum += a[i][k] * b[k][j]
                result[i][j] = sum

        return result

    @staticmethod
    def get_matrix_MoorePenrose(matrix):
        u, singular_values, v = prob.get_singular_values(matrix)
        u = u.tolist()
        singular_values = singular_values.tolist()
        v = v.tolist()
        sI=prob.compute_sI(singular_values, len(matrix), len(matrix[0]))
        uT = prob.get_matrix_transpose(u)
        v = prob.get_matrix_transpose(v)
        return prob.multiply_matrixes(prob.multiply_matrixes(v, sI), uT)

    @staticmethod
    def multiply_classic_matrix_vector(matrix, vector):
        result = [0 for _ in range(len(matrix))]
        for i in range(len(matrix)):
            sum = 0
            for j in range(len(vector)):
                sum += vector[j] * matrix[i][j]
            result[i] = sum
        return result

    @staticmethod
    def get_matrix_transpose(matrix):
        result = [[0 for _ in range(len(matrix))] for _ in range(len(matrix[0]))]
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                result[j][i] = matrix[i][j]
        return result

    @staticmethod
    def compute_smallest_square_inverse(matrix):
        transpose = prob.get_matrix_transpose(matrix)
        return prob.multiply_matrixes(np.linalg.inv(prob.multiply_matrixes(transpose, matrix)).tolist(), transpose)

    @staticmethod
    def get_norm_for_matrices(matrix1, matrix2):
        dif = [[matrix1[i][j] - matrix2[i][j] for j in range(len(matrix1[0]))] for i in range(len(matrix1))]
        for i in range(len(matrix1)):
            for j in range(len(matrix1[0])):
                dif[i][j] = matrix1[i][j] - matrix2[i][j]
        max = dif[0][0]
        for i in range(len(dif)):
            for j in range(len(dif[0])):
                if dif[i][j] > max:
                    max = dif[i][j]

        return max

    @staticmethod
    def compute_xI(matrix, free):
        u, s, vT = np.linalg.svd(matrix)
        v = prob.get_matrix_transpose(vT.tolist())
        uT = prob.get_matrix_transpose(u.tolist())
        sI = prob.compute_sI(s.tolist(), len(matrix), len(matrix[0]))
        vsI = prob.multiply_matrixes(v, sI)
        vsIuT = prob.multiply_matrixes(vsI, uT)
        xI = prob.multiply_classic_matrix_vector(vsIuT, free)
        return xI

if __name__ == '__main__':
    p = prob()
    p.solve()


