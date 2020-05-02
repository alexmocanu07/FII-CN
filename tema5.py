import random
from Matrix import Matrix
class prob:
    n = None
    p = None
    KMAX = 10000
    EPSILON = 10 ** -10

    def __init__(self):
        # prob.n = int(input("n=\n"))
        # self.p = int(input("p=\n"))
        prob.n = 500
        prob.p = 500

        self.generated = self.generate_matrix()
        self.generated = Matrix.erase_null_values(self.generated)
        self.generated = Matrix.sortMatrix(self.generated)
        self.read, self.transpose = Matrix.getMatrixAndTranspose("../resurse_tema5/a_500.txt")

    def solve(self):

        gen_v, gen_lmd = prob.get_values(self, self.generated)
        print(gen_lmd)
        print(gen_v)
        if Matrix.compare(self.read, self.transpose):
            print(prob.power_method(self, self.read))


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
        return v, lmd

    def get_values(self, matrix):
        print("Trying with Kmax = " + str(self.KMAX) + " and epsilon = " + str(self.EPSILON) + "...")
        result = self.power_method(matrix)
        while result == "None":
            self.KMAX += 10000
            self.EPSILON *= 10
            print("Trying with Kmax = " + str(self.KMAX) + " and epsilon = " + str(self.EPSILON) + "...")
            result = self.power_method(matrix)
        return result

if __name__ == '__main__':
    p = prob()
    p.solve()