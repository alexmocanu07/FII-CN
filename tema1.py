import random
import math
def prob1():
    m = 0
    u = 1
    while 1 + u != 1:
        m +=1
        u /=10

    return [u,m]

def prob2():
    random.seed()
    x = random.random()
    y = random.random()
    z = random.random()
    while (x * y) * z == x * (y * z):
        x = random.random()
        y = random.random()
        z = random.random()
    return [x,y,z]

# eps = prob1()

def plus(a,b):
    if a == 0 and b == 0: return 0
    return 1


def times(a,b):
    return a*b


def bordare_linii(matrix, n):
    local_n = n
    while math.log(local_n, 2) != int(math.log(local_n, 2)):
        matrix.append([0 for j in range(0,n)])
        local_n += 1
    new_logn = math.log(local_n, 2)
    while local_n % new_logn != 0:
        matrix.append([0 for j in range(0,n)])
        local_n +=1


def bordare_coloane(matrix, n):
    local_n = n
    while math.log(local_n, 2) != int(math.log(local_n, 2)):
        for i in range(0,n):
            matrix[i].append(0)
        local_n += 1
    new_logn = math.log(local_n,2)
    while local_n % new_logn != 0:
        for i in range(0,n):
            matrix[i].append(0)
        local_n += 1
    return local_n


def split_by_columns(A, n, logn):
    output = list()
    for i in range(0, logn):
        A_i = list()
        for j in range(0,n):
            A_i.append(A[j][i*logn:(i+1)*logn])
        output.append(A_i)
    return output


def split_by_rows(B, n, logn):
    output = list()
    for i in range(0, logn):
        output.append(B[i*logn:(i+1)*logn])
    return output


def NUM(v):
    v = v[::-1]
    v_string = ""
    for i in range(0,len(v)):
        v_string += str(v[i])
    return int(v_string, 2)


def get_k(j):
    k = 0
    while 2**k < j: k +=1
    if 2**k > j: k -=1
    return k


def lines_sum(a,b):
    output = list()
    for i in range(0, len(a)):
        output.append(plus(a[i], b[i]))
    return output


def matrix_sum(a,b):
    output = list()
    for i in range(0,len(a)):
        line = list()
        for j in range(0,len(a)):
            line.append(plus(a[i][j], b[i][j]))
        output.append(line)
    return output


def get_C(B_split, A_split, n, p, m):
    C = [[0]* n] * n
    for i in range(0,p):
        sum_linii_B = list()
        sum_linii_B.append([0 for i in range(0,n)])
        for j in range(1, 2**m):
            k = get_k(j)

            sum_linii_B.append(lines_sum(sum_linii_B[j-2**k], B_split[i][k]))

        C = matrix_sum(C, [sum_linii_B[NUM(A_split[i][r])] for r in range(0, n)])

    return C


def multiply_lines(a, b, n):
    sum = 0
    for i in range(0,n):
        sum = plus(sum, times(a[i], b[i]))
    return sum


def classic_matrix_multiply(A, B, n):
    C = list()
    for i in range(0,n):
        C.append([0 for j in range(0,n)])
    for i in range(0,n):
        for j in range(0, n):
            a = A[i]
            b = [B[k][j] for k in range(0,n)]
            C[i][j] = multiply_lines(a, b, n)
    return C


def prob3(A, B, n):
    bordare_linii(B, n)
    new_n = bordare_coloane(A, n)

    logn = int(math.log(new_n, 2))
    A_split = split_by_columns(A, n, logn)
    B_split = split_by_rows(B, n, logn)
    m = int(math.log(n, 2))
    if m != math.log(n, 2): m += 1

    return get_C(B_split, A_split, n, logn, m)


A = [[1, 0, 0, 1, 1, 0, 1],
     [0, 0, 1, 0, 0, 1, 0],
     [1, 0, 0, 0, 1, 0, 0],
     [0, 0, 0, 1, 0, 1, 0],
     [0, 1, 0, 1, 1, 0, 1],
     [1, 1, 1, 0, 0, 1, 0],
     [0, 1, 0, 0, 1, 0, 0]]

B = [[0, 0, 0, 1, 0, 0, 1],
     [1, 0, 1, 1, 0, 0, 0],
     [0, 1, 0, 0, 1, 0, 1],
     [0, 1, 1, 1, 1, 0, 0],
     [0, 1, 1, 0, 0, 0, 1],
     [0, 1, 0, 1, 1, 0, 0],
     [1, 1, 0, 1, 0, 1, 1]]


A2 = A
B2 = B

print("rezultatul la problema 1: " + str(prob1()))
result = prob2()
x = result[0]
y = result[1]
z = result[2]
print("x = " + str(result[0]) + " y = " + str(result[1]) + " z = " + str(result[2]))
print("(x * y) * z = " + str((x*y)*z))
print("x * (y * z) = " + str(x * (y*z)))

print("Matricea calculata cu metoda celor 4 rusi: \n" + str(prob3(A, B, len(A))))
print("Matricea calculata cu metoda clasica: \n" + str(classic_matrix_multiply(A2, B2, len(A2))))









