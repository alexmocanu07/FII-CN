from Matrix import Matrix


def _Norm(x):
    return sum([x[i]**2 for i in range(0, len(x))])**(1/2)

def computeXc(xp, a, b):
    xc = xp.copy()
    for i in range(len(a.keys())):
        top = b[i]
        poz = 0
        col = a[i][poz][1]
        while col < i:
            top -= a[i][poz][0] * xc[col]
            poz += 1
            col = a[i][poz][1]
        if col != i:
            print("0 pe diagonala")
            exit(-2)
        bottom = a[i][poz][0]
        poz += 1
        while poz < len(a[i]):
            col = a[i][poz][1]
            top -= a[i][poz][0] * xp[col]
            poz += 1
        xc[i] = top / bottom

    return xc


def GaussSeidel(matrix, free):
    xc = [0 for _ in range(len(free))]
    xp = [0 for _ in range(len(free))]
    # xc =[1,2,3,4,5]
    # xp =[1,2,3,4,5]
    k = 0
    delta = None
    looping = True
    while looping:
        xp = xc.copy()
        xc = computeXc(xp, matrix, free)
        # print(k, xc)
        looping = False
        for i in range(len(xp)):
            dif = abs(xc[i] - xp[i])
            if dif > Matrix.EPSILON and dif < 10 ** 10:
                looping = True
                break
        k += 1
        if k > Matrix.KMAX:
            break
    if not looping:
        return xc
    else:
        return "DIVERGENTA"


def infinityNorm(v1, v2):
    max = abs(v1[0] - v2[0])
    for i in range(1, len(v1)):
        if max < abs(v1[i] - v2[i]):
            max = abs(v1[i] - v2[i])
    return max


def computeAxGS(A, x):
    result = list()
    for i in range(len(A)):
        sum = 0
        for poz in range(len(A[i])):
            col = A[i][poz][1]
            sum += A[i][poz][0] * x[col]
        result.append(sum)
    return result

##################

a = Matrix.getMatrix("a_1.txt", "matrix")
b = Matrix.getMatrix("b_1.txt", "vector")
# b = [6,7,8,9,1]
result = GaussSeidel(a, b)
if result == "DIVERGENTA":
    print("Divergenta")
else:
    print("Rezultatul sistemului: " + str(result))
    print("Norma: " + str(infinityNorm(computeAxGS(a, result), b)))

