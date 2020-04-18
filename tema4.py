from Matrix import Matrix


def _Norm(x):
    return sum([x[i]**2 for i in range(0, len(x))])**(1/2)


def _GaussSeidelLineGenerator(xp, a, b):
    # just to make sure the indexes are available for overriding values after computation
    xc = [0 for _ in range(len(xp))]

    for i in range(0, len(xp)):
        aLine = a[i]
        # the first sum if made up of value computed in the new vector, before the current index
        s1 = 0
        for j in range(0, i-1 + 1):
            # find the value in the imaginary a[i][j]
            indexes = [index for index in range(
                0, len(aLine)) if aLine[index][1] == j]
            aValue = 0 if indexes is None or len(
                indexes) != 1 else aLine[indexes[0]][0]
            if(aValue != 0):
                s1 += xc[j] * aValue
        # the second sum is built from values computed in the old vector
        s2 = 0
        for j in range(i+1, len(xp)):
            # find the value in the imaginary a[i][j]
            indexes = [index for index in range(
                0, len(aLine)) if aLine[index][1] == j]
            aValue = 0 if indexes is None or len(
                indexes) != 1 else aLine[indexes[0]][0]
            if(aValue != 0):
                s2 += xp[j] * aValue
        # compute the top section of the formula
        top = b[i] - s1 - s2

        indexes = [index for index in range(
            0, len(aLine)) if aLine[index][1] == i]
        bottom = 0 if indexes is None or len(
            indexes) != 1 else aLine[indexes[0]][0]
        if(bottom == 0):
            raise Exception("0 value on the diagonal of our matrix :( oops.")
        xc[i] = top / bottom
    return xc

def computeXc(xp, a, b):
    # xc = [0 for _ in range(len(xp))]
    xc = xp.copy()
    for i in range(len(a.keys())):
        result = b[i]
        j = a[i][0][1]
        index = 0
        while j < i:
            result -= a[i][index][0] * xc[j]
            index += 1
            j = a[i][index][1]
        if j != i:
            print("Am 0 pe diagonala, ia gata")
            exit(-2)
        middle = a[i][index][0]  # {1 : [[25, 1], [2, 5], [3, 7]]
        index += 1
        if index == len(a[i]):
            xc[i] = result / middle
            continue
        j = a[i][index][1]
        while index < len(a[i]):
            result -= a[i][index][0] * xp[j]
            index += 1

        xc[i] = result / middle
    return xc

def computeXc_v2(xp, a, b):
    xc = xp.copy()
    for i in range(len(a.keys())):
        top = b[i]
        poz = 0
        col = a[i][poz][1]
        while col < i:
            top -= a[i][poz][0] * xc[col]
            poz += 1
            col = a[i][poz][1]
        # col >= i
        if col != i:
            print("ia gata")
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
        xc = computeXc_v2(xp, matrix, free)
        # print(k, xc)
        looping = False
        for i in range(len(xp)):
            dif = abs(xc[i] - xp[i])
            if dif > Matrix.EPSILON:
                looping = True
                break
        k += 1
        if k > Matrix.KMAX:
            break
    if not looping:
        return xc
    else :
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

a = Matrix.getMatrix("a_3.txt", "matrix")
b = Matrix.getMatrix("b_3.txt", "vector")
# b = [6,7,8,9,1]
result = GaussSeidel(a, b)
print("Rezultatul sistemului: " + str(result))
print("Norma: " + str(infinityNorm(computeAxGS(a, result), b)))

