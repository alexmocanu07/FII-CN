from Matrix import Matrix


def _Norm(x):
    return sum([x[i]**2 for i in range(0, len(x))])**(1/2)


def _GaussSeidelLineGenerator(xp, a, b):
    # just to make sure the indexes are available for overriding values after computation
    xc = [0] * len(xp)

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


def GaussSeidel(matrix, free):
    xc = xp = [0] * len(free)
    k = 0
    delta = None
    while (True):
        xp = xc
        xc = _GaussSeidelLineGenerator(xp, matrix, free)
        print(k, xc)
        delta = _Norm([xc[i] - xp[i] for i in range(0, len(xc))])
        k += 1
        if (delta >= Matrix.EPSILON and k <= Matrix.KMAX and delta <= 10**8):
            break
    if(delta != None and delta < Matrix.EPSILON):
        return xc
    print("Divergență")
    return None


##################

a = Matrix.translate("a_1.txt", "matrix")
b = Matrix.translate("b_1.txt", "vector")

print(GaussSeidel(a, b))
