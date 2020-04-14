from Matrix import Matrix


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


# def GaussSeidel(matrix, free):
#     xc = xp = [0] * len(free)
#     k = 0
#     while (True):
#         xp = xc
#         # compute xc with the formula
#         # compute delta x
#         k += 1
#         if (delta >= Matrix.EPSILON and k <= Matrix and delta <= 10**8):
#             break


##################

a = Matrix.translate("practice/a.txt")
b = [6.0, 7.0, 8.0, 9.0, 1.0]
x = [1.0, 2.0, 3.0, 4.0, 5.0]

print(_GaussSeidelLineGenerator(x, a, b))
