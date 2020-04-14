from Matrix import Matrix
import time


res = Matrix.getMatrixAndTranspose("a.txt")
a = res[0]
aT = res[1]
res = Matrix.getMatrixAndTranspose("b.txt")
b = res[0]
bT = res[1]


aplusb = Matrix.getMatrix("aplusb.txt")
aplusbcomputed = Matrix.add(a, b)


print("OK: a + b = aplusb" if Matrix.compare(aplusb,
                                             aplusbcomputed) else "NOPE: a + b != aplusb")


aorib = Matrix.getMatrix("aorib.txt")
start = time.time()
aoribcomputed = Matrix.multiply(a, bT)
print("Total time: " + str(time.time() - start))


print("OK: a * b = aorib" if Matrix.compare(aorib,
                                            aoribcomputed) else "NOPE: a * b != aorib")
