from Matrix import Matrix
import time


res = Matrix.getMatrixAndTranspose("a.txt")
a = res[0]
aT = res[1]
# print(a[0])
res = Matrix.getMatrixAndTranspose("b.txt")
b = res[0]
bT = res[1]
# print(b[0])

# print(Matrix.transpose_v2("b.txt")[0])

# Matrix.printTranslated(a)
# Matrix.printTranslated(b)

aplusb = Matrix.translate("aplusb.txt")
aplusbcomputed = Matrix.add(a, b)


print("OK: a + b = aplusb" if Matrix.compare(aplusb,
                                             aplusbcomputed) else "NOPE: a + b != aplusb")


aorib = Matrix.translate("aorib.txt")
start = time.time()
aoribcomputed = Matrix.multiply_v2(a, bT)
print(time.time() - start)

# print(aoribcomputed)

print("OK: a * b = aorib" if Matrix.compare(aorib,
                                            aoribcomputed) else "NOPE: a * b != aorib")
