from Matrix import Matrix
import time


a = Matrix.translate("a.txt")

b = Matrix.translate("b.txt")
print(b)

# Matrix.printTranslated(a)
# Matrix.printTranslated(b)

aplusb = Matrix.translate("aplusb.txt")
aplusbcomputed = Matrix.add(a, b)


print("OK: a + b = aplusb" if Matrix.compare(aplusb,
                                             aplusbcomputed) else "NOPE: a + b != aplusb")


aorib = Matrix.translate("aorib.txt")
start = time.time()
aoribcomputed = Matrix.multiply_v2(a, b)
print(time.time() - start)

# print(aoribcomputed)

print("OK: a * b = aorib" if Matrix.compare(aorib,
                                            aoribcomputed) else "NOPE: a * b != aorib")
