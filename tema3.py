from Matrix import Matrix


a = Matrix.translate("a.txt")
b = Matrix.translate("b.txt")
# Matrix.printTranslated(a)
# Matrix.printTranslated(b)

aplusb = Matrix.translate("aplusb.txt")
aplusbcomputed = Matrix.add(a, b)


print("OK: a + b = aplusb" if Matrix.compare(aplusb,
                                             aplusbcomputed) else "NOPE: a + b != aplusb")


aorib = Matrix.translate("aorib.txt")
aoribcomputed = Matrix.multiply(a, b)

print("OK: a * b = aorib" if Matrix.compare(aorib,
                                            aoribcomputed) else "NOPE: a * b != aorib")
