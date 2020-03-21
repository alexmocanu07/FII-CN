import math

EPSILON = 0.000001


def printTranslated(matrix):
    for line in range(0, int(sorted(matrix.keys())[-1]) + 1):
        print(matrix[line] if line in matrix else [])
    print()


def readFromFile(filename):
    f = open("resources/"+filename, "r")
    lines = []
    for x in f:
        lines.append(x.rstrip())
    return lines


def translate(filename):
    lines = readFromFile(filename)
    end = True
    store = {}
    size = None
    for line in lines:
        # read the size (first line)
        if(size == None):
            size = int(math.sqrt(int(line)))
            continue
        # read the data
        data = line.split(", ")
        # last line is probably a blank line, skip it
        if(len(data) != 3):
            continue
        # line structure -> [value, line, column]
        # convert to -> store[line] = [{value, column},{...},...]
        dataValue = float(data[0])
        dataLine = int(data[1])
        dataColumn = int(data[2])
        # if the element is already declared, find index and sum up the values
        if dataLine not in store:
            store[dataLine] = []

        try:
            index = store[dataLine].index([dataValue, dataColumn])
            store[dataLine][index][0] += dataValue
            print("joker", data)
        except:
            # new element, push it to the "line"
            store[dataLine].append([dataValue, dataColumn])

    # to save space, empty line will not be stored
    # to recreate the matrix, we get the max line (last key in sorted dictionary) and use it in range
    # printTranslated(store)
    return store


def compare(a, b):
    # line count is different
    if len(a.keys()) != len(b.keys()):
        return False

    for lineKey in a:
        # line from a is missing from b
        if(lineKey not in b):
            return False
        # cell count on line is different
        if len(a[lineKey]) != len(b[lineKey]):
            return False

        aLine = a[lineKey]
        bLine = b[lineKey]

        for aCell in aLine:
            # find the bCell (maybe order is different, but cell still exists)
            indexes = [index for index in range(
                0, len(bLine)) if bLine[index][1] == aCell[1]]
            # if bCell is missing (no index found), a != b
            if indexes is None or len(indexes) != 1:
                return False
            else:
                # if bCell is there, but its value is EPSILON-different than the one in aCell, a != b
                if abs(bLine[indexes[0]][0] - aCell[0]) >= EPSILON:
                    return False
    return True


def add(a, b):
    result = b.copy()
    # for each line in "a"
    for lineKey in a:
        # if "result" doesn't have that line (missing key "i"), copy it over there
        if(lineKey not in result):
            result[lineKey] = a[lineKey]
            continue
        aLine = a[lineKey]
        bLine = result[lineKey]
        for aCell in aLine:
            indexes = [index for index in range(
                0, len(bLine)) if bLine[index][1] == aCell[1]]
            # if the element is already in the "result" matrix, do sum (indexes will have max one element)
            if(indexes is not None and len(indexes) > 0):
                index = indexes[0]
                result[lineKey][index][0] += aCell[0]
            # else, push the new element into the "result" matrix
            else:
                result[lineKey].append(aCell)
    # printTranslated(result)
    return result


# a = translate("practice/a.txt")
# b = translate("practice/b.txt")

a = translate("a.txt")
b = translate("b.txt")


aplusb = translate("aplusb.txt")
aplusbcomputed = add(a, b)


print("OK: a + b = aplusb" if compare(aplusb,
                                      aplusbcomputed) else "NOPE: a + b != aplusb")
