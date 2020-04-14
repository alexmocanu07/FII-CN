import math


class Matrix:
    KMAX = 10
    EPSILON = 10 ** -16

    ##### OPERATIONS ######

    @staticmethod
    def transpose(a):
        result = {}
        for lineKey in a:
            aLine = a[lineKey]
            for aCell in aLine:
                # insure new line = old column is created
                if aCell[1] not in result:
                    result[aCell[1]] = []
                result[aCell[1]].append([aCell[0], int(lineKey)])
        return result

    @staticmethod
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
                    if abs(bLine[indexes[0]][0] - aCell[0]) >= Matrix.EPSILON:
                        return False
        return True

    @staticmethod
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
        return result

    @staticmethod
    def multiply(a, b_temp):
        result = {}
        # 🚨 to make multiplication easier, we use the transpose
        b = Matrix.transpose(b_temp.copy())

        for aLineIndex in range(0, int(sorted(a.keys())[-1]) + 1):
            if aLineIndex not in a:
                continue
            else:
                result[aLineIndex] = []
                for bLineIndex in range(0, int(sorted(a.keys())[-1]) + 1):
                    if bLineIndex not in b:
                        continue
                    # after we pass the cases with [0,0,...,0] (so no lines/column declared), we multiply elements based on their line+col match
                    else:
                        sum = 0
                        aLine = a[aLineIndex]
                        bLine = b[bLineIndex]
                        for aCell in aLine:
                            # find the matching bCell
                            indexes = [index for index in range(
                                0, len(bLine)) if bLine[index][1] == aCell[1]]
                            if (indexes is not None and len(indexes) > 0):
                                bCell = bLine[indexes[0]]
                                sum += aCell[0] * bCell[0]
                        if (sum > 0):
                            result[aLineIndex].append([sum, bLineIndex])
        return result
    @staticmethod
    def multiply_v2(a, b_temp):
        result = {}
        # 🚨 to make multiplication easier, we use the transpose
        b = Matrix.sortMatrix(Matrix.transpose(b_temp.copy()))
        print("Start multiply\n")
        for a_line in a.keys():
            # print("[AAAAAAAAAAAAAAAAA] " + str(a_line) + "\n", end="")
            for b_line in b.keys():
                # print("[BBBBB] " + str(b_line) + '\n', end="")
                a_list = a[a_line]
                b_list = b[b_line]

                a_index = 0
                b_index = 0

                sum = 0

                while a_index < len(a_list) and b_index < len(b_list):
                    a_col = a_list[a_index][1]
                    b_col = b_list[b_index][1]
                    if a_col < b_col:
                        a_index += 1
                    elif a_col == b_col:
                        sum += a_list[a_index][0] * b_list[b_index][0]
                        a_index += 1
                        b_index += 1
                    else:
                        b_index += 1

                if sum > 0:
                    if a_line in result:
                        result[a_line].append([sum, b_line])
                    else:
                        result[a_line] = [[sum, b_line]]
        print("end multiply")
        return result


    ###### UTILITY FUNCTIONS ######

    @staticmethod
    def readFromFile(filename):
        f = open("resources/"+filename, "r")
        lines = []
        for x in f:
            lines.append(x.rstrip())
        return lines
    @staticmethod
    def getMatrix(filename):
        lines = Matrix.readFromFile(filename)
        lines.pop(0)
        matrix = dict()
        for line in lines:
            data = line.split(", ")
            if len(data) != 3:
                continue
            number = float(data[0])
            lin = int(data[1])
            col = int(data[2])
            elt = matrix.get(lin, "none")
            if elt == "none":
                matrix[lin] = [[number,col]]
            else:
                exists = False
                for cell in elt:
                    if cell[1] == col:
                        cell[0] += number
                        exists = True
                        break
                if not exists:
                    matrix[lin].append([number, col])
        return Matrix.sortMatrix(matrix)


    @staticmethod
    def translate(filename, type="matrix"):
        lines = Matrix.readFromFile(filename)
        if(type == "vector"):
            del lines[-1]
            return [float(lines[i]) for i in range(0, len(lines))]
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
        return Matrix.sortMatrix(store)
    @staticmethod
    def sortMatrix(matrix):
        matrix2 = dict(sorted(matrix.items()))
        matrix = matrix2
        for k in matrix.keys():
            matrix[k].sort(key = lambda x: x[1])

        return matrix

    @staticmethod
    def printTranslated(matrix):
        for line in range(0, int(sorted(matrix.keys())[-1]) + 1):
            print(matrix[line] if line in matrix else [])
        print()

    @staticmethod
    def check10Rarity(a):
        for lineKey in a:
            aLine = a[lineKey]
            if (len(aLine) > 10):
                return False
        return True

    @staticmethod
    def checkDiagonalValues(matrix, wrongValue=0):
        for lineKey in matrix:
            matrixLine = matrix[lineKey]
            indexes = [index for index in range(
                0, len(matrixLine)) if matrixLine[index][1] == lineKey]
            # check if there is a diagonal value to search for on the current line
            if indexes is None or len(indexes) != 1:
                return False
            # check if the value is 0 or EPSILON-near-0
            if abs(matrixLine[indexes[0]][0] - wrongValue) <= Matrix.EPSILON:
                return False
        return True
