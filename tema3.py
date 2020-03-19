

def readFromFile(filename):
    f = open("resources/"+filename, "r")
    lines = []
    for x in f:
        lines.append(x.rstrip())
    return lines


def bind(filename):
    lines = readFromFile(filename)
    end = True
    store = []
    size = None
    for line in lines:
        if(size == None):
            size = int(line)
            store = [[] for i in range(size)]
            continue
        data = line.split(", ")
        # last line has one element (probably a blank line)
        if(len(data) != 3):
            continue
        print(data)
        store[int(data[1])] = [float(data[0]), float(data[2])]
    print(store)


bind("a.txt")
