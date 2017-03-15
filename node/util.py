import numpy as np

"""
def origin_remover(bounds):
    new_bounds = []
    init = None

    for bound in bounds:
        if bound[1] == 360.0:
            init = bound[0]
        elif bound[0] == 0.0 and init is not None:
            new_bounds.append([init, bound[1]])
            init = None
        else:
            new_bounds.append(bound)

    return new_bounds
"""

def bound_bonder(bounds):
    new_bounds = []
    prev = None
    for bound in bounds:
        if prev:
            if prev[1] == bound[0] or (prev[1] == 360.0 and bound[0] == 0.0):
                prev = [prev[0], bound[1]]
            else:
                new_bounds.append(prev)
                prev = bound
        else:
            prev = bound

    if prev:
        new_bounds.append(prev)

    return new_bounds


def get_outter(values, bound):
    if bound[1] < values[0] or bound[0] > values[1]:
        #print("X1")
        return [bound]
    else:
        #print("X2")
        new_bound = []
        if bound[0] < values[0]:
            new_bound.append([bound[0], values[0]])
        if bound[1] > values[1]:
            new_bound.append([values[1], bound[1]])
        return new_bound

def get_inner(values, bound):

    if not(bound[1] < values[0] or bound[0] > values[1]):
        if bound[0] > values[0]:
            if bound[0] != values[1]:
                return [[bound[0], min(bound[1], values[1])]]
            else:
                return []
        else:
            return [[values[0], min(bound[1], values[1])]]

    else:
        return []


def bound_generator(data, index, bounds, circular):

    #print("Input <-", bounds)

    if circular:
        #First cicular split
        if bounds == [[-np.inf, np.inf]]:
            #print("A Output <-", [[data[index[1]], data[index[0]]]], [[data[index[0]], data[index[1]]]])
            return [[data[index[1]], data[index[0]]]], [[data[index[0]], data[index[1]]]]

        #Subsequent classical cicular splits
        elif index[0] == None:
            #print("B Output <-", [[data[index[1]], bounds[0][1]]], [[bounds[0][0], data[index[1]]]])
            return [[data[index[1]], bounds[0][1]]], [[bounds[0][0], data[index[1]]]]

        else:
            #print("Lund Bounds:", bounds)
            outter = []
            inner = []
            for bound in bounds:
                #print(bound)
                if bound[0] > bound[1]:
                    #print("a")

                    if data[index[0]] > data[index[1]]:

                        for split in get_outter((data[index[0]], 360.0), [bound[0], 360.0]):
                            #print("A1-o", split)
                            outter.append(split)
                        for split in get_outter((0.0, data[index[1]]), [0.0, bound[1]]):
                            #print("B1-o", split)
                            outter.append(split)
                        for split in get_inner((data[index[0]], 360.0), [bound[0], 360.0]):
                            #print("C1-i", split)
                            inner.append(split)
                        for split in get_inner((0.0, data[index[1]]), [0.0, bound[1]]):
                            #print("D1-i", split)
                            inner.append(split)
                    else:
                        for split in get_outter((data[index[0]], data[index[1]]), [bound[0], 360.0]):
                            #print("A2-o", split)
                            outter.append(split)
                        for split in get_outter((data[index[0]], data[index[1]]), [0.0, bound[1]]):
                            #print("B2-o", split)
                            outter.append(split)
                        for split in get_inner((data[index[0]], data[index[1]]), [bound[0], 360.0]):
                            #print("C2-i", split)
                            inner.append(split)
                        for split in get_inner((data[index[0]], data[index[1]]), [0.0, bound[1]]):
                            #print("D2-i", split)
                            inner.append(split)

                else:
                    #print("b")
                    if data[index[0]] > data[index[1]]:
                        for split in get_outter((data[index[0]], 360.0), bound):
                            #print("A3-o", split)
                            outter.append(split)
                        for split in get_outter((0.0, data[index[0]]), bound):
                            #print("B3-o", split)
                            outter.append(split)
                        for split in get_inner((data[index[0]], 360.0), bound):
                            #print("C3-i", split)
                            inner.append(split)
                        for split in get_inner((0.0, data[index[0]]), bound):
                            #print("D3-i", split)
                            inner.append(split)
                    else:
                        for split in get_outter((data[index[0]], data[index[1]]), bound):
                            #print("A3-o", split)
                            outter.append(split)
                        for split in get_inner((data[index[0]], data[index[1]]), bound):
                            #print("B3-i", split)
                            inner.append(split)

            #print("C Output <-", outter, inner)
            #print("C Output <-", bound_bonder(outter), bound_bonder(inner))

            return bound_bonder(outter), bound_bonder(inner)

    else:
        if index[0] is None:
            #print("D Output <-", get_outter((-np.inf, data[index[1]]), bounds[0]), get_inner((-np.inf, data[index[1]]), bounds[0]))
            return get_outter((-np.inf, data[index[1]]), bounds[0]), get_inner((-np.inf, data[index[1]]), bounds[0])

        else:
            outter = []
            inner = []
            for bound in bounds:
                for split in get_outter((data[index[0]], data[index[1]]), bound):
                    outter.append(split)
                for split in get_inner((data[index[0]], data[index[1]]), bound):
                    inner.append(split)

            #print("E Output <-", outter, inner)
            return outter, inner


