import numpy as np

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

    if circular:
        #First cicular split
        if bounds == [[-np.inf, np.inf]]:
            return [[data[index[1]], data[index[0]]]], [[data[index[0]], data[index[1]]]]

        #Subsequent classical cicular splits
        elif index[0] == None:
            return [[data[index[1]], bounds[0][1]]], [[bounds[0][0], data[index[1]]]]

        else:
            outter = []
            inner = []
            for bound in bounds:
                if bound[0] > bound[1]:

                    if data[index[0]] > data[index[1]]:

                        for split in get_outter((data[index[0]], 360.0), [bound[0], 360.0]):
                            outter.append(split)
                        for split in get_outter((0.0, data[index[1]]), [0.0, bound[1]]):
                            outter.append(split)
                        for split in get_inner((data[index[0]], 360.0), [bound[0], 360.0]):
                            inner.append(split)
                        for split in get_inner((0.0, data[index[1]]), [0.0, bound[1]]):
                            inner.append(split)
                    else:
                        for split in get_outter((data[index[0]], data[index[1]]), [bound[0], 360.0]):
                            outter.append(split)
                        for split in get_outter((data[index[0]], data[index[1]]), [0.0, bound[1]]):
                            outter.append(split)
                        for split in get_inner((data[index[0]], data[index[1]]), [bound[0], 360.0]):
                            inner.append(split)
                        for split in get_inner((data[index[0]], data[index[1]]), [0.0, bound[1]]):
                            inner.append(split)

                else:
                    if data[index[0]] > data[index[1]]:
                        for split in get_outter((data[index[0]], 360.0), bound):
                            outter.append(split)
                        for split in get_outter((0.0, data[index[0]]), bound):
                            outter.append(split)
                        for split in get_inner((data[index[0]], 360.0), bound):
                            inner.append(split)
                        for split in get_inner((0.0, data[index[0]]), bound):
                            inner.append(split)
                    else:
                        for split in get_outter((data[index[0]], data[index[1]]), bound):
                            outter.append(split)
                        for split in get_inner((data[index[0]], data[index[1]]), bound):
                            inner.append(split)

            return bound_bonder(outter), bound_bonder(inner)

    else:
        if index[0] is None:
            return get_outter((-np.inf, data[index[1]]), bounds[0]), get_inner((-np.inf, data[index[1]]), bounds[0])

        else:
            outter = []
            inner = []
            for bound in bounds:
                for split in get_outter((data[index[0]], data[index[1]]), bound):
                    outter.append(split)
                for split in get_inner((data[index[0]], data[index[1]]), bound):
                    inner.append(split)

            return outter, inner


