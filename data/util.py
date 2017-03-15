import numpy as np

def get_score(left_data, right_data):

    left_size = float(len(left_data))
    right_size = float(len(right_data))
    total_size = left_size + right_size

    #ponderate_var = ((left_size/total_size)**2 * np.var(left_data)) + ((right_size/total_size)**2 * np.var(right_data))
    ponderate_var = (left_size / total_size * np.var(left_data)) + (right_size / total_size * np.var(right_data))
    total_var = np.var(np.concatenate((left_data, right_data), axis=0))

    return total_var - ponderate_var