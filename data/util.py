import numpy as np
import math

def circular_mean(data):
    x = np.sum(np.cos(np.radians(data)))
    y = np.sum(np.sin(np.radians(data)))

    return math.degrees(math.atan2(y, x)) % 360

def circular_variance(data):
    c = np.sum(np.cos(np.radians(data)))
    s = np.sum(np.sin(np.radians(data)))

    return 1 - math.sqrt(math.pow(c/data.shape[0], 2) + math.pow(s/data.shape[0], 2))

def get_score(left_data, right_data, circular=False):
    
    left_size = float(len(left_data))
    right_size = float(len(right_data))
    total_size = left_size + right_size

    if circular:
        ponderate_var = ((left_size / total_size) * circular_variance(left_data)) + ((right_size / total_size) * circular_variance(right_data))
        total_var = circular_variance(np.concatenate((left_data, right_data), axis=0))
        return total_var - ponderate_var
        
    ponderate_var = ((left_size / total_size) * np.var(left_data)) + ((right_size / total_size) * np.var(right_data))
    total_var = np.var(np.concatenate((left_data, right_data), axis=0))
    return total_var - ponderate_var
