import numpy as np

def get_score(left_data, right_data):
    left_size = left_data.shape[0]
    right_size = right_data.shape[0]
    total_size = left_size + right_size
    
    return (left_size/total_size * np.var(left_data)) + (right_size/total_size * np.var(right_data))
