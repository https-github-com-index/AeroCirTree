import numpy as np
from data.util import get_score
import sys

class Data(object):
    __class_description__ = """Class modelling Data a Binary Tree based on Pandas DataFrame"""
    __version__ = 0.1

    def __init__(self, df, class_var, var_desc):
        self.df = df.copy()
        self.var_desc = var_desc
        self.class_var = class_var


    def sort_by(self, col_name):
        self.df = self.df.sort_values(col_name)

        if self.var_desc[col_name]["type"] == 'cir':
            #print(self.var_desc[col_name]["bounds"])
            start = self.var_desc[col_name]["bounds"][0][0]
            end = self.var_desc[col_name]["bounds"][-1][1]

            if start > end:
                self.df = self.df.query('{} <= {} <= 360.0'.format(start, col_name)).copy().append(
                    self.df.query('0.0 <= {} < {}'.format(col_name, end)).copy())

        self.df.index = range(0, len(self.df.index))


    def split_generator(self):
        for var_name in list(self.var_desc.keys()):
            self.sort_by(var_name)
            iter_idx = np.where(self.df[var_name].values[:-1] != self.df[var_name].values[1:])[0] + 1

            # All the values are the same and it is not possible to split
            if iter_idx.shape[0] == 0:
                continue
           
            """
            if self.var_desc[var_name]["method"] == "cont":
                if self.var_desc[var_name]["type"] == "cir" and self.var_desc[var_name]["bounds"] == [[-np.inf, np.inf]]:
                    iter_idx = np.insert(iter_idx, 0, 0)
                    iter_idx = np.insert(iter_idx, iter_idx.shape[0], len(self.df.index)-1)

                    for i, start in enumerate(iter_idx):
                        for end in iter_idx[i+1:-1]:
                            yield (var_name, [start, end], 
                                    np.concatenate((self.df.iloc[:start][self.class_var].values, 
                                       self.df.iloc[end:][self.class_var].values)), 
                                    self.df.iloc[start:end][self.class_var].values)
            """              

            # If the variable is non-contiguous or if it's the first circular split
            if self.var_desc[var_name]["method"] == "non-cont" or self.var_desc[var_name]["type"] == "cir" and self.var_desc[var_name]["bounds"] == [[-np.inf, np.inf]]:
                iter_idx = np.insert(iter_idx, 0, 0)
                iter_idx = np.insert(iter_idx, iter_idx.shape[0], len(self.df.index)-1)

                for i, start in enumerate(iter_idx):
                    for end in iter_idx[i+1:-1]:
                        yield (var_name, [start, end], 
                                np.concatenate((self.df.iloc[:start][self.class_var].values, 
                                   self.df.iloc[end:][self.class_var].values)), 
                                self.df.iloc[start:end][self.class_var].values)
            else: 
                for i in iter_idx:
                    yield (var_name, [i, None], self.df.iloc[:i][self.class_var].values, self.df.iloc[i:][self.class_var].values)


    def get_best_split(self):
        best_split = {'var_name': None, 'score': np.inf, 'index': [None, 0]}

        for (var_name, idx, left, right) in self.split_generator():
            score = get_score(left, right)
            if score < best_split['score']:
                best_split.update({'var_name': var_name, 'score': score, 'index': idx[:]})

        return best_split
