import numpy as np
from data.util import get_score
import sys

class Data(object):
    __class_description__ = """Class modelling Data a Binary Tree based on Pandas DataFrame"""
    __version__ = 0.1

    def __init__(self, df, class_var, var_desc):
        self.df = df.copy()

        # Dict with the info of input variables
        self.var_desc = var_desc
        self.class_var = class_var


    def sort_by(self, col_name):
        self.df = self.df.sort_values(col_name)

        if self.var_desc[col_name]["type"] == 'cir':
            #print(col_name, self.var_desc)
            #print(col_name, self.var_desc[col_name]["bounds"], self.var_desc[col_name]["bounds"][0][0], self.var_desc[col_name]["bounds"][-1][1])
            start = self.var_desc[col_name]["bounds"][0][0]
            end = self.var_desc[col_name]["bounds"][-1][1]

            if start > end:
                self.df = self.df.query('{} <= {} <= 360.0'.format(start, col_name)).copy().append(
                    self.df.query('0.0 <= {} < {}'.format(col_name, end)).copy())

        self.df.index = range(0, len(self.df.index))


    def split_generator(self):

        for var in list(self.var_desc.keys()):

            if self.var_desc[var]["method"] == "classic":
                if self.var_desc[var]["type"] == "cir" and self.var_desc[var]["bounds"] == [[-np.inf, np.inf]]:
                    iter_i = [0, 0]
                else:
                    iter_i = [None, 0]
            elif self.var_desc[var]["method"] == "subset":
                iter_i = [0, 0]
            else:
                raise Exception("Method not recognised.")

            self.sort_by(var)
            iter_idx = np.where(self.df[var].values[:-1] != self.df[var].values[1:])[0] + 1

            # Otherwise all the values are the same and it is not possible to split
            if iter_idx.shape[0] > 2:
                #print(iter_idx.shape[0])

                iter_idx = np.insert(iter_idx, 0, 0)
                iter_idx = np.insert(iter_idx, iter_idx.shape[0], len(self.df.index)-1)

                if iter_i[0] is None:
                    for idx in iter_idx[2:-2]:
                        yield (var, [None, idx],
                               self.df.iloc[:idx][self.class_var].values,
                               self.df.iloc[idx:][self.class_var].values)

                else:

                    for i in range(iter_idx.shape[0]):
                        for j in range(i+2, iter_idx.shape[0]-1):
                            if (i-j) % iter_idx.shape[0]-1 > 1:
                                yield (var, [iter_idx[i], iter_idx[j]],
                                       self.df.iloc[iter_idx[i]:iter_idx[j]][self.class_var].values,
                                       np.concatenate((self.df.iloc[:iter_idx[i]][self.class_var].values,
                                                       self.df.iloc[iter_idx[j]:][self.class_var].values)))



    def get_best_split(self):
        best_split = {'var_name': None, 'score': 0.0, 'index': [None, 0]}

        for (var_name, i, left, right) in self.split_generator():
            score = get_score(left, right)
            if score > best_split['score']:
                best_split.update({'var_name': var_name, 'score': score, 'index': i[:]})

        #print(best_split, self.df.iloc[i[1]][best_split["var_name"]], self.var_desc[best_split["var_name"]]["bounds"])

        return best_split
