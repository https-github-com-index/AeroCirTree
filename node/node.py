import copy
import pandas as pd
from node.util import bound_generator
from data.data import Data

def is_in_bounds(bounds, value):
    for bound in bounds:
        if bound[0] > bound[1]:
            if bound[0] <= value <= 360.0 or 0.0 <= value < bound[1]:
                return True
        elif bound[0] == 0.0 and value == 360:
                return True
        else:
            if bound[0] <= value < bound[1]:
                return True

    return False

class Node(object):
    __class_description__ = """Class modelling the actual structure of a binary tree node"""
    __version__ = 0.1

    def __init__(self, data, stop, variance, level=0):
        self.split_var = None
        self.data = data
        self.level = level
        self.stop = stop
        self.variance = variance
        self.left_child = None
        self.right_child = None
        
    def split(self):

        if len(self.data.df.index) > self.stop:
            split = self.data.get_best_split()

            if split['score'] > self.variance:

                self.split_var = split["var_name"]
                self.data.sort_by(self.split_var)

                outer_bounds, inner_bounds = bound_generator(self.data.df[split["var_name"]].values,
                                                             split['index'],
                                                             self.data.var_desc[split["var_name"]]["bounds"],
                                                             self.data.var_desc[split["var_name"]]['type'] == 'cir')

                #print(outer_bounds, inner_bounds)

                left_desc = copy.deepcopy(self.data.var_desc)
                left_desc[split["var_name"]]["bounds"] = inner_bounds

                right_desc = copy.deepcopy(self.data.var_desc)
                right_desc[split["var_name"]]["bounds"] = outer_bounds

                #if self.data.var_desc[split["var_name"]]["method"] == "classic":
                # Because the circular classic first split has to be treated as subset
                if split['index'][0] is None:
                    self.left_child = Node(Data(self.data.df.iloc[:split['index'][1]],
                                                self.data.class_var, left_desc),
                                           self.stop, self.variance, self.level+1)
                    self.right_child = Node(Data(self.data.df.iloc[split['index'][1]:],
                                                 self.data.class_var, right_desc),
                                            self.stop, self.variance, self.level+1)

                #elif self.data.var_desc[split["var_name"]]["method"] == "subset":
                else:
                    self.left_child = Node(Data(self.data.df.iloc[split['index'][0]:split['index'][1]],
                                                self.data.class_var, left_desc),
                                           self.stop, self.variance, self.level+1)
                    self.right_child = Node(Data(pd.concat([self.data.df.iloc[:split['index'][0]],
                                                            self.data.df.iloc[split['index'][1]:]]),
                                                 self.data.class_var, right_desc),
                                            self.stop, self.variance, self.level+1)


                self.left_child.split()
                self.right_child.split()
