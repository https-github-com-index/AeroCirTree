import json
import random
import pickle
import time
import sys
import math
import copy
import os.path
import pandas as pd
import numpy as np
import argparse
import itertools

from data.data import Data
from node.node import Node

def tree_pprinter(node):
    fmtr = ""

    def pprinter(anode):
        nonlocal fmtr

        if anode.split_var is not None:
            print("({0}) {1} {2}".format(len(anode.data.df.index), anode.split_var, anode.data.var_desc[anode.split_var]['bounds']))

        else:
            print("({0}) Leaf {1:.2f} {2}".format(len(anode.data.df.index), np.var(anode.data.df[anode.data.class_var].values),
                                        ["{} {}".format(key, anode.data.var_desc[key]['bounds']) for key in anode.data.var_desc.keys() if anode.data.var_desc[key]['bounds'] != [[-np.inf, np.inf]]]))

        if anode.left_child is not None:
            print("{} `--".format(fmtr), end="")
            fmtr += "  | "
            pprinter(anode.left_child)
            fmtr = fmtr[:-4]

            print("{} `--".format(fmtr), end="")
            fmtr += "    "
            pprinter(anode.right_child)
            fmtr = fmtr[:-4]

    return pprinter(node)


def tree_trainer(df, class_var, var_desc, stop=50, variance=.001):
        if class_var not in df.columns:
            raise Exception('Class variable not in DataFrame')
    
        data = Data(df, class_var, var_desc)

        node = Node(data, stop=stop, variance=variance)
        node.split()

        return node

def is_in_bounds(bounds, value):
    for bound in bounds:
        if bound[0] > bound[1]:
            if bound[0] <= value <= 360.0 or 0.0 <= value < bound[1]:
                return True
            elif bound[0] == 0.0 and value == 360:
                return True
        else:
            if bound[0] == 0.0 and value == 360:
                return True
            elif bound[0] <= value < bound[1]:
                return True

    return False


def tree_eval(node, row):
    result = None

    def eval(node, row):
        nonlocal result

        if node.split_var is None:
            result = np.mean(node.data.df[node.data.class_var].values)

        else:
            if is_in_bounds(node.left_child.data.var_desc[node.split_var]["bounds"], row[node.split_var]):
                eval(node.left_child, row)
            elif is_in_bounds(node.right_child.data.var_desc[node.split_var]["bounds"], row[node.split_var]):
                eval(node.right_child, row)
            else:
                print(node.data.var_desc[node.split_var]["bounds"], row[node.split_var])
                print(node.left_child.data.var_desc[node.split_var]["bounds"], row[node.split_var])
                print(node.right_child.data.var_desc[node.split_var]["bounds"], row[node.split_var])

    eval(node, row)

    return result

def cxval_k_folds_split(df, k_folds, seed=1):
    random.seed(seed)

    dataframes = []
    group_size = int(round(df.shape[0]*(1.0/k_folds)))

    for i in range(k_folds-1):
        rows = random.sample(list(df.index), group_size)
        dataframes.append(df.ix[rows])
        df = df.drop(rows)

    dataframes.append(df)

    return dataframes


def cxval_select_fold(i_fold, df_folds):
    df_folds_copy = copy.deepcopy(df_folds)

    if 0 <= i_fold < len(df_folds):
        test_df = df_folds_copy[i_fold]
        del df_folds_copy[i_fold]
        train_df = pd.concat(df_folds_copy)
        return train_df, test_df

    else:
        raise Exception('Group not in range!')


def cxval_test(df_folds, class_var, var_desc, leaf_size):
    rmse_results = []
    ia_results = []
    ia2_results = []

    for i in range(len(df_folds)):
        train_df, test_df = cxval_select_fold(i, df_folds)
        tree = tree_trainer(train_df, class_var, var_desc, leaf_size)
        rmse_results.append(tree_rmse_calc(tree, test_df))
        ia_results.append(tree_ia_calc(tree, test_df))
        ia2_results.append(tree_ria_calc(tree, test_df))

    return sum(rmse_results)/len(df_folds), sum(ia_results)/len(df_folds), sum(ia2_results)/len(df_folds)


def tree_rmse_calc(node, df):
    acc = 0.0
    total_len = len(df.index)

    for _, row in df.iterrows():
        acc += math.pow((tree_eval(node, row) - row[node.data.class_var]), 2)

    return math.sqrt(acc / total_len)


def tree_ia_calc(node, df):
    total_len = len(df.index)
    sim = np.zeros(total_len)
    df.reset_index(drop=True, inplace=True)
    for i, row in df.iterrows():
        sim[i] = tree_eval(node, row)

    obs = df[node.data.class_var]

    return 1 - np.sum(np.square(obs - sim)) / np.sum(np.square(np.abs(sim - np.mean(obs)) + np.abs(obs - np.mean(obs))))


def tree_ria_calc(node, df):
    total_len = len(df.index)
    sim = np.zeros(total_len)
    df.reset_index(drop=True, inplace=True)
    for i, row in df.iterrows():
        sim[i] = tree_eval(node, row)

    obs = df[node.data.class_var]
    first = np.sum(np.abs(obs - sim))
    second = (2*np.sum(np.abs(obs - np.mean(obs))))
    if first <= second:
        return 1 - first/second
    else:
        return second/first - 1

def config_generator():
    outputs = ["metar_wind_spd", "metar_temp", "metar_rh"]
    inputs_lin = ["gfs_wind_spd", "gfs_temp", "gfs_rh"]
    inputs_cir = ["gfs_wind_dir", "time", "date"]

    for output in outputs:
        for l in range(1, len(inputs_lin)+1):
            for subset_lin in itertools.combinations(inputs_lin, l):
                if not "gfs"+output[5:] in subset_lin:
                    continue
                for subset_cir in itertools.combinations(inputs_cir, l):
                    config = {}
                    config["output"] = output
                    config["input"] = []
                    for var in subset_lin:
                        var_desc = {}
                        var_desc["name"] = var
                        var_desc["type"] = "lin"
                        config["input"].append(var_desc)
                    for var in subset_cir:
                        var_desc = {}
                        var_desc["name"] = var
                        var_desc["type"] = "cir"
                        config["input"].append(var_desc)
                    yield config

if __name__ == "__main__":
   
    if len(sys.argv) == 1:
        print("Please specify a tree configuration as an argument to this script")
        sys.exit()

    with open(sys.argv[1]) as conf_file:
        tree_params = json.load(conf_file)
        class_var = tree_params['output']['name']

        tree_desc_lin = {}
        for var in tree_params['input']:
            tree_desc_lin[var['name']] = {"type": "lin", "method": "cont", "bounds": [[-np.inf, np.inf]]}
    
        tree_desc_lund = {}
        for var in tree_params['input']:
            if var["type"] == "cir":
                tree_desc_lund[var['name']] = {"type": var['type'], "method": "non-cont", "bounds": [[-np.inf, np.inf]]}
            else: 
                tree_desc_lund[var['name']] = {"type": var['type'], "method": "cont", "bounds": [[-np.inf, np.inf]]}

        tree_desc_cir = {}
        for var in tree_params['input']:
            tree_desc_cir[var['name']] = {"type": var['type'], "method": "cont", "bounds": [[-np.inf, np.inf]]}

        tree_desc_uv = {}
        for var in tree_params['input']:
            if var['name'] == "gfs_wind_dir":
                tree_desc_uv["u_speed"] = {"type": "lin", "method": "cont", "bounds": [[-np.inf, np.inf]]}
                tree_desc_uv["v_speed"] = {"type": "lin", "method": "cont", "bounds": [[-np.inf, np.inf]]}
            if var['name'] == "time":
                tree_desc_uv["u_time"] = {"type": "lin", "method": "cont", "bounds": [[-np.inf, np.inf]]}
                tree_desc_uv["v_time"] = {"type": "lin", "method": "cont", "bounds": [[-np.inf, np.inf]]}
            if var['name'] == "date":
                tree_desc_uv["u_date"] = {"type": "lin", "method": "cont", "bounds": [[-np.inf, np.inf]]}
                tree_desc_uv["v_date"] = {"type": "lin", "method": "cont", "bounds": [[-np.inf, np.inf]]}
            if var['type'] == "lin" and var['name'] != "gfs_wind_spd":
                tree_desc_uv[var['name']] = {"type": "lin", "method": "cont", "bounds": [[-np.inf, np.inf]]}

        #print(tree_desc_lin)
        #print(tree_desc_lund)
        #print(tree_desc_cir)
        #print(tree_desc_uv)
        print("")
        print("")
        print("| Airport | Method   | 1000   | 500    | 250    | 100    | 50     |")
        print("| ------- |----------| ------:|-------:| ------:|-------:| ------:|")

        data_paths = ["datasets/eddt.csv", "datasets/egll.csv", "datasets/lebl.csv", "datasets/lfpg.csv", "datasets/limc.csv"]
        airport_codes = ['EDDT', 'EGLL', 'LEBL', 'LFPG', 'LIMC']
        for ds_idx, dataset in enumerate(data_paths):
            df = pd.read_csv(dataset)
          
            for var in tree_params['input']:
                if var['type'] == "cir":
                    df[var['name']].replace([360], 0, inplace=True)
            
            df["v_speed"] = df["gfs_wind_spd"] * np.sin(np.deg2rad(df["gfs_wind_dir"]))
            df["u_speed"] = df["gfs_wind_spd"] * np.cos(np.deg2rad(df["gfs_wind_dir"]))
            df["v_time"] = np.sin(np.deg2rad(df["time"]))
            df["u_time"] = np.cos(np.deg2rad(df["time"]))
            df["v_date"] = np.sin(np.deg2rad(df["date"]))
            df["u_date"] = np.cos(np.deg2rad(df["date"]))
            
            df_folds = cxval_k_folds_split(df, 5, 0)
            print("| {}    |".format(airport_codes[ds_idx]), end='')
            tree_names = ["Linear", "Lund", "Circular", "UV"]
            for tree_idx, tree_desc in enumerate([tree_desc_lin, tree_desc_lund, tree_desc_cir, tree_desc_uv]):
                if tree_idx == 0:
                    print(" {0: <8} |".format(tree_names[tree_idx]), end='')
                else:
                    print("|         | {0: <8} |".format(tree_names[tree_idx]), end='')
                for size in [1000, 500, 250, 100, 50]:
                    rmse, ia, ria = cxval_test(df_folds, class_var, tree_desc, size)
                    print(" {0:0.4f} |".format(ria), end='')
                print("", flush=True)
