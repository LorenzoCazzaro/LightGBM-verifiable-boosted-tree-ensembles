import argparse
import os
from from_lightgbm_to_silva import from_lightgbm_json_to_silva
import lightgbm as lgb

_SAVE_GBDT_PATH = "gbdt/"
_GBDT_FILENAME = "lightgbm_{}_{}_{}_{}"
_SAVE_GBDT_LSE_PATH = "gbdt_lse/"
_GBDT_LSE_FILENAME = "lightgbm_lse_{}_{}_{}_{}_{}"
_DATASET_FOLDER = "./datasets/{}/dataset/"
_MODELS_FOLDER = "./datasets/{}/models/"

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str)
parser.add_argument('n_trees', type=int, default=101)
parser.add_argument('max_depth', type=int, default=6)
parser.add_argument('random_state', type=int, default=0)
parser.add_argument('k', type=float, default=-0.1)
parser.add_argument('--min_data_in_leaf', type=int, default=20)
args = parser.parse_args()

training_cmd = "./lightgbm config=train.conf data={} valid_data={} num_trees={} max_depth={} num_leaves={} k={} seed={} output_model={} min_data_in_leaf={} verbose=2"
data = _DATASET_FOLDER.format(args.dataset) + "training_set_normalized.csv"
valid_data = _DATASET_FOLDER.format(args.dataset) + "validation_set_normalized.csv"
num_trees = args.n_trees
max_depth = args.max_depth
num_leaves = 2**(int(max_depth)+1) #fa già lightgbm il -1
k = args.k
seed = args.random_state
min_data_in_leaf = args.min_data_in_leaf
output_gbdt_model = _MODELS_FOLDER.format(args.dataset) + (_SAVE_GBDT_PATH if k < 0 else _SAVE_GBDT_LSE_PATH) + (_GBDT_FILENAME.format(args.n_trees, args.max_depth, args.random_state, args.min_data_in_leaf) if k < 0 else _GBDT_LSE_FILENAME.format(args.n_trees, args.max_depth, args.random_state, args.k, args.min_data_in_leaf))

#training gbdt LSE model
os.system(training_cmd.format(data, valid_data, num_trees, max_depth, num_leaves, k, seed, output_gbdt_model + ".txt", min_data_in_leaf))
#convert gbdt LSE model to silva
#from txt to json
bst = lgb.Booster(model_file=output_gbdt_model + ".txt")
booster_json = bst.dump_model()
#from json to silva
f = open(output_gbdt_model + ".silva", "w")
f.write(from_lightgbm_json_to_silva(booster_json))
f.close()
