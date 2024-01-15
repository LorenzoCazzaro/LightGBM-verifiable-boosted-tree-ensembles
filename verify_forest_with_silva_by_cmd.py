import argparse
import os
from from_lightgbm_to_silva import from_lightgbm_json_to_silva
import lightgbm as lgb

_SAVE_GBDT_PATH = "gbdt/"
_GBDT_FILENAME = "lightgbm_{}_{}_{}_{}"
_SAVE_GBDT_LSE_PATH = "gbdt_lse/"
_GBDT_LSE_FILENAME = "lightgbm_lse_{}_{}_{}_{}_{}"
_MODELS_FOLDER = "./datasets/{}/models/"
_DATASET_FOLDER = "./datasets/{}/dataset/"
_TEST_SET = "test_set_normalized.csv"

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str)
parser.add_argument('n_trees', type=int, default=101)
parser.add_argument('max_depth', type=int, default=6)
parser.add_argument('random_state', type=int, default=0)
parser.add_argument('k_model', type=float, default=-0.1)
parser.add_argument('k_verif', type=float, default=-0.1)
parser.add_argument('--min_data_in_leaf', type=int, default=20)

args = parser.parse_args()

verify_cmd = "./silva/src/silva {} {} --perturbation l_inf {} --voting softargmax > {}"
model_folder = _SAVE_GBDT_PATH if args.k_model < 0 else _SAVE_GBDT_LSE_PATH
model_filename = _GBDT_FILENAME.format(args.n_trees, args.max_depth, args.random_state, args.min_data_in_leaf) if args.k_model < 0 else _GBDT_LSE_FILENAME.format(args.n_trees, args.max_depth, args.random_state, args.k_model, args.min_data_in_leaf)
model_path = _MODELS_FOLDER.format(args.dataset) + model_folder + model_filename + ".silva"
dataset_path = _DATASET_FOLDER.format(args.dataset) + _TEST_SET
log_path = _MODELS_FOLDER.format(args.dataset) + model_folder + "log_silva_" + model_filename + "_" + str(args.k_verif) + ".txt"

#convert from txt to json
os.system(verify_cmd.format(model_path, dataset_path, args.k_verif, log_path))
