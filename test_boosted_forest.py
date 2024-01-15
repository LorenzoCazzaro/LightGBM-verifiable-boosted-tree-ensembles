import pandas as pd
import lightgbm as lgb
import argparse

_DATASET_FOLDER = "./datasets/{}/dataset/"
_MODELS_FOLDER = "./datasets/{}/models/"
_TEST_SET_NAME = "test_set_normalized.csv"
_SAVE_GBDT_PATH = "gbdt/"
_SAVE_GBDT_LSE_PATH = "gbdt_lse/"
_GBDT_FILENAME = "lightgbm_{}_{}_{}_{}"
_GBDT_LSE_FILENAME = "lightgbm_lse_{}_{}_{}_{}_{}"

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str)
parser.add_argument('n_trees', type=int, default=101)
parser.add_argument('max_depth', type=int, default=6)
parser.add_argument('random_state', type=int, default=0)
parser.add_argument('k', type=float, default=-0.1)
parser.add_argument('--min_data_in_leaf', type=int, default=20)
args = parser.parse_args()

gbdt_model = _MODELS_FOLDER.format(args.dataset) + (_SAVE_GBDT_PATH if args.k < 0 else _SAVE_GBDT_LSE_PATH) + (_GBDT_FILENAME.format(args.n_trees, args.max_depth, args.random_state) if args.k < 0 else _GBDT_LSE_FILENAME.format(args.n_trees, args.max_depth, args.random_state, args.k, args.min_data_in_leaf)) + ".txt"

booster = lgb.Booster(model_file=gbdt_model)

data = pd.read_csv(_DATASET_FOLDER.format(args.dataset) + _TEST_SET_NAME, delimiter = ",", skiprows = [0], header=None).to_numpy()
y = data[:, 0].astype(int)
X = data[:, 1:].astype(float)

predictions = booster.predict(X)
pred_class = (predictions > 0.5).astype("int")
print("ACC: {}".format(sum(pred_class == y)/len(y)))
