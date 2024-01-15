import lightgbm as lgb
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('txt_filename', type=str)
parser.add_argument('json_filename', type=str)
args = parser.parse_args()

bst = lgb.Booster(model_file=args.txt_filename)

booster_json = bst.dump_model()
f = open(args.json_filename, "w")
json.dump(booster_json, f)
f.close()