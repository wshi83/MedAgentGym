import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--local_path", type=str, default=None)
args = parser.parse_args()

file_list = os.listdir(args.local_path)
score = 0
for file_name in file_list:
    file_path = os.path.join(args.local_path, file_name)
    with open(file_path, 'r') as f:
        data = json.load(f)
    if data[-1]['result'] == 'success':
        score += 1

print(f"Score: {score}/{len(file_list)}={score/len(file_list)}")