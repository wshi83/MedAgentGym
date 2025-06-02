import json
import pandas as pd
import random

random.seed(123)

with open("test_data_v2.json", "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# print(df.head())
train_idx, test_idx = [], []
for i in range(10):
    task_test_idx = random.sample([i for i in range(30)], int(30 * 0.2))
    for j in range(30):
        if j in task_test_idx:
            test_idx.append(i*30 + j)
        else:
            train_idx.append(i*30 + j)

train_df = df.loc[train_idx]
test_df = df.loc[test_idx]

train_df.to_csv(f"train_tasks.csv", index=False)
train_df.to_json(f"train_tasks.json", orient='records')
train_df.to_json(f"train_tasks.jsonl", orient='records', lines=True)

test_df.to_csv(f"test_tasks.csv", index=False)
test_df.to_json(f"test_tasks.json", orient='records')
test_df.to_json(f"test_tasks.jsonl", orient='records', lines=True)