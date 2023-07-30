import datetime
import calendar
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

old_json_filename = "vrg"
new_json_filename = "vrg_clean"
textOnlyKey = "Text"
size = 1 #float
jsonl = False
remove_deleted = True
remove_ref = True
remove_links = True

def remove_ref(input_string):
    pattern = r">>\d+"
    return re.sub(pattern, "", input_string)
def remove_links(input_string):
    pattern = r'http[s]?://\S+|www\.\S+'
    return re.sub(pattern, '', input_string)

def dump_json(data, file_path):
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def dump_jsonl(data, file_path):
    with open(file_path, 'w') as jsonl_file:
        for item in data:
            json.dump(item, jsonl_file)
            jsonl_file.write('\n')
def main():
    dataset = pd.read_json(f"{old_json_filename}.json")
    dataset = dataset[dataset['isop'] == False]
    if remove_deleted:
        dataset = dataset[dataset['type'] != 'deleted']
    if remove_ref:
        dataset['content'] = dataset['content'].apply(remove_ref)
    if remove_links:
        dataset['content'] = dataset['content'].apply(remove_links)
    dataset = dataset[dataset['content'] != '']
    dataset = dataset.drop(columns=dataset.columns.difference(['content']))
    dataset.rename(columns={'content': textOnlyKey}, inplace=True)
    dataset = dataset.sample(frac=size).reset_index(drop=True)
    dict_dataset = dataset.to_dict(orient='records')
    if jsonl:
        dump_jsonl(dict_dataset, f"{new_json_filename}.jsonl")
    else:
        dump_json(dict_dataset, f"{new_json_filename}.json")

main()
