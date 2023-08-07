import datetime
import calendar
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.model_selection import train_test_split

old_json_filename = "test"
new_json_filename = "vrg_clean"
textOnlyKey = "Text"
dataset_size = 1 #float
test_size = 0.2 #float
min_word_count = 5
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

def remove_leading_trailing_newlines(input_string):
    return input_string.lstrip('\n').rstrip('\n')

def replace_newlines_with_space(input_string):
    return input_string.replace('\n', ' ')

def filter_word_count(dataset, text_column_name, min_length):
    dataset_filtered = dataset[dataset[text_column_name].apply(lambda x: len(x.split()) >= min_word_count)]
    return dataset_filtered

def dump_json(data, file_path):
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def dump_jsonl(data, file_path):
    with open(file_path, 'w') as jsonl_file:
        for item in data:
            json.dump(item, jsonl_file)
            jsonl_file.write('\n')

def dump_dataset(data,name):
    dict_dataset = data.to_dict(orient='records')
    if jsonl:
        dump_jsonl(dict_dataset, f"{name}.jsonl")
    else:
        dump_json(dict_dataset, f"{name}.json")

def main():
    global jsonl
    dataset = pd.read_json(f"{old_json_filename}.json")
    dataset = dataset[dataset['isop'] == False]
    if remove_deleted:
        dataset = dataset[dataset['type'] != 'deleted']
    if remove_ref:
        dataset['content'] = dataset['content'].apply(remove_ref)
    if remove_links:
        dataset['content'] = dataset['content'].apply(remove_links)
    dataset['content'] = dataset['content'].apply(remove_leading_trailing_newlines)

    unwanted_text = ['',' ','\n']
    for symbol in unwanted_text:    
        dataset = dataset[dataset['content'] != symbol]
    
    dataset['cleantext'] = dataset['content'].apply(replace_newlines_with_space)

    dataset = filter_word_count(dataset,'cleantext', min_word_count)

    print(dataset.head(10))

    dataset = dataset.drop(columns=dataset.columns.difference(['content']))
    dataset.rename(columns={'content': textOnlyKey}, inplace=True)

    #reduces size of dataset
    dataset = dataset.sample(frac=dataset_size).reset_index(drop=True)

    dataset_train, dataset_test = train_test_split(dataset, test_size=test_size, random_state=0)

    dump_dataset(dataset_train,f"{new_json_filename}_train")
    dump_dataset(dataset_test,f"{new_json_filename}_test")

main()
