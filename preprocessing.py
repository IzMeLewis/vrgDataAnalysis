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
min_replies = 5
max_ref = 5
jsonl = False
remove_deleted = True
remove_ref = True
remove_links = True

def remove_ref(input_string):
    pattern = r">>>/[\w/]*\d+|>>\d+"
    return re.sub(pattern, "", input_string)

def extract_ref(input_string):
    pattern = r">>\d+"
    matches = re.findall(pattern, input_string)
    if matches:
        return matches
    return []

def get_reply_text(row, df):
    reply_ids = row['replies']

    # Handle None and non-list cases
    if not isinstance(reply_ids, list):
        return []
    
    reply_texts = []
    for reply_id in reply_ids:
        reply_id = reply_id.replace('>>', '')
        reply_text = df.loc[df['id'] == int(reply_id), 'cleantext'].tolist()
        if reply_text:
            reply_texts.extend(reply_text)
    return reply_texts

def remove_links(input_string):
    pattern = r'http[s]?://\S+|www\.\S+'
    return re.sub(pattern, '', input_string)

def remove_leading_trailing_newlines(input_string):
    return input_string.lstrip('\n').rstrip('\n')

def replace_newlines_with_space(input_string):
    return input_string.replace('\n', ' ')

def filter_word_count(dataset, text_column_name, min_length):
    dataset_filtered = dataset[dataset[text_column_name].apply(lambda x: len(x.split()) >= min_length)]
    return dataset_filtered

def filter_reply_count(dataset, min_replies):
    dataset_filtered = dataset[dataset['replies'].apply(lambda x: x is not None and (len(x) >= min_replies or x == []))]
    return dataset_filtered

def filter_ref_count(dataset, max_ref):
    dataset_filtered = dataset[dataset['references'].apply(lambda x: x is not None and (len(x) <= max_ref or x == []))]
    return dataset_filtered

def dump_json(data, file_path):
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def dump_jsonl(data, file_path):
    with open(file_path, 'w') as jsonl_file:
        for item in data:
            json.dump(item, jsonl_file)
            jsonl_file.write('\n')

def dump_dataset(df,name):
    output_list = []
    for index, row in df.iterrows():
        output_dict = {textOnlyKey: row['cleantext']}
        for i, reply_text in enumerate(row['reply_texts']):
            output_dict[f'reply{i+1}'] = reply_text
        output_list.append(output_dict)
    # dict_dataset = data.to_dict(orient='records')
    if jsonl:
        dump_jsonl(output_list, f"{name}.jsonl")
    else:
        dump_json(output_list, f"{name}.json")

def main():
    global jsonl
    dataset = pd.read_json(f"{old_json_filename}.json")
    dataset = dataset[dataset['isop'] == False]
    if remove_deleted:
        dataset = dataset[dataset['type'] != 'deleted']

    dataset['references'] = dataset['content'].apply(extract_ref)
    if remove_ref:
        dataset['content'] = dataset['content'].apply(remove_ref)
    if remove_links:
        dataset['content'] = dataset['content'].apply(remove_links)
    dataset['content'] = dataset['content'].apply(remove_leading_trailing_newlines)

    unwanted_text = ['',' ','\n']
    for symbol in unwanted_text:    
        dataset = dataset[dataset['content'] != symbol]
    
    dataset['cleantext'] = dataset['content'].apply(replace_newlines_with_space)

    dataset['reply_texts'] = dataset.apply(get_reply_text, args=(dataset,), axis=1)

    dataset = filter_word_count(dataset,'cleantext', min_word_count)

    dataset = filter_reply_count(dataset, min_replies)

    dataset = filter_ref_count(dataset,max_ref)

    print(dataset.head())

    #reduces size of dataset
    dataset = dataset.sample(frac=dataset_size).reset_index(drop=True)
    if test_size != 0:
        dataset_train, dataset_test = train_test_split(dataset, test_size=test_size, random_state=0)
        dump_dataset(dataset_train,f"{new_json_filename}_train")
        dump_dataset(dataset_test,f"{new_json_filename}_test")
    else:
        dump_dataset(dataset,new_json_filename)


main()
