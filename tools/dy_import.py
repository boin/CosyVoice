import requests
import os
import json
from pathlib import Path

base_folder = '.links'

def load_json_from_folders(folder_list, base=base_folder):
    result_dict = {}
    
    for folder in folder_list:
        # 构建每个文件夹中 dayan.json 的路径
        json_file_path = os.path.join(base, folder, 'dayan.json')
        
        # 检查文件是否存在
        if os.path.isfile(json_file_path):
            with open(json_file_path, 'r', encoding='utf-8') as json_file:
                try:
                    # 读取 JSON 文件内容
                    json_data = json.load(json_file)
                    # 将文件夹名和对应的 JSON 数据存入字典
                    result_dict[folder] = json_data
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from {json_file_path}: {e}")
        else:
            print(f"{json_file_path} does not exist.")
    
    return result_dict

actors = [ d.name for  d in os.scandir(base_folder) if d.is_dir() ]

print(f'found {actors}')

post = load_json_from_folders(actors, base_folder)

print(f'found voices {len(post)}')

#import pprint
#pprint.pprint(post)

dayan_endpoint = 'http://ttd-server:3333/api/actors/sync'

json = json.dumps(post, ensure_ascii=False)

print(json)

r = requests.post(dayan_endpoint, json=post)

print(r.status_code)

print(r.text)
