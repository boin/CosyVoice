import requests
import os
import json
from pathlib import Path

base_folder = "/opt/CosyVoice/data/.links"


def make_ref_link(actor, voice: str | None = None) -> str:
    link = "http://ttd-server/ref/"
    link += f"{actor}/{voice}.wav" if voice else f"{actor}.wav"
    return link


def make_voices(actor, voices: dict):
    result = {}
    best_voice = ""
    intro_file = Path(base_folder) / actor / "intro.wav"

    for key, voice in voices.items():  # 使用 items() 遍历字典
        voice["src"] = make_ref_link(actor, key)
        result[key] = voice
        if len(key) > len(best_voice):
            best_voice = key
    # 复制最佳声音到 intro_file
    best_file = Path(base_folder) / actor / "train" / f"{best_voice}.wav"
    intro_file.write_bytes(best_file.read_bytes())
    return result


def load_and_clean_actors():
    actors = []
    for d in os.scandir(base_folder):
        if not d.is_dir():  # 如果不是目录
            if d.is_symlink() and not d.exists():  # 检查是否是失效的符号链接
                Path(d).unlink()  # 删除失效的符号链接
            continue

        # 如果是有效的目录，添加到 actors 列表
        actors.append(d.name)
    return actors


def load_voice_from_folders(folder_list, base=base_folder):
    result_dict = {}

    for folder in folder_list:
        # 构建每个文件夹中 dayan.json 的路径
        json_file_path = os.path.join(base, folder, "dayan.json")

        # 检查文件是否存在
        if os.path.isfile(json_file_path):
            with open(json_file_path, "r", encoding="utf-8") as json_file:
                try:
                    # 读取 JSON 文件内容
                    json_data = json.load(json_file)
                    # 检查并创建每个Actor 的 intro.wav，同时填充voice的src字段
                    voice_data = make_voices(folder, json_data)
                    # 将文件夹名和对应的 JSON 数据存入字典
                    result_dict[folder] = voice_data
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from {json_file_path}: {e}")
        else:
            print(f"{json_file_path} does not exist.")

    return result_dict


actors = load_and_clean_actors()
print(f"found {actors}")

post_data = load_voice_from_folders(actors, base_folder)

print(f"found voices {len(post_data)}")

# import pprint
# pprint.pprint(post)

dayan_endpoint = "http://ttd-server:3333/api/actors/sync"

json = json.dumps(post_data, ensure_ascii=False)

print(json)

r = requests.post(dayan_endpoint, json=post_data)

print(r.status_code)

print(r.text)
