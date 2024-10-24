import requests
import os
from pathlib import Path

VC_WEIGHT_ROOT = "./data/.vc_weights"


def load_vc_actor(project_name: str, actor: str | None = None):
    actors = []
    disk_actors = os.scandir(f"{VC_WEIGHT_ROOT}/{project_name}")

    for f in disk_actors:
        if f.is_file() and not f.name.startswith(".") and f.name.endswith(".pth"):
            role = Path(f).stem.split("_")[0]  # 【师父,师傅】_ZZJ,MY,XXXX_+3_001
            if actor and actor in role:
                actors.insert(0, Path(f).stem)  # 将匹配的角色放在开头，不带扩展名
            else:
                actors.append(Path(f).stem)  # 其他角色放在末尾，不带扩展名
    return actors


def load_keytone_from_actor(actor: str) -> str | None:
    # 师父,师傅_ZZJ,MY,XXXX_+3_001
    parts = actor.split("_")
    return parts[2] if len(parts) == 4 else None


def load_vc_actor_ref(project_name, actor):
    path = Path(VC_WEIGHT_ROOT) / project_name / f"{actor}.wav"
    return path if path.is_file() else None


def request_vc(project_name, sid, audio_path, result_name, tone_key):
    # audio_path = 'http://ttd-server/file/cosy-voice/output/5b/d9/5bd95bc0-3359-45e6-bd87-243de0d46138.wav'
    # 定义要测试的 URL
    url = "http://ttd-server:7878/infer_vc"

    # 定义请求的 payload
    request = {
        "actor": f"{project_name}/{sid}",  # 确保这里的值是有效的
        "index": tone_key,  # 确保这里的值是有效的
    }

    # 发送 POST 请求
    with open(audio_path, "rb") as f:
        files = {"file": f}  # 文件上传
        try:
            response = requests.post(url, params=request, files=files)
            response.raise_for_status()  # 如果请求失败，抛出异常
            # 将内容写入文件
            with open(result_name, "wb") as wav_file:
                wav_file.write(response.content)
            print("Response length:", len(response.content))
        except requests.exceptions.HTTPError as http_err:
            print(
                f"HTTP error occurred: {http_err} - Response content: {response.text}"
            )
        except requests.exceptions.RequestException as e:
            print("Error:", e)
