import requests
import os
import logging as logger
from pathlib import Path

DATA_ROOT = os.environ["DATA_ROOT"] if "DATA_ROOT" in os.environ else "./data"
OUTPUT_ROOT = (
    os.environ["OUTPUT_ROOT"] if "OUTPUT_ROOT" in os.environ else f"{DATA_ROOT}/outputs"
)
VC_WEIGHT_ROOT = f"{DATA_ROOT}/vc_weights"


def vc_output_path(base_path, project_name) -> str:
    return Path(f"{OUTPUT_ROOT}/{project_name}/vc/{base_path}")


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


def load_keytone_from_actor(actor: str) -> str:
    # 师父,师傅_ZZJ,MY,XXXX_+3_001
    parts = actor.split("_")
    return parts[2] if len(parts) == 4 else "0"


def load_vc_actor_ref(project_name, actor):
    path = Path(VC_WEIGHT_ROOT) / project_name / f"{actor}.wav"
    return path if path.is_file() else None


def request_vc(
    project_name: str, sid: str, audio_path: str, result_name: str, tone_key: str
) -> (int, str):
    """
    发送语音转换请求并保存结果。
     audio_path = ‘http://ttd-server/file/cosy-voice/output/5b/d9/5bd95bc0-3359-45e6-bd87-243de0d46138.wav’
     定义要测试的 URL

    参数:
    project_name (str): 项目名称。
    sid (str): 语音 ID。
    audio_path (str): 输入音频文件的路径。
    result_name (str): 输出结果文件的名称。
    tone_key (str): 音调键。

    返回:
    tuple: 状态码和消息。
    """

    url = "http://ttd-server:7878/infer_vc"

    # 请求的 payload
    request_payload = {
        "actor": f"{project_name}/{sid}",
        "index": tone_key,
    }

    try:
        with open(audio_path, "rb") as audio_file:
            files = {"file": audio_file}
            response = requests.post(url, params=request_payload, files=files)
            response.raise_for_status()  # 如果请求失败，抛出异常

            # 将内容写入文件
            with open(result_name, "wb") as wav_file:
                wav_file.write(response.content)

            logger.info("Response length: %d", len(response.content))
            return 0, "Success"

    except requests.exceptions.HTTPError as http_err:
        message = f"HTTP error occurred: {http_err} - Response content: {response.text}"
        logger.error(message)
        return 1, message

    except requests.exceptions.RequestException as e:
        message = f"Request error: {e}"
        logger.error(message)
        return 1, message
