import glob
import os
from pathlib import Path

from tools.vad import findNearestKW, findNearestVAD, findNearestASR

DATA_ROOT = os.environ["DATA_ROOT"] if "DATA_ROOT" in os.environ else "./data"
MODEL_ROOT = f"{DATA_ROOT}/models"
TTD_LIB = f"{DATA_ROOT}/ttd_lib/"
LIB_SUB = "02_E-Motion/Tag"


def load_projects(root_dir=MODEL_ROOT):
    return [
        d.name
        for d in os.scandir(f"{root_dir}")
        if d.is_dir() and not d.name.startswith(".") and not d.name.startswith("@")
    ]


def load_prj_actors(project_name, root_dir=MODEL_ROOT):
    return [
        d.name
        for d in os.scandir(f"{root_dir}/{project_name}")
        if d.is_dir() and not d.name.startswith(".") and not d.name.startswith("@")
    ]


def load_lib_projects(root_dir=TTD_LIB):
    lib_projects = [
        d.name
        for d in os.scandir(f"{root_dir}")
        if d.is_dir() and not d.name.startswith(".") and not d.name.startswith("@")
    ]
    return lib_projects


def load_lib_prj_actors(project_name, root_dir=TTD_LIB):
    lib_actors = [
        d.name
        for d in os.scandir(f"{root_dir}/{project_name}/{LIB_SUB}")
        if d.is_dir() and not d.name.startswith(".") and not d.name.startswith("@")
    ]
    # print(lib_actors)
    return lib_actors


def check_proj_actor_wavs(project_name, actor):
    count = len(
        glob.glob(str(Path(TTD_LIB) / project_name / LIB_SUB / actor) + "/*.wav")
    )
    return count


def get_uut_by_name(project_name, actor, exact=False):
    for dir in [
        d.name
        for d in os.scandir(f"{MODEL_ROOT}/{project_name}")
        if d.is_dir() and not d.name.startswith(".") and not d.name.startswith("@")
    ]:
        if dir.find(actor) > -1:
            return dir
    if not exact:  # 非精确返回最后一个结果，作为debug
        return dir


def load_refrence_wav(project_name, actor, voice):
    """根据项目名称，角色和音色加载试听wav文件"""
    return Path(TTD_LIB) / project_name / LIB_SUB / actor / f"{voice}.wav"


def load_refrence(
    project_name,
    actor: str,
    emo: [str or int, str or int, str or int] or None,
    emo_kw: str,
    text: str,
):
    """加载最接近的参考音，
        1. 使用VAD筛选
        2. 关键KeyWord匹配内容
        3. ASR文本匹配  如！？直接匹配对应语料
        旁白_脸红_003_507828_谁能拿到紫青双剑，一切都看运气。.wav
    Args:
        actor (str): 古装_旁白,ZYH,_灵异
        emo (str or int, str or int, str or int]orNone): VAD
        emo_kw : strings of emo KeyWord
    """
    # print("load refrence called:", project_name, actor, emo, emo_kw)
    root_dir = f"{MODEL_ROOT}/{project_name}/{actor}"
    # for compability
    content = Path(f"{root_dir}/output/train/temp1/utt2spk").read_text().splitlines()
    if emo:
        vad = [
            int(float(emo[0]) * 100),
            int(float(emo[1]) * 100),
            int(float(emo[2]) * 100),
        ]
        # print(vad)
        # vad find 10 matches
        vad_content = findNearestVAD(vad, content, 20)
    voices = vad_content
    # kw_voice = findNearestKW(emo_kw, content)
    # if kw_voice != voices[0]: #dont add same voice
    #    voices.insert(0, kw_voice) # KW voice first
    # asr_match
    # asr_match = findNearestASR(text, content)
    # if asr_match and asr_match != voices[0]:
    #    voices.insert(0, voices.pop(voices.index(asr_match)))
    return [f.split(" ")[0] for f in voices]


def load_actor(actor: str, project_name):
    """
        加载所有Actors信息，根据制定项目名称。
        每一个项目中可能有很多个角色目录，按照目录名匹配
        ls data/有声书_殓葬禁忌
            ├── 古装_旁白,ZYH,_灵异
            ├── 古装_师父,GZJ_灵异
            ├── 古装_师父,LJDY_灵异
            └── 古装_肖魏魃,LCM_灵异
    Returns:
        人物角色列表
        []
    """
    # print('load actor called:', actor, project_name)
    root_dir = f"{MODEL_ROOT}/{project_name}"
    # content = Path(f"{root_dir}/output/train/temp1/spk2utt").read_text()
    with os.scandir(root_dir) as entries:
        # 创建一个列表，包含文件和其修改时间
        dirs = [
            (f.name, f.stat().st_mtime)
            for f in entries
            if f.is_dir() and not f.name.startswith(".") and not f.name.startswith("@")
        ]
    # 根据修改时间进行排序，越新越后
    content = sorted(dirs, key=lambda x: x[1])

    # print('loaded content:', content, 'need:', actor)
    spks = []
    for item, _ in content:
        if item and item.find("_") > -1:
            spkr = item.split("_")[1]  # 旁白,ZYH,
            if spkr.find(actor) > -1:
                spks.insert(
                    0, item
                )  # 命中了模型，放在列表的第一个，多次命中，那就是选中最后一个
            else:
                spks.append(item)
        # fail-back
        elif item:
            spks.append(item)  # TODO remove later

    # print("globing:", content, "got:", spks)
    return spks
