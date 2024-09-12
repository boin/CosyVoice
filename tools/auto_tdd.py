from pathlib import Path
from tools.vad import findNearestVAD

# 取得最相似的情绪关键词结果
def findNearestKW(keyword):
    return []

def load_refrence(project_name, actor: str or None, emo: [str or int, str or int, str or int] or None):
    print("load refrence called:", actor, emo)
    root_dir = f"./data/{project_name}"
    content = Path(f"{root_dir}/output/train/temp1/utt2spk").read_text().splitlines()
    if emo:
        vad = [float(emo[0]) * 100, float(emo[1]) * 100, float(emo[2]) * 100]
        print(vad)
        search_content = findNearestVAD(vad, content, 20)
        content = search_content or content
    voices = []
    spk_voice = []
    for item in content:
        if item:
            [voice, spkr] = item.split(" ")
            voices.append(f"{spkr} - {voice}")
            if actor and spkr == actor:
                spk_voice.append(voice)
    return spk_voice or voice


def load_actor(actor: str, project_name):
    """
        加载所有Actors信息，根据制定项目名称
    Returns:
        人物角色列表
    """
    # print('load actor called:', actor, project_name)
    root_dir = f"./data/{project_name}"
    content = Path(f"{root_dir}/output/train/temp1/spk2utt").read_text()
    spks = []
    for item in content.split("\n"):
        if item:
            spkr = item.split(" ")[0]
            spks.append(f"{spkr}")

    #print("globing:", content, "got:", spks)
    return spks
