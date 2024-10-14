import json
from pathlib import Path
from typing import Dict, Optional, Tuple

# 导出训练结果Actors/Voices给大雁JSON
"""
  sha1str: {
        name: 玄幻_旁白_大叔,
        voices: {
           '旁白_中性_001_505050_难道这白光除了可以治疗唐雅兰本人以外，对于他人也能奇效吗？' :{
                title: "旁白_中性_001",
                vad: [0.5, 0.5, 0.5],
                asr: "难道这白光除了可以治疗唐雅兰本人以外，对于他人也能奇效吗？"
            },
           '旁白_中性_001_505050_难道这白光除了可以治疗唐雅兰本人以外，对于他人也能奇效吗？' :{
                title: "旁白_中性_001",
                vad: [0.5, 0.5, 0.5],
                asr: "难道这白光除了可以治疗唐雅兰本人以外，对于他人也能奇效吗？"
            } ...
        }
      }
   }
"""


def parse_utt(content: str) -> Tuple[str, Dict[str, Dict[str, list]]]:
    data = {}
    lines = content.splitlines()
    for line in lines:
        if len(line) > 1:
            voice, actor = line.split(" ", 1)  # 使用 1 限制分割次数，确保 actor 包含空格
            parts = voice.split("_")
            if len(parts) == 5:
                title = f"{parts[0]}_{parts[1]}_{parts[2]}"
                vad = [int(parts[3][i:i + 2]) for i in range(0, 6, 2)]  # 提取 VAD 值
                asr = parts[4]

                # 将语音信息存储到字典中
                data[voice] = {"title": title, "vad": vad, "asr": asr}
    return actor, data

def export_dayan_json(utt_file: str, export_file: Optional[str] = None) -> Optional[Dict[str, Dict[str, list]]]:
    actor, data = parse_utt(Path(utt_file).read_text()) 

    # 构建数据字典
    json_data = {"name": actor, "voices": data}

    # 如果 export_file 为空，则返回数据字典
    if export_file is None:
        return json_data

    # 写入 JSON 文件
    with open(export_file, "w", encoding="utf-8") as file:
        json.dump(json_data, file, ensure_ascii=False, indent=2)  # 使用 indent 使 JSON 更易读