import argparse
import logging
import glob
import os
import shutil
import gradio as gr
from pathlib import Path
import math
from tqdm import tqdm
from hashlib import md5
from tools.dayan import export_dayan_json
from tools.auto_ttd import TTD_LIB, LIB_SUB, DATA_ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger()

# 240915_有声书_殓葬禁忌/02_E-Motion/Tag/古装_师父,GZJ_灵异/旁白_脸红_003_507828_谁能拿到紫青双剑，一切都看运气。.wav
LNIK_DIR = f"{DATA_ROOT}/links"
os.makedirs(LNIK_DIR, mode=0o777, exist_ok=True)


def init_from_lib(prj_name, actor, split_ratio, src_dir=None):
    """
    项目名，角色名, 训练集和验证集的分配比例，
    增加项目外部语料库
    """

    src_dir = Path(src_dir) if src_dir else Path(TTD_LIB) / prj_name / LIB_SUB / actor
    if not src_dir.is_dir():
        raise gr.Error(f"导入语料库 {src_dir} 目录不存在!")
    wavs = [
        f.path
        for f in os.scandir(src_dir)
        if f.is_file() and f.name.endswith(".wav") and not f.name.startswith(".")
    ]
    count = len(wavs)
    if count < 2:
        raise gr.Error("语料少于2条，无法导入")
    stop_num = (
        math.floor(count * split_ratio) * 2
    )  # 最小1， 0为自定义分配模式 # 小于0则是不分配一点给验证集
    model_dir = (
        Path(DATA_ROOT) / "models" / actor
        if prj_name == "models"  # dayan 特殊逻辑
        else Path(DATA_ROOT) / "models" / prj_name / actor
    )
    tr_dir = model_dir / "train"
    vl_dir = model_dir / "val"
    tr_cnt = 0
    vl_cnt = 0
    os.makedirs(tr_dir, exist_ok=True)
    os.makedirs(vl_dir, exist_ok=True)
    for i in range(count):
        idx = Path(wavs[i]).name.split("_")[2]  # 取音色序号供奇偶分配模式判断
        idx = idx[2] if len(idx) > 0 else 1  # 未按照命名格式的wav默认音色序号为1
        if (stop_num == 0 and int(idx) % 2 == 0) or (  # idx 偶数
            i < stop_num and i % 2 == 1  # 或者i偶数
        ):  # 分配验证集
            shutil.copyfile(wavs[i], os.path.join(vl_dir, Path(wavs[i]).name))
            vl_cnt += 1
            continue
        shutil.copyfile(
            wavs[i], os.path.join(tr_dir, Path(wavs[i]).name)
        )  # 其它都是默认分配训练集
        tr_cnt += 1
    return tr_cnt, vl_cnt


def prepare_normalize_txt(file):
    if file.count(" ") > 0:
        raise gr.Error(f"文件路径和文件名不得含有空格：{file}")
    fn_length = len(Path(file).name.encode("utf-8"))
    # print(f'finename {Path(file).name} actual length: {fn_length}')
    if fn_length > 244:
        raise gr.Error(f"文件名长度不得超过255个字符: {file}")
    txt_path = file.rpartition(".wav")[0] + ".normalized.txt"
    if not os.path.exists(txt_path):
        # logger.warning('{} do not exsist'.format(txt_path))
        meta = os.path.basename(file).rpartition(".wav")[0].split("_")
        if len(meta) != 5:
            raise gr.Error(f"文件元数据错误: {file}")
        Path(txt_path).write_text(meta[4])
        logger.info(f"{txt_path} created.")
    return txt_path


def calculate_wav_hash(file_path, bytes_to_read=1024):
    hash_md5 = md5()

    with open(file_path, "rb") as f:
        # 读取文件开头的字节
        start_bytes = f.read(bytes_to_read)
        hash_md5.update(start_bytes)

        # 移动到文件末尾
        f.seek(0, 2)  # 移动到文件末尾
        file_size = f.tell()  # 获取文件大小

        # 确保不越界，读取结尾的字节
        if file_size > bytes_to_read:
            f.seek(file_size - bytes_to_read)  # 移动到文件末尾前 bytes_to_read 字节
            end_bytes = f.read(bytes_to_read)
            hash_md5.update(end_bytes)

    return hash_md5.hexdigest()


def main():
    # src_dir = data/有声书_殓葬禁忌/古装_师父,GZJ_灵异/train
    # train / val 文件夹会调用此方法2遍，请注意安全
    src_dir = Path(args.src_dir)
    stage = src_dir.name  # train
    prj_dir = src_dir.parents[1]  # data/有声书_殓葬禁忌
    # dayan  模式下会变成 prj_dir = data/models，特殊处理，后续重构
    actor = args.actor  # 古装_师父,GZJ_灵异
    raw_dir = args.raw_dir

    folder_hash = md5()

    wavs = list(glob.glob("{}/{}/{}/*wav".format(prj_dir, actor, stage)))
    # ./data/models/240915_有声书_殓葬禁忌/ 古装_师父,GZJ_灵异/ train/ 旁白_脸红_003_507828_谁能拿到紫青双剑，一切都看运气。.wav
    if len(wavs) < 1 or args.force_flag == "True":
        if len(wavs) < 1:
            logger.warning(f"{prj_dir}/{actor}/{stage}/*wav 没有wav文件，开始初始化")
        if args.force_flag:
            logger.warning("强制重新初始化")
            shutil.rmtree(Path(f"{prj_dir}/{actor}/{stage}"), ignore_errors=True)
        [tc, cc] = init_from_lib(
            os.path.basename(prj_dir),
            actor=actor,
            split_ratio=args.init_split_ratio,
            src_dir=raw_dir,
        )
        logger.info(
            f"{prj_dir} {actor} {stage}初始化完毕， 训练集文件数量 {tc}， 测试集文件数量 {cc}"
        )
        wavs = list(glob.glob("{}/{}/{}/*wav".format(prj_dir, actor, stage)))  # Re-Scan
    utt2wav, utt2text, utt2spk, spk2utt = {}, {}, {}, {}
    for wav in tqdm(wavs):
        txt = prepare_normalize_txt(wav)
        folder_hash.update(calculate_wav_hash(wav).encode("utf-8"))
        with open(txt, encoding="utf-8") as f:
            content = "".join(line.replace("\n", "") for line in f.readline())
        utt = Path(wav).stem
        # 提取 utt 如 旁白_脸红_003_507828_谁能拿到紫青双剑，一切都看运气。.wav  => 旁白_脸红_003_507828_谁能拿到紫青双剑，一切都看运气。
        spk = actor
        utt2wav[utt] = wav
        utt2text[utt] = content
        utt2spk[utt] = spk
        if spk not in spk2utt:
            spk2utt[spk] = []
        spk2utt[spk].append(utt)

    with open("{}/wav.scp".format(args.des_dir), "w") as f:
        for k, v in utt2wav.items():
            f.write("{} {}\n".format(k, v))
    with open("{}/text".format(args.des_dir), "w") as f:
        for k, v in utt2text.items():
            f.write("{} {}\n".format(k, v))
    with open("{}/utt2spk".format(args.des_dir), "w") as f:
        for k, v in utt2spk.items():
            f.write("{} {}\n".format(k, v))
    with open("{}/spk2utt".format(args.des_dir), "w") as f:
        for k, v in spk2utt.items():
            f.write("{} {}\n".format(k, " ".join(v)))
    if stage == "train":
        link_name = (
            Path(LNIK_DIR)
            / (md5(actor.encode("utf-8")) if raw_dir else folder_hash).hexdigest()
        )
        try:
            os.unlink(link_name)
        except Exception:
            pass
        # ../models/240915_有声书_殓葬禁忌/古装_老八,DZVC_灵异 -> 9d87535ca928d1a5214dd7f84b39d6ad
        # (dayan) ../models/古装_老八,DZVC_灵异 -> 9d87535ca928d1a5214dd7f84b39d6ad
        link_src = (
            f"../models/{actor}"
            if raw_dir
            else f"../models/{os.path.basename(prj_dir)}/{actor}"
        )
        os.symlink(link_src, link_name)
        export_dayan_json(
            "{}/utt2spk".format(args.des_dir), f"{src_dir.parents[0]}/dayan.json"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str)
    parser.add_argument("--des_dir", type=str)
    parser.add_argument("--actor", type=str)
    parser.add_argument("--init_split_ratio", type=int)
    parser.add_argument("--force_flag", type=str)
    parser.add_argument("--raw_dir", type=str, default=None)

    args = parser.parse_args()
    main()
