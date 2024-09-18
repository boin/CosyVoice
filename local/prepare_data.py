import argparse
import logging
import glob
import os
import shutil
import gradio as gr
from pathlib import Path 
import math
from tqdm import tqdm
from tools.auto_ttd import TTD_LIB, LIB_SUB

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger()

#240915_有声书_殓葬禁忌/02_E-Motion/Tag/古装_师父,GZJ_灵异/旁白_脸红_003_507828_谁能拿到紫青双剑，一切都看运气。.wav

def init_from_lib(prj_name, actor, split_ratio):
    """
    项目名，角色名, 训练集和验证集的分配比例
    """

    src_dir = Path(TTD_LIB) / prj_name / LIB_SUB / actor
    if not src_dir.is_dir():
        raise gr.Error(f"导入语料库 {src_dir} 目录不存在!")
    wavs = [ f.path for f in os.scandir(src_dir) if f.is_file() and f.name.endswith('.wav') and not f.name.startswith('.')]
    count = len(wavs)
    if count < 2: raise gr.Error("语料少于2条，无法训练")
    stop_num = math.floor(count * split_ratio) * 2 #最小1
    tr_dir = f'./data/{prj_name}/{actor}/train'
    vl_dir = f'./data/{prj_name}/{actor}/val'
    tr_cnt = 0 
    vl_cnt = 0
    os.makedirs(tr_dir, exist_ok=True)
    os.makedirs(vl_dir, exist_ok=True)
    for i in range(count):
        if i < stop_num and i%2 == 1 : #分配验证集
            shutil.copyfile(wavs[i], os.path.join(vl_dir, Path(wavs[i]).name))
            vl_cnt+=1
            continue
        shutil.copyfile(wavs[i], os.path.join(tr_dir, Path(wavs[i]).name)) #分配训练集
        tr_cnt+=1
    return tr_cnt, vl_cnt

def prepare_normalize_txt(file):
    txt_path = file.rpartition(".wav")[0] + ".normalized.txt"
    if not os.path.exists(txt_path):
        logger.warning('{} do not exsist'.format(txt_path))
        meta = os.path.basename(file).rpartition('.wav')[0].split("_")
        if len(meta) != 5:
            raise gr.Error(f"invalid file meta: {file}")
        Path(txt_path).write_text(meta[4])
        logger.info(f'{txt_path} created.')
    return txt_path

def main():
    #src_dir = data/有声书_殓葬禁忌/古装_师父,GZJ_灵异/train
    src_dir = Path(args.src_dir)
    stage = src_dir.name  #train
    src_dir = src_dir.parents[1]  # data/有声书_殓葬禁忌
    actor = args.actor       #古装_师父,GZJ_灵异

    wavs = list(glob.glob('{}/{}/{}/*wav'.format(src_dir, actor, stage)))
    #./data/240915_有声书_殓葬禁忌/古装_师父,GZJ_灵异/train/旁白_脸红_003_507828_谁能拿到紫青双剑，一切都看运气。.wav
    if len(wavs) < 1 or args.force_flag:
        if len(wavs) < 1: logger.warning(f"{src_dir}/{actor}/{stage}/*wav 没有wav文件，开始初始化")
        if args.force_flag: logger.info("强制重新初始化")
        [tc, cc] = init_from_lib(os.path.basename(src_dir), actor=actor, split_ratio=args.init_split_ratio)
        logger.info(f"{src_dir} {actor} {stage}初始化完毕， 训练集文件数量 {tc}， 测试集文件数量 {cc}")
        wavs = list(glob.glob('{}/{}/{}/*wav'.format(src_dir, actor, stage))) #Re-Scan
    utt2wav, utt2text, utt2spk, spk2utt = {}, {}, {}, {}
    for wav in tqdm(wavs):
        txt = prepare_normalize_txt(wav)
        with open(txt, encoding='utf-8') as f:
            content = ''.join(l.replace('\n', '') for l in f.readline())
        utt = Path(wav).stem #提取 utt 如 旁白_脸红_003_507828_谁能拿到紫青双剑，一切都看运气。.wav  => 旁白_脸红_003_507828_谁能拿到紫青双剑，一切都看运气。
        spk = actor
        utt2wav[utt] = wav
        utt2text[utt] = content
        utt2spk[utt] = spk
        if spk not in spk2utt:
            spk2utt[spk] = []
        spk2utt[spk].append(utt)

    with open('{}/wav.scp'.format(args.des_dir), 'w') as f:
        for k, v in utt2wav.items():
            f.write('{} {}\n'.format(k, v))
    with open('{}/text'.format(args.des_dir), 'w') as f:
        for k, v in utt2text.items():
            f.write('{} {}\n'.format(k, v))
    with open('{}/utt2spk'.format(args.des_dir), 'w') as f:
        for k, v in utt2spk.items():
            f.write('{} {}\n'.format(k, v))
    with open('{}/spk2utt'.format(args.des_dir), 'w') as f:
        for k, v in spk2utt.items():
            f.write('{} {}\n'.format(k, ' '.join(v)))
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir',
                        type=str)
    parser.add_argument('--des_dir',
                        type=str)
    parser.add_argument('--actor',
                        type=str)
    parser.add_argument('--init_split_ratio',
                        type=int)
    parser.add_argument('--force_flag',
                        type=bool)

    args = parser.parse_args()
    main()
