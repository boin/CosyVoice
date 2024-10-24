# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import argparse
import logging
import os
import json
import random
import warnings
from tempfile import NamedTemporaryFile
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from cosyvoice.cli.model import CosyVoiceModel
from cosyvoice.dataset.dataset import Dataset

warnings.simplefilter(action="ignore", category=FutureWarning)

logging.getLogger("matplotlib").setLevel(logging.WARNING)

def parse_tts_text(text: str):
    try:
        json.loads(text)
    except Exception:
        text = text.lstrip("'").rstrip("'")
    return text

def get_args():
    parser = argparse.ArgumentParser(description="inference with your model")
    parser.add_argument("--config", required=True, help="config file")
    parser.add_argument("--prompt_data", required=True, help="prompt data file")
    parser.add_argument("--prompt_utt2data", required=True, help="prompt data file")
    parser.add_argument("--tts_text", required=True, help="tts input file")
    parser.add_argument("--llm_model", required=True, help="llm model file")
    parser.add_argument("--flow_model", required=True, help="flow model file")
    parser.add_argument("--hifigan_model", required=True, help="hifigan model file")
    parser.add_argument(
        "--gpu", type=int, default=-1, help="gpu id for this rank, -1 for cpu"
    )
    parser.add_argument(
        "--mode", default="sft", choices=["sft", "zero_shot"], help="inference mode"
    )
    parser.add_argument("--result_dir", required=True, help="asr result file")
    parser.add_argument("--file_name", required=False, help="result file name")
    parser.add_argument("--rseed", required=False, help="setup a ramdon seed")
    parser.add_argument("--mix", required=False, help="another voice to mix")
    parser.add_argument(
        "--mix_pt", required=False, help="pt path contains the voice embedding"
    )
    parser.add_argument("--mix_rate", required=False, help="mix rate with origin voice")

    args = parser.parse_args()
    logging.info(args)
    return args


def main():
    """
    取传入的各种参数(一个实例)
    config='conf/cosyvoice.yaml',   #项目自带的配置文件
    prompt_data='data/LHB_0.5MIN_ID30/output/train/temp2/data.list',         # prompt 数据为之前预处理的数据列表
    prompt_utt2data='data/LHB_0.5MIN_ID30/output/train/temp2/utt2data.list', # utt数据同理
    tts_text='data/LHB_0.5MIN_ID30/output/outputs/tts_text.json',            # tts_text就是要合成的文本
    llm_model='data/LHB_0.5MIN_ID30/output/models/epoch_5_whole.pt',         # llm 模型地址，这就是训练出来的模型
    flow_model='pretrained_models/CosyVoice-300M/flow.pt',                   # flow 模型地址，这是内置模型
    hifigan_model='pretrained_models/CosyVoice-300M/hift.pt',                # hifigan 模型地址，这是内置模型
    gpu=0,                                                                   # 使用第一块GPU
    mode='zero_shot',                                                        # 推理模式 zero_shot
    result_dir='data/LHB_0.5MIN_ID30/output/outputs'                         # 输出地址
    file_name=''                                                             # 文件名称
    rssed=234324231                                                          # 随机数
    mix_pt='output/train/temp1/utt2embedding.pt'                             # 某一条语料音色混合的PT文件
    mix = 'utt'                                                              # 音色混合的音色名
    mix_rate = 0.5-0.3                                                       # 音色混合的比例
    """
    args = get_args()

    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s"
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Init cosyvoice models from configs 这是从config.yaml里面取默认配置
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    with open(args.config, "r") as f:
        configs = load_hyperpyyaml(f)
    # re-seed after load_conf
    seed = int(args.rseed) if args.rseed != "None" else None
    if seed:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 使用配置里的llm/flow/hift 初始化一个CV模型，并加载对应的模型文件
    model = CosyVoiceModel(configs["llm"], configs["flow"], configs["hift"])
    model.load(args.llm_model, args.flow_model, args.hifigan_model)

    tts_text = args.tts_text
    if not os.path.isfile(tts_text): # tts_text 是 raw_json
        tts_text = parse_tts_text(tts_text)
        tmpf = NamedTemporaryFile(mode="w+", suffix=".json")
        tmpf.write(tts_text)
        tmpf.seek(0) # w模式不seek回去，后面都读不到内容
        logging.info(f'{tts_text} is not file. write it to {tmpf.name}')
        tts_text = tmpf.name

    # 用参数中的合成文本，prompt数据初始化一个测试数据集并放到一个加载器（Torch.Dataloader）中
    test_dataset = Dataset(
        args.prompt_data,
        data_pipeline=configs["data_pipeline"],
        mode="inference",
        shuffle=False,
        partition=False,
        tts_file=tts_text,
        prompt_utt2data=args.prompt_utt2data,
    )
    test_data_loader = DataLoader(test_dataset, batch_size=None, num_workers=0)

    # 不重要的清理内存/检查文件夹等操作
    del configs
    os.makedirs(args.result_dir, exist_ok=True)
    fn = os.path.join(args.result_dir, "wav.scp")
    f = open(fn, "w")
    tts_key = ""
    # 初始化torch
    with torch.no_grad():
        # 以 测试数据集中的每一行数据作为循环（实际上推理的时候只循环一次，因为只有一行数据）
        for batch_idx, batch in tqdm(enumerate(test_data_loader)):
            utts = batch[
                "utts"
            ]  # {'utts': ['李弘彬_疑惑']  <--- 这里取的就是在WebUI里选的音色模型，就是['李弘彬_疑惑']
            assert len(utts) == 1, "inference mode only support batchsize 1"
            text = batch["text"]  # noqa: F841
            text_token = batch["text_token"].to(device)
            text_token_len = batch["text_token_len"].to(device)
            tts_text = batch["tts_text"]  # noqa: F841
            tts_index = batch["tts_index"]
            tts_text_token = batch["tts_text_token"].to(device)
            tts_text_token_len = batch["tts_text_token_len"].to(device)
            speech_token = batch["speech_token"].to(device)
            speech_token_len = batch["speech_token_len"].to(device)
            speech_feat = batch["speech_feat"].to(device)
            speech_feat_len = batch["speech_feat_len"].to(device)
            utt_embedding = batch["utt_embedding"].to(device)
            spk_embedding = batch["spk_embedding"].to(device)
            # 以上是各种LHB的变量绑定

            # 如果是sft推理
            if args.mode == "sft":
                # 定义input为 合成文本的分词token 合成文字长度 llm模型的embedding 是 spk_embedding  flow模型的embedding 也是 spk_embedding
                # 传embedding的时候是把整个人的传过去了，包含训练集中的所有音色元数据。
                model_input = {
                    "text": tts_text_token,
                    "text_len": tts_text_token_len,
                    "llm_embedding": spk_embedding,
                    "flow_embedding": spk_embedding,
                }
            else:
                # 如果是zero-shot推理
                # input的区别是使用了测试数据集中的数据
                model_input = {
                    "text": tts_text_token,
                    "text_len": tts_text_token_len,
                    "prompt_text": text_token,
                    "prompt_text_len": text_token_len,
                    "llm_prompt_speech_token": speech_token,
                    "llm_prompt_speech_token_len": speech_token_len,
                    "flow_prompt_speech_token": speech_token,
                    "flow_prompt_speech_token_len": speech_token_len,
                    "prompt_speech_feat": speech_feat,
                    "prompt_speech_feat_len": speech_feat_len,
                    "llm_embedding": utt_embedding,
                    "flow_embedding": utt_embedding,
                }
            # 如果涉及到音色融合，那么使用的flow_embedding使用融合的pt版本
            # 根据 voice 取到对应的 embbeding
            if args.mix:
                pt = torch.load(args.mix_pt)
                mix_embbeding = pt[args.mix]
                [r1, r2] = args.mix_rate.split("-")
                if mix_embbeding and r1 and r2:
                    # print(r1, r2, mix_embbeding, utt_embedding.tolist()[0])
                    model_input["flow_embedding"] = torch.tensor(
                        np.array(
                            [  # 绑定成2维数组
                                np.asarray(utt_embedding.tolist()[0]) * float(r1)
                                + np.asarray(mix_embbeding) * float(r2)
                            ]
                        ),
                        dtype=torch.float32,
                    )
            # 这就是开始干活了…
            model_output = model.inference(**model_input)
            # testing
            out_audio = model_output["tts_speech"]
            # 以下是把数据存成音频文件（wav.scp），不重要了。
            tts_key = args.file_name or f"{utts[0]}_{tts_index[0]}"
            tts_fn = os.path.join(args.result_dir, f"{tts_key}.wav")
            torchaudio.save(tts_fn, out_audio, sample_rate=22050)
            # audio normalize
            data, sr = sf.read(tts_fn)  # load audio
            # measure the loudness first
            meter = pyln.Meter(sr)  # create BS.1770 meter
            loudness = meter.integrated_loudness(data)
            # loudness normalize audio to -25 dB LUFS
            out_audio = pyln.normalize.loudness(data, loudness, -25.0)
            sf.write(tts_fn, out_audio, sr)
            f.write(f"{tts_key} {tts_fn}\n")
            f.flush()
    f.close()
    print(tts_fn)
    logging.info(f"Result wav.scp saved in {fn}")


if __name__ == "__main__":
    main()
