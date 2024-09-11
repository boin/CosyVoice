#!/usr/bin/env python3
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
import argparse

import onnxruntime
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from tqdm import tqdm


def main(args):
    utt2wav, utt2spk = {}, {}
    with open("{}/wav.scp".format(args.dir)) as f:
        for l in f:
            l = l.replace("\n", "").split()
            utt2wav[l[0]] = l[1]
    with open("{}/utt2spk".format(args.dir)) as f:
        for l in f:
            l = l.replace("\n", "").split()
            utt2spk[l[0]] = l[1]

    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    providers = ["CPUExecutionProvider", "CUDAExecutionProvider"]
    ort_session = onnxruntime.InferenceSession(
        args.onnx_path, sess_options=option, providers=providers
    )

    # gen spk2info.pt        #{'embedding': 'speech_token': 'speech_feat': }
    base_spkinfo = torch.load("./pretrained_models/CosyVoice-300M/spk2info.pt")

    utt2embedding, spk2embedding = {}, {}
    for utt in tqdm(utt2wav.keys()):
        audio, sample_rate = torchaudio.load(utt2wav[utt])
        if sample_rate != 16000:
            audio = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=16000
            )(audio)
        feat = kaldi.fbank(audio, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = (
            ort_session.run(
                None,
                {ort_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()},
            )[0]
            .flatten()
            .tolist()
        )
        utt2embedding[utt] = embedding #每句话都生成一个embedding
        spk = utt2spk[utt]
        if spk not in spk2embedding:
            spk2embedding[spk] = []
        spk2embedding[spk].append(embedding) #spk2embedding 的 spk 下有每句话的 embeddeding
    for k, v in spk2embedding.items():
        flat_embedding = torch.tensor(v).mean(dim=0, keepdim=True) #把每个spk的所有语料embedding都flat一遍
        spk2embedding[k] = (flat_embedding.tolist())[0]
        base_spkinfo[k] = {
            "embedding": flat_embedding,
            "speech_feat": [],
            "speech_token": [],
        }

    torch.save(utt2embedding, "{}/utt2embedding.pt".format(args.dir))    #每一条语料的embedding
    torch.save(spk2embedding, "{}/spk2embedding.pt".format(args.dir))    #每个spk的所有embedding数组

    torch.save(base_spkinfo, "{}/spk2info.pt".format(args.dir))          #每个spk存一个flat（混合）embeddeding


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)
    parser.add_argument("--onnx_path", type=str)
    args = parser.parse_args()
    main(args)
