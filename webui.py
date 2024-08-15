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
import logging
import os
import random
import shutil
import sys
from pathlib import Path

import ffmpeg
import gradio as gr
import librosa
import numpy as np
import torch
import torchaudio
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from gradio_log import Log

from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav

os.environ["CURL_CA_BUNDLE"] = ""
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["MODELSCOPE_CACHE"] = "./.cache/modelscope"
os.environ["TORCH_HOME"] = "./.cache/torch"  # 设置torch的缓存目录
os.environ["HF_HOME"] = "./.cache/huggingface"  # 设置transformer的缓存目录
os.environ["XDG_CACHE_HOME"] = "./.cache"
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append("{}/third_party/AcademiCodec".format(ROOT_DIR))
sys.path.append("{}/third_party/Matcha-TTS".format(ROOT_DIR))

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")

asr_model = AutoModel(
    model="iic/SenseVoiceSmall",
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cuda:0",
    disable_update=True,
)


def get_docker_logs():
    base_path = "./logs/ui.log"
    log_path = base_path
    return log_path


def speed_change(input_audio: np.ndarray, speed: float, sr: int):
    # 检查输入数据类型和声道数
    if input_audio.dtype != np.int16:
        raise ValueError("输入音频数据类型必须为 np.int16")

    # 转换为字节流
    raw_audio = input_audio.astype(np.int16).tobytes()

    # 设置 ffmpeg 输入流
    input_stream = ffmpeg.input(
        "pipe:", format="s16le", acodec="pcm_s16le", ar=str(sr), ac=1
    )

    # 变速处理
    output_stream = input_stream.filter("atempo", speed)

    # 输出流到管道
    out, _ = output_stream.output("pipe:", format="s16le", acodec="pcm_s16le").run(
        input=raw_audio, capture_stdout=True, capture_stderr=True
    )

    # 将管道输出解码为 NumPy 数组
    processed_audio = np.frombuffer(out, np.int16)

    return processed_audio


reference_wavs = ["请选择参考音频或者自己上传"]
for name in os.listdir("./参考音频/"):
    reference_wavs.append(name)

spk_new = ["无"]

for name in os.listdir("./voices/"):
    # print(name.replace(".pt",""))
    spk_new.append(name.replace(".pt", ""))


def refresh_choices():
    spk_new = ["无"]

    for name in os.listdir("./voices/"):
        # print(name.replace(".pt",""))
        spk_new.append(name.replace(".pt", ""))

    return {"choices": spk_new, "__type__": "update"}


def change_choices():
    reference_wavs = ["选择参考音频,或者自己上传"]

    for name in os.listdir("./参考音频/"):
        reference_wavs.append(name)

    return {"choices": reference_wavs, "__type__": "update"}


def change_wav(audio_path):
    text = audio_path.replace(".wav", "").replace(".mp3", "").replace(".WAV", "")

    return f"./参考音频/{audio_path}", text


def save_name(name):
    if not name or name == "":
        gr.Info("音色名称不能为空")
        return False

    shutil.copyfile("./output.pt", f"./voices/{name}.pt")
    gr.Info("音色保存成功,存放位置为voices目录")


def auto_asr(audio_path):
    res = asr_model.generate(
        input=audio_path,
        cache={},
        language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
        use_itn=True,
        batch_size_s=60,
        merge_vad=True,
        merge_length_s=15,
    )
    return rich_transcription_postprocess(res[0]["text"]).replace(" ", "")


def generate_seed():
    seed = random.randint(1, 100000000)
    return {"__type__": "update", "value": seed}


def set_all_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


max_val = 0.8


def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db, frame_length=win_length, hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(target_sr * 0.2))], dim=1)
    return speech


def change_llm_model(llm_path="", model_dir=""):
    """变更模型路径和地址
    Args:
        model_dir (_type_): 模型路径 默认 ./pretrained_modles/-300M
        llm_dir (_type_): LLM PT路径 默认 {model_dir}/llm.pt
    """
    global cosyvoice
    model_dir = model_dir if model_dir else args.model_dir
    spkinfo_dir = f"{model_dir}/spk2info.pt"
    spkinfo_path = ""
    if llm_path:
        spkinfo_path = Path(
            f"{os.path.dirname(llm_path)}/../train/temp1/spk2info.pt"
        ).resolve()
    else:
        llm_path = f"{model_dir}/llm.pt"
    spkinfo_dir = spkinfo_path if os.path.exists(spkinfo_path) else spkinfo_dir
    cosyvoice = CosyVoice(model_dir, llm_path, spkinfo_dir)
    return {"choices": cosyvoice.list_avaliable_spks(), "__type__": "update"}


inference_mode_list = ["预训练音色", "3s极速复刻", "跨语种复刻", "自然语言控制"]
instruct_dict = {
    "预训练音色": "1. 选择预训练音色\n2.点击生成音频按钮",
    "3s极速复刻": "1. 选择prompt音频文件，或录入prompt音频，若同时提供，优先选择prompt音频文件\n2. 输入prompt文本\n3.点击生成音频按钮",
    "跨语种复刻": "1. 选择prompt音频文件，或录入prompt音频，若同时提供，优先选择prompt音频文件\n2.点击生成音频按钮",
    "自然语言控制": "1. 输入instruct文本\n2.点击生成音频按钮",
}


def change_instruction(mode_checkbox_group):
    return instruct_dict[mode_checkbox_group]


def generate_audio(
    tts_text,
    mode_checkbox_group,
    sft_dropdown,
    prompt_text,
    prompt_wav_upload,
    prompt_wav_record,
    instruct_text,
    seed,
    speed_factor,
    new_dropdown,
    prompt_wav_select,
):
    if prompt_wav_select is not None:
        prompt_wav = prompt_wav_select
    elif prompt_wav_upload is not None:
        prompt_wav = prompt_wav_upload
    elif prompt_wav_record is not None:
        prompt_wav = prompt_wav_record
    else:
        prompt_wav = None
    # if instruct mode, please make sure that model is speech_tts/CosyVoice-300M-Instruct and not cross_lingual mode
    if mode_checkbox_group in ["自然语言控制"]:
        if cosyvoice.frontend.instruct is False:
            gr.Warning(
                "您正在使用自然语言控制模式, {}模型不支持此模式, 请使用speech_tts/CosyVoice-300M-Instruct模型".format(
                    args.model_dir
                )
            )
            return (target_sr, default_data)
        if instruct_text == "":
            gr.Warning("您正在使用自然语言控制模式, 请输入instruct文本")
            return (target_sr, default_data)
        if prompt_wav is not None or prompt_text != "":
            gr.Info("您正在使用自然语言控制模式, prompt音频/prompt文本会被忽略")
    # if cross_lingual mode, please make sure that model is speech_tts/CosyVoice-300M and tts_text prompt_text are different language
    if mode_checkbox_group in ["跨语种复刻"]:
        if cosyvoice.frontend.instruct is True:
            gr.Warning(
                "您正在使用跨语种复刻模式, {}模型不支持此模式, 请使用speech_tts/CosyVoice-300M模型".format(
                    args.model_dir
                )
            )
            return (target_sr, default_data)
        if instruct_text != "":
            gr.Info("您正在使用跨语种复刻模式, instruct文本会被忽略")
        if prompt_wav is None:
            gr.Warning("您正在使用跨语种复刻模式, 请提供prompt音频")
            return (target_sr, default_data)
        gr.Info("您正在使用跨语种复刻模式, 请确保合成文本和prompt文本为不同语言")
    # if in zero_shot cross_lingual, please make sure that prompt_text and prompt_wav meets requirements
    if mode_checkbox_group in ["3s极速复刻", "跨语种复刻"]:
        if prompt_wav is None:
            gr.Warning("prompt音频为空，您是否忘记输入prompt音频？")
            return (target_sr, default_data)
        if torchaudio.info(prompt_wav).sample_rate < prompt_sr:
            gr.Warning(
                "prompt音频采样率{}低于{}".format(
                    torchaudio.info(prompt_wav).sample_rate, prompt_sr
                )
            )
            return (target_sr, default_data)
    # sft mode only use sft_dropdown
    if mode_checkbox_group in ["预训练音色"]:
        if instruct_text != "" or prompt_wav is not None or prompt_text != "":
            gr.Info(
                "您正在使用预训练音色模式，prompt文本/prompt音频/instruct文本会被忽略！"
            )
    # zero_shot mode only use prompt_wav prompt text
    if mode_checkbox_group in ["3s极速复刻"]:
        if prompt_text == "":
            gr.Warning("prompt文本为空，您是否忘记输入prompt文本？")
            return (target_sr, default_data)
        if instruct_text != "":
            gr.Info("您正在使用3s极速复刻模式，预训练音色/instruct文本会被忽略！")

    if mode_checkbox_group == "预训练音色":
        logging.info("get sft inference request")
        set_all_random_seed(seed)
        output = cosyvoice.inference_sft(tts_text, sft_dropdown, new_dropdown)
    elif mode_checkbox_group == "3s极速复刻":
        logging.info("get zero_shot inference request")
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        set_all_random_seed(seed)
        output = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k)
    elif mode_checkbox_group == "跨语种复刻":
        logging.info("get cross_lingual inference request")
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        set_all_random_seed(seed)
        output = cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k)
    else:
        logging.info("get instruct inference request")
        set_all_random_seed(seed)
        output = cosyvoice.inference_instruct(
            tts_text, sft_dropdown, instruct_text, new_dropdown
        )

    if speed_factor != 1.0:
        try:
            numpy_array = output["tts_speech"].numpy()
            audio = (numpy_array * 32768).astype(np.int16)
            audio_data = speed_change(audio, speed=speed_factor, sr=int(target_sr))
        except Exception as e:
            print(f"Failed to change speed of audio: \n{e}")
    else:
        audio_data = output["tts_speech"].numpy().flatten()

    return (target_sr, audio_data)


def generate_audio_stream(
    tts_text,
    mode_checkbox_group,
    sft_dropdown,
    prompt_text,
    prompt_wav_upload,
    prompt_wav_record,
    instruct_text,
    seed,
    speed_factor,
    new_dropdown,
    prompt_wav_select,
):
    if mode_checkbox_group != "预训练音色":
        gr.Warning("流式推理只支持预训练音色推理")
        return (target_sr, default_data)
    #     logging.info('get sft inference request')
    #     set_all_random_seed(seed)
    #     # output = next(cosyvoice.inference_sft_stream(tts_text,sft_dropdown,new_dropdown))
    #     yield output

    spk_id = sft_dropdown

    if new_dropdown is not None:
        spk_id = "中文女"

    joblist = cosyvoice.frontend.text_normalize_stream(tts_text, split=True)

    for i in joblist:
        print(i)
        tts_speeches = []
        model_input = cosyvoice.frontend.frontend_sft(i, spk_id)
        if new_dropdown is not None:
            # 加载数据
            print(new_dropdown)
            print("读取pt")
            newspk = torch.load(f"{new_dropdown}")
            # with open(f'./voices/{new_dropdown}.py','r',encoding='utf-8') as f:
            #     newspk = f.read()
            #     newspk = eval(newspk)
            model_input["flow_embedding"] = newspk["flow_embedding"]
            model_input["llm_embedding"] = newspk["llm_embedding"]

            model_input["llm_prompt_speech_token"] = newspk["llm_prompt_speech_token"]
            model_input["llm_prompt_speech_token_len"] = newspk[
                "llm_prompt_speech_token_len"
            ]

            model_input["flow_prompt_speech_token"] = newspk["flow_prompt_speech_token"]
            model_input["flow_prompt_speech_token_len"] = newspk[
                "flow_prompt_speech_token_len"
            ]

            model_input["prompt_speech_feat_len"] = newspk["prompt_speech_feat_len"]
            model_input["prompt_speech_feat"] = newspk["prompt_speech_feat"]
            model_input["prompt_text"] = newspk["prompt_text"]
            model_input["prompt_text_len"] = newspk["prompt_text_len"]

        model_output = next(cosyvoice.model.inference_stream(**model_input))
        # print(model_input)
        tts_speeches.append(model_output["tts_speech"])
        output = torch.concat(tts_speeches, dim=1)

        if speed_factor != 1.0:
            try:
                numpy_array = output.numpy()
                audio = (numpy_array * 32768).astype(np.int16)
                audio_data = speed_change(audio, speed=speed_factor, sr=int(target_sr))
            except Exception as e:
                print(f"Failed to change speed of audio: \n{e}")
        else:
            audio_data = output.numpy().flatten()

        yield (target_sr, audio_data)


def main():
    with gr.Blocks() as demo:
        gr.Markdown(
            "### 代码库 [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) 预训练模型 [CosyVoice-300M](https://www.modelscope.cn/models/speech_tts/CosyVoice-300M) [CosyVoice-300M-Instruct](https://www.modelscope.cn/models/speech_tts/CosyVoice-300M-Instruct) [CosyVoice-300M-SFT](https://www.modelscope.cn/models/speech_tts/CosyVoice-300M-SFT)"
        )
        gr.Markdown("#### 请输入需要合成的文本，选择推理模式，并按照提示步骤进行操作")
        with gr.Row():
            mode_checkbox_group = gr.Radio(
                choices=inference_mode_list,
                label="选择推理模式",
                value=inference_mode_list[0],
            )
            instruction_text = gr.Text(
                label="操作步骤", value=instruct_dict[inference_mode_list[0]]
            )

        gr.Markdown(
            "### 预训练音色区（SFT推理） - 选择自定义底模中的讲述人（spk） 注意自定义底模也会影响3秒复刻"
        )
        with gr.Row():
            llm_model = gr.FileExplorer(
                glob="**/*.pt",
                ignore_glob="*._*",
                root_dir="./data",
                label="选择llm底模，不选为默认",
                interactive=True,
                file_count="single",
            )
            sft_dropdown = gr.Dropdown(
                choices=sft_spk, label="选择预训练音色", value=sft_spk[0]
            )
            with gr.Column():
                seed_button = gr.Button(value="\U0001f3b2")
                seed = gr.Number(value=0, label="随机推理种子(影响全局推理)")
        gr.Markdown("### 三秒复刻区（Zero-Shot推理） - 参考音频Prompt选择")
        with gr.Row():
            new_dropdown = gr.FileExplorer(
                glob="**/*.pt",
                ignore_glob="*._*",
                root_dir="./voices/",
                label="选择自定义音色",
                interactive=True,
                file_count="single",
            )
            prompt_wav_select = gr.FileExplorer(
                glob="**/*.wav",
                ignore_glob="*._*",
                root_dir="./ttd_lib/",
                label="从音色库选择prompt",
                interactive=True,
                file_count="single",
            )
            prompt_wav_upload = gr.Audio(
                sources="upload",
                type="filepath",
                label="上传prompt，采样率≥16khz",
            )
            prompt_wav_record = gr.Audio(
                sources="microphone", type="filepath", label="录制prompt"
            )
        tts_text = gr.Textbox(
            label="输入合成文本",
            lines=1,
            value="我是通义实验室语音团队全新推出的生成式语音大模型，提供舒适自然的语音合成能力。",
        )
        speed_factor = gr.Slider(
            minimum=0.25,
            maximum=4,
            step=0.05,
            label="语速调节",
            value=1.0,
            interactive=True,
        )
        prompt_text = gr.Textbox(
            label="输入prompt文本",
            lines=1,
            placeholder="请输入prompt文本，需与prompt音频内容一致，暂时不支持自动识别...",
            value="",
        )
        instruct_text = gr.Textbox(
            label="输入instruct文本",
            lines=1,
            placeholder="请输入instruct文本.",
            value="",
        )

        new_name = gr.Textbox(
            label="输入新的音色名称", lines=1, placeholder="输入新的音色名称.", value=""
        )

        save_button = gr.Button("保存刚刚推理的zero-shot音色")

        save_button.click(save_name, inputs=[new_name])

        llm_model.change(change_llm_model, inputs=[llm_model], outputs=[sft_dropdown])

        prompt_wav_select.change(
            fn=auto_asr,
            inputs=[prompt_wav_select],
            outputs=[prompt_wav_upload, prompt_text],
        )
        prompt_wav_upload.change(
            fn=auto_asr, inputs=[prompt_wav_upload], outputs=[prompt_text]
        )
        prompt_wav_record.change(
            fn=auto_asr, inputs=[prompt_wav_record], outputs=[prompt_text]
        )

        generate_button = gr.Button("生成音频")
        generate_button_stream = gr.Button("流式生成")

        # audio_output = gr.Audio(label="合成音频")
        audio_output = gr.Audio(
            label="合成音频",
            value=None,
            streaming=True,
            autoplay=True,  # disable auto play for Windows, due to https://developer.chrome.com/blog/autoplay#webaudio
            interactive=False,
            show_label=True,
            show_download_button=True,
        )

        Log(
            get_docker_logs(),
            dark=True,
            xterm_font_size=12,
            render=bool(get_docker_logs()),
        )

        # result2 = gr.Textbox(label="翻译结果(会在项目目录生成two.srt/two.srt is generated in the current directory)")
        # audio_output
        seed_button.click(generate_seed, inputs=[], outputs=seed)
        generate_button.click(
            generate_audio,
            inputs=[
                tts_text,
                mode_checkbox_group,
                sft_dropdown,
                prompt_text,
                prompt_wav_upload,
                prompt_wav_record,
                instruct_text,
                seed,
                speed_factor,
                new_dropdown,
                prompt_wav_select,
            ],
            outputs=[audio_output],
        )

        generate_button_stream.click(
            generate_audio_stream,
            inputs=[
                tts_text,
                mode_checkbox_group,
                sft_dropdown,
                prompt_text,
                prompt_wav_upload,
                prompt_wav_record,
                instruct_text,
                seed,
                speed_factor,
                new_dropdown,
                prompt_wav_select,
            ],
            outputs=[audio_output],
        )
        mode_checkbox_group.change(
            fn=change_instruction,
            inputs=[mode_checkbox_group],
            outputs=[instruction_text],
        )
    demo.queue(max_size=4, default_concurrency_limit=2)
    demo.launch(server_name="0.0.0.0", server_port=args.port, inbrowser=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--model_dir",
        type=str,
        default="speech_tts/CosyVoice-300M",
        help="local path or modelscope repo id",
    )
    args = parser.parse_args()
    cosyvoice = CosyVoice(args.model_dir)
    sft_spk = cosyvoice.list_avaliable_spks()
    prompt_sr, target_sr = 16000, 22050
    default_data = np.zeros(target_sr)
    main()
