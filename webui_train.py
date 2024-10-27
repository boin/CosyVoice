import json
import logging
import os
import random
import subprocess
import warnings
from pathlib import Path

import gradio as gr
import psutil
import torch
from gradio_log import Log

import tools.auto_ttd as ttd

warnings.simplefilter(action="ignore", category=FutureWarning)

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

device = torch.cuda.is_available() and torch.cuda.get_device_name(0) or "CPU"

DATA_ROOT = os.environ["DATA_ROOT"] if "DATA_ROOT" in os.environ else "./data"
OUTPUT_ROOT = (
    os.environ["OUTPUT_ROOT"] if "OUTPUT_ROOT" in os.environ else f"{DATA_ROOT}/outputs"
)

def data_model_path(model_dir, project_name, actor):
    # {DATA_ROOT=data/cosy}/models/{PRJ_NAME}/{actor}/{path=output}
    return Path(DATA_ROOT) / "models" / project_name / actor / model_dir


def data_output_path(base_path, project_name):
    # {OUTPUT_ROOT}/{project_name}/cosy/{base_path}
    return Path(OUTPUT_ROOT) / project_name / "cosy" / base_path


def log(data):
    print(data)
    return data


def get_docker_logs():
    base_path = "./logs/train.log"
    log_path = base_path
    return log_path


"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
"""


def generate_seed():
    seed = random.randint(1, 100000000)
    return {"__type__": "update", "value": seed}


def refresh_lib_projects(project_input_dir):
    return gr.Dropdown(choices=ttd.load_lib_projects())


def refresh_lib_actors(project_input_dir):
    list = ttd.load_lib_prj_actors(project_input_dir)
    # print(list)
    return {"__type__": "update", "choices": list, "value": list[0]}


def refresh_voice(project_input_dir, output_path, actor):
    content = (
        data_model_path(output_path, project_input_dir, actor)
        / "train"
        / "temp1"
        / "utt2spk"
    ).read_text()
    voices = []
    for item in content.split("\n"):
        if item:
            [voice, spkr] = item.split(" ")
            voices.append(f"{spkr} - {voice}")
    return gr.Dropdown(choices=voices)


def load_refrence_wav(refrence_name, project_input_dir, actor):
    [spkr, voice] = refrence_name.split(" - ")
    path = ttd.load_refrence_wav(project_input_dir, actor, voice)
    return path


def load_mix_ref(pt_dir):
    if not pt_dir:
        return ""
    content = (
        Path(f"{os.path.dirname(pt_dir)}/../train/temp1/utt2spk")
    ).read_text() or ""
    voices = []
    for item in content.split("\n"):
        if item:
            [voice, spkr] = item.split(" ")
            voices.append(f"{spkr} - {voice}")
    return gr.Dropdown(choices=voices)


def preprocess(project_input_dir, output_path, actor, split_ratio, force_flag):
    # check src first
    if ttd.check_proj_actor_wavs(project_input_dir, actor) < 1:
        raise gr.Error("该角色没有可训练的文件")

    for state, input_path in zip(
        ["train", "val"],
        [
            data_model_path("train", project_input_dir, actor),
            data_model_path("val", project_input_dir, actor),
        ],
    ):
        temp1 = data_model_path(output_path, project_input_dir, actor) / state / "temp1"
        temp2 = data_model_path(output_path, project_input_dir, actor) / state / "temp2"
        try:
            temp1.mkdir(parents=True)
            temp2.mkdir(parents=True)
        except Exception:
            pass

        logging.info(
            f'processing state {state}, with input_path:  {project_input_dir},  {input_path},  temp_path:  {temp1},  {temp2}, "src_split_ratio:" {split_ratio}, "force_flag" {force_flag}'
        )

        # subprocess.run([r'.\py311\python.exe', 'local/prepare_data.py',
        out = subprocess.run(
            [
                r"python3",
                "local/prepare_data.py",
                "--src_dir",
                str(input_path),
                "--des_dir",
                str(temp1),
                "--actor",
                actor,
                "--init_split_ratio",
                str(split_ratio),
                "--force_flag",
                str(force_flag),  # str True / False
            ],
            # capture_output= True
            env=dict(
                os.environ,
                PYTHONIOENCODING="UTF-8",
                PYTHONPATH="./",
            ),
        )
        if out.returncode == 0:
            log(f"{state} 数据初始化完成")
        else:
            return log(f"{state} 数据初始化出错 {out}")

        # subprocess.run([r'.\py311\python.exe', 'tools/extract_embedding.py',
        out = subprocess.run(
            [
                r"python3",
                "tools/extract_embedding.py",
                "--dir",
                str(temp1),
                "--onnx_path",
                f"{DATA_ROOT}/pretrained_models/CosyVoice-300M/campplus.onnx",
            ],
            # capture_output= True
        )
        if out.returncode == 0:
            log(f"{state} 导出 embeddeding 完成")
        else:
            return log(f"{state} 导出 embeddeding 出错 {out}")

        # subprocess.run([r'.\py311\python.exe', 'tools/extract_speech_token.py',
        out = subprocess.run(
            [
                r"python3",
                "tools/extract_speech_token.py",
                "--dir",
                str(temp1),
                "--onnx_path",
                f"{DATA_ROOT}/pretrained_models/CosyVoice-300M/speech_tokenizer_v1.onnx",
            ],
            # capture_output= True
        )
        if out.returncode == 0:
            log(f"{state} 导出分词 token 完成")
        else:
            return log(f"{state} 导出分词 token 出错 {out}")

        # subprocess.run([r'.\py311\python.exe', 'tools/make_parquet_list.py',
        out = subprocess.run(
            [
                r"python3",
                "tools/make_parquet_list.py",
                "--num_utts_per_parquet",
                "100",
                "--num_processes",
                "1",
                "--src_dir",
                str(temp1),
                "--des_dir",
                str(temp2),
            ],
            # capture_output= True
        )
        if out.returncode == 0:
            log(f"{state} 导出 parquet 列表完成")
        else:
            return log(f"{state} 导出 parquet 列表出错 {out}")

    return log("预处理全部完成，可以开始训练")


def train(project_input_dir, output_path, actor, pre_model_path, thread_num, max_epoch):
    output_path = data_model_path(output_path, project_input_dir, actor)
    train_list = output_path / "train/temp2/data.list"
    val_list = output_path / "val/temp2/data.list"
    model_dir = output_path / "pts"
    model_dir.mkdir(exist_ok=True, parents=True)

    out = subprocess.run(
        [
            r"torchrun",
            "--nnodes",
            "1",
            "--nproc_per_node",
            "1",
            "--rdzv_id",
            "1986",
            "--rdzv_backend",
            "c10d",
            "--rdzv_endpoint",
            "localhost:0",
            "cosyvoice/bin/train.py",
            "--train_engine",
            "torch_ddp",
            "--config",
            "conf/cosyvoice.yaml",
            "--max_epoch",
            str(max_epoch),
            "--train_data",
            train_list,
            "--cv_data",
            val_list,
            "--model",
            "llm",
            "--checkpoint",
            os.path.join(pre_model_path, "llm.pt"),
            "--model_dir",
            str(model_dir),
            "--tensorboard_dir",
            str(model_dir),
            "--ddp.dist_backend",
            "nccl",
            "--num_workers",
            str(thread_num),
            "--prefetch",
            "200",
            "--pin_memory",
            "--deepspeed_config",
            "./conf/ds_stage2.json",
            "--deepspeed.save_states",
            "model+optimizer",
        ],
        env=dict(
            os.environ,
            PYTHONIOENCODING="UTF-8",
            PYTHONPATH="./:./third_party/Matcha-TTS:./third_party/AcademiCodec",
        ),
    )
    if out.returncode == 0:
        return log("训练完成")
    else:
        return log(f"训练出错啦 {out}")


def inference(
    mode,
    project_input_dir,
    model_output_path,
    epoch,
    pre_model_path,
    text,
    actor,
    voice,
    seed,
    spk_mix: str | None = None,
    mix_file: str | None = None,
    w1: str | None = None,
    w2: str | None = None,
):
    if not text:
        raise gr.Error("no text.")
    if not voice:
        raise gr.Error("no voice.")
    output_path = data_output_path(actor, project_input_dir)
    model_path = data_model_path(model_output_path, project_input_dir, actor)
    train_list = os.path.join(model_path, "train", "temp2", "data.list")
    utt2data_list = Path(train_list).with_name("utt2data.list")
    llm_model = (
        epoch
        and os.path.join(model_path, "pts", f"epoch_{epoch}_whole.pt")
        or os.path.join(pre_model_path, "llm.pt")
    )
    flow_model = os.path.join(pre_model_path, "flow.pt")
    hifigan_model = os.path.join(pre_model_path, "hift.pt")
    res_dir = Path(output_path)
    res_dir.mkdir(exist_ok=True, parents=True)
    voice = voice.split(" - ")[1]  # spkr1 - voice1 => voice1
    if not voice:
        raise "empty voice."
    spk_mix = spk_mix and spk_mix.split(" - ")[1] or ""
    mix_file = mix_file and os.path.dirname(mix_file) or ""
    mix_rate = f"{w1}-{w2}"
    if spk_mix:
        mix_file = os.path.realpath(f"{mix_file}/../train/temp1/utt2embedding.pt")
    json_path = str(Path(res_dir) / "tts_text.json")
    with open(json_path, "wt", encoding="utf-8") as f:
        json.dump({voice: [text]}, f)

    logging.info(
        f"call cosyvoice/bin/inference.py {mode} => {voice} says: {text} with r_seed {seed} llm {llm_model}"
    )
    # subprocess.run([r'.\pyr11\python.exe', 'cosyvoice/bin/inference.py',
    cmd = [
        r"python3",
        "cosyvoice/bin/inference.py",
        "--mode",
        mode,
        "--gpu",
        "0",
        "--config",
        "conf/cosyvoice.yaml",
        "--prompt_data",
        train_list,
        "--prompt_utt2data",
        str(utt2data_list),
        "--tts_text",
        json_path,
        "--llm_model",
        llm_model,
        "--flow_model",
        flow_model,
        "--hifigan_model",
        hifigan_model,
        "--result_dir",
        str(res_dir),
        "--rseed",
        str(seed),
        "--mix",
        spk_mix,
        "--mix_pt",
        mix_file,
        "--mix_rate",
        mix_rate,
    ]
    subprocess.run(
        cmd,
        env=dict(
            os.environ,
            PYTHONIOENCODING="UTF-8",
            PYTHONPATH="./:./third_party/Matcha-TTS:./third_party/AcademiCodec",
        ),
    )
    output_path = str(Path(res_dir) / f"{voice}_0.wav")
    return output_path


def stop_training(proc_name="torchrun"):
    num = 0
    for proc in psutil.process_iter():
        if proc.name() == proc_name:
            try:
                proc.kill()
                num = num + 1
            except OSError:
                pass
    return f"停止了{num}个进程"


with gr.Blocks() as demo:
    with gr.Row():
        project_input_dir = gr.Dropdown(
            choices=ttd.load_lib_projects(),
            value=ttd.load_lib_projects()[0],
            container=True,
            label="项目根目录",
            info="项目根目录，会在TeamSpace/TTD-Space/项目/ 目录下寻找",
        )
        actor = gr.Dropdown(
            choices=ttd.load_lib_prj_actors(project_input_dir.value),
            label="欲训练角色",
            info="现在是一个一个角色单独训练",
            interactive=True,
        )
        gr.Button(
            value="刷新项目角色",
        ).click(refresh_lib_actors, [project_input_dir], [actor])
        gr.Text(
            label="当前显卡",
            value=device,
            info="显示当前使用的GPU型号，如果没有检测到则显示CPU",
        )
    with gr.Accordion("高级选项，一般不用管", open=False):
        output_dir = gr.Text(
            label="模型输出文件夹",
            value="output",
            info="预处理与训练最终会输出在项目根目录的本文件夹下，没有会自动新建，一般不用改",
        )
        pretrained_model_path = gr.Text(
            f"{DATA_ROOT}/pretrained_models/CosyVoice-300M",
            label="预训练模型文件夹",
            info="可选 300M-SFT/330M-Insturct 一般不用改",
        )
    with gr.Tab("训练"):
        with gr.Row():
            split_ratio = gr.Radio(
                choices=[
                    ("不分配", -1),
                    ("1:1", 50),
                    ("6:4", 60),
                    ("7:3", 70),
                    ("按序号分配", 0),
                ],
                label="预料训练集/验证集分配比例。选择不分配则全部分配训练集合",
                info="注意：如果选择了“按照序号分配语料”，在同一音色设置多个序号不同的参考音。（如：XX_开心愤怒_001_YY, XX_开心愤怒_002_YY）奇数序号会分配给训练集，偶数分配测试集",
                value=-1,
            )
            re_init = gr.Checkbox(
                value=True,
                interactive=False,
                label="重新分配语料",
                info="如果语料库中此角色的语料有更新，或者调整了分配比例，那么就需要勾选此选项重新预处理",
            )
        preprocess_btn = gr.Button(
            "开始预处理（提取训练集音色数据，如果只是要新增推理的音色，只点这个就行了）",
            variant="primary",
        )
        with gr.Row():
            max_epoch = gr.Number(
                value=1,
                interactive=True,
                precision=0,
                label="训练总轮次",
                info="1-1000",
            )
            thread_num = gr.Number(
                value=1,
                interactive=True,
                precision=0,
                label="训练线程数量",
                info="每次+1康康，爆显存杀手",
            )
        with gr.Row():
            train_btn = gr.Button(
                "开始训练（如果要把训练数据来影响底层模型，可以用训练的方式）",
                variant="primary",
                scale=8,
            )
            stop_btn = gr.Button("停止训练", variant="stop", scale=2)
        status = gr.Text(label="状态")
    with gr.Tab("推理"):
        with gr.Row():
            mode = gr.Dropdown(
                choices=["sft", "zero_shot"],
                label="推理模式",
                value="zero_shot",
                info="SFT模型（SFT）和3秒复刻模型（zero-shot）",
                scale=1,
            )
            epoch = gr.Text(
                interactive=True,
                label="模型轮次ID",
                info="使用模型输出文件夹中训练第？轮次的模型",
                scale=1,
            )
            with gr.Column():
                seed = gr.Number(value=0, label="随机推理种子(影响全局推理)")
                seed_button = gr.Button(value="\U0001f3b2")
        with gr.Row():
            with gr.Column(scale=2):
                refresh = gr.Button("刷新音色列表", variant="primary")
                voices = gr.Dropdown(
                    label="首选音色列表",
                    info="根据训练集的数据，在预处理中生成，点右侧刷新",
                )
            preview = gr.Audio(
                label="首选参考音预览",
                show_download_button=False,
                show_share_button=False,
                sources=[],
                scale=2,
            )
            """ 暂时关闭音色融合
            with gr.Column(scale=2):
                mix_file = gr.FileExplorer(
                    label="加载融合音色底模",
                    file_count="single",
                    root_dir="./data",
                    glob="*/*.pt",
                    interactive=False,
                )
                spk_mix = gr.Dropdown(
                    label="选择融合音色",
                    info="如果需要融合某个音色，可以在这里选择",
                )
            preview2 = gr.Audio(
                label="融合参考音预览",
                show_download_button=False,
                show_share_button=False,
                sources=[],
                scale=2,
            )
        with gr.Row(visible=False):
            w1 = gr.Number(value=0.5, label="首选音色权重", interactive=True)
            w2 = gr.Number(value=0.5, label="融合音色权重", interactive=True)
        """
        text = gr.Text(label="输入文字")
        inference_btn = gr.Button("开始推理", variant="primary")
        out_audio = gr.Audio(label="音频输出")
    Log(
        get_docker_logs(), dark=True, xterm_font_size=12, render=bool(get_docker_logs())
    )
    project_input_dir.change(
        refresh_lib_actors, inputs=[project_input_dir], outputs=[actor]
    )
    voices.change(
        load_refrence_wav, inputs=[voices, project_input_dir, actor], outputs=preview
    )
    # spk_mix.change(load_refrence_wav, [spk_mix, mix_file, actor], preview2)

    seed_button.click(generate_seed, inputs=[], outputs=seed)
    # mix_file.change(load_mix_ref, inputs=[mix_file], outputs=[spk_mix])

    preprocess_btn.click(
        preprocess,
        inputs=[project_input_dir, output_dir, actor, split_ratio, re_init],
        outputs=status,
    )
    train_btn.click(
        train,
        inputs=[
            project_input_dir,
            output_dir,
            actor,
            pretrained_model_path,
            thread_num,
            max_epoch,
        ],
        outputs=status,
    )
    inference_btn.click(
        inference,
        inputs=[
            mode,
            project_input_dir,
            output_dir,
            epoch,
            pretrained_model_path,
            text,
            actor,
            voices,
            seed,
            # spk_mix,
            # mix_file,
            # w1,
            # w2,
        ],
        outputs=out_audio,
    )
    refresh.click(
        refresh_voice, inputs=[project_input_dir, output_dir, actor], outputs=voices
    )
    stop_btn.click(stop_training, outputs=[status])

demo.launch(server_name="0.0.0.0", server_port=9883, inbrowser=False)
