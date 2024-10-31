import glob
import json
import logging
import os
import random
import subprocess
from hashlib import md5
from pathlib import Path
from zipfile import ZipFile

import gradio as gr
import openpyxl
import torch

from tools.auto_ttd import load_actor, load_projects, load_refrence
from tools.emo_dialog_parser import dialog_parser
from tools.vc import (
    load_keytone_from_actor,
    load_vc_actor,
    load_vc_actor_ref,
    request_vc,
    vc_output_path,
)

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DATA_ROOT = os.environ["DATA_ROOT"] if "DATA_ROOT" in os.environ else "./data"
OUTPUT_ROOT = (
    os.environ["OUTPUT_ROOT"] if "OUTPUT_ROOT" in os.environ else f"{DATA_ROOT}/outputs"
)
# global vars
wavs = {}
vcs = {}
projects = load_projects()
hash = ""
chap_name = ""
device = torch.cuda.is_available() and torch.cuda.get_device_name(0) or "CPU"


def data_model_path(model_dir, project_name, actor):
    # {DATA_ROOT=data/cosy}/models/{PRJ_NAME}/{actor}/{path=output}
    return Path(DATA_ROOT) / "models" / project_name / actor / model_dir


def data_output_path(base_path, project_name):
    return Path(DATA_ROOT) / "outputs" / project_name / "cosy" / base_path


def download_all_wavs(wavs, hash=hash):
    logging.debug(f"download all calle with:{wavs}, {hash}, {chap_name}")
    fn = f"/tmp/gradio/{chap_name}.zip"
    with ZipFile(fn, "w") as zip:
        for f in wavs:
            if not Path(str(wavs[f])).is_file():
                continue
            zip.write(wavs[f], f"{Path(wavs[f]).stem}.wav")
    zip.close()
    logging.info(f"zipfile {fn} writed. size {(Path(fn).lstat()).st_size}")
    return gr.DownloadButton(label="点击下载", value=fn, variant="stop")


def play_ref_audio(project, actor, voice):
    audio_path = (
        Path(DATA_ROOT)
        / "models"
        / project
        / actor
        / "train"
        / f'{voice.split(" ")[0]}.wav'
    )
    return audio_path


def play_vc_ref_audio(project, actor):
    path = load_vc_actor_ref(project, actor)
    return path


def upload_textbook(text_url: str, project: str):
    global hash, wavs, vcs, chap_name
    hash = md5(Path(text_url).read_bytes()).hexdigest()  # save file hash for future use
    chap_name = Path(text_url).stem
    workbook = openpyxl.load_workbook(text_url, read_only=True)
    rows = [row for row in workbook.active.rows]
    if not len(rows) > 0:
        raise gr.Error("Excel文档为空！")
    lines = list(dialog_parser(rows))
    # all_wavs = load_wav_cache(project, hash)  # merge old wavs with new hash values
    # all_vcs = load_wav_cache(project, hash, "vc")  # also vc
    # reset wavs and vc on txt change
    wavs = {}
    vcs = {}
    # 合并所有 WAV 和 VC
    # wavs.update(all_wavs)
    # vcs.update(all_vcs)
    return lines


def start_inference(
    project_name, model_dir, actor, text, voice, id: str, r_seed, g_seed
):
    """开始推理
    Args:
        project_name (str): 项目名称
        model_dir (str): 模型路径
        text (str): 推理文本
        voice (str): 选定音色
        id (str): uniqid
        r_seed (str): 随机种子
        g_seed (str): 全局随机种子
    """
    global wavs
    # 240915_有声书_殓葬禁忌  旁白_脸红_002_507828_并且在峨眉危难之际自动出现，化解危机。  Says: 我，是一名殓葬师！ {}
    if not text:
        raise gr.Error("no text.")
    if not voice:
        raise gr.Error("no voice.")
    mode = "zero_shot"
    epoch = 0
    pre_model_path = Path(f"{DATA_ROOT}/pretrained_models/CosyVoice-300M")
    model_path = data_model_path(model_dir, project_name, actor)
    train_list = model_path / "train" / "temp2" / "data.list"
    utt2data_list = Path(train_list).with_name("utt2data.list")
    llm_model = model_path / "models" / f"epoch_{epoch}_whole.pt"
    llm_model = pre_model_path / "llm.pt"  # just using default pt
    flow_model = pre_model_path / "flow.pt"
    hifigan_model = pre_model_path / "hift.pt"
    res_dir = data_output_path(hash, project_name)
    res_dir.mkdir(exist_ok=True, parents=True)
    json_path = str(Path(res_dir) / f"{id}.json")
    # 旁白_001_ASR.wav
    out_name = f'{actor.split("_")[1]}_{id}_{text[:30]}'
    with open(json_path, "wt", encoding="utf-8") as f:
        json.dump({voice: [text]}, f)
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
        str(r_seed or g_seed),
        "--file_name",
        out_name,
    ]
    logging.info(
        f"call inference {project_name} => actor: {actor} voice: {voice} says: {text} with cmd {cmd}"
    )
    subprocess.run(
        cmd,
        env=dict(
            os.environ,
            PYTHONIOENCODING="UTF-8",
            PYTHONPATH="./:./third_party/Matcha-TTS:./third_party/AcademiCodec",
        ),
    )
    output_file = str(Path(res_dir) / f"{out_name}.wav")
    wavs[id] = output_file
    return output_file


def start_vc(project: str, actor: str, audio_path: str, id, text, tone_key=0):
    global vcs
    if not audio_path:
        gr.Error("没有已生成的推理音频，无法VC")
    res_dir = vc_output_path(id.rpartition("-")[0], project)
    res_dir.mkdir(exist_ok=True, parents=True)
    # 旁白_001_ASR.wav
    out_name = f'{actor.split("_")[1]}_{id}_{text[:30]}'
    output_path = str(Path(res_dir) / f"{out_name}.wav")
    status, message = request_vc(project, actor, audio_path, output_path, tone_key)
    if status == 1:
        gr.Error(f"VC 失败：{message}")
    vcs[id] = output_path
    return output_path


with gr.Blocks(fill_width=True) as demo:
    # wavs.value = upload_textbook(
    #     "data/Ch001_天命，将至_QC.xlsx", projects[-1], wavs.value
    # )
    lines = gr.State([])
    lines.change(lambda x: logging.debug(f"wavs changed.{x}\n"), inputs=lines)
    with gr.Row():
        with gr.Column(variant="panel"):
            project = gr.Dropdown(
                choices=projects,
                value=projects[-1] if len(projects) > 0 else "",
                label="项目名称",
            )
            gr.Button("刷新").click(
                lambda: {"__type__": "update", "choices": load_projects()},
                inputs=[],
                outputs=[project],
            )
        with gr.Column(variant="panel"):
            seed = gr.Number(value=0, label="全局推理种子", interactive=True)
            with gr.Row():
                gr.Button("\U0001f3b2").click(
                    lambda: random.randint(1, 1e8), outputs=seed
                )
                gr.Button("清除").click(lambda: None, outputs=seed)
        with gr.Column(variant="panel"):
            output_dir = gr.Textbox("output", label="输出路径")
            gr.Text(container=False, value="当前显卡: " + device)
        upload = gr.File(label="上传台词本", file_types=["text", ".xlsx"])
        upload.upload(upload_textbook, inputs=[upload, project], outputs=[lines])
        upload.clear(lambda: [], outputs=lines)
    with gr.Row():
        gen_all = gr.Button("一键推理", variant="primary")
        gen_all.click(
            None,
            js="()=>{[...document.querySelectorAll('.gen-btn')].reduce((p,e)=>p.then(()=>(!e.id&&e.click(),new Promise(r=>setTimeout(r, 50)))),Promise.resolve());}",
        )
        re_gen_all = gr.Button("全部重推理", variant="primary")
        re_gen_all.click(
            None,
            js="()=>{[...document.querySelectorAll('.gen-btn')].reduce((p,e)=>p.then(()=>(e.click(),new Promise(r=>setTimeout(r, 50)))),Promise.resolve());}",
        )
        dl_all = gr.DownloadButton("打包下载推理")
        dl_all.click(lambda: download_all_wavs(wavs, hash), outputs=[dl_all])
        vc_all = gr.Button("一键VC", variant="primary")
        vc_all.click(
            None,
            js="()=>{[...document.querySelectorAll('.vc-btn')].reduce((p,e)=>p.then(()=>(!e.id&&e.click(),new Promise(r=>setTimeout(r, 50)))),Promise.resolve());}",
        )
        re_vc_all = gr.Button("全部重VC", variant="primary")
        re_vc_all.click(
            None,
            js="()=>{[...document.querySelectorAll('.vc-btn')].reduce((p,e)=>p.then(()=>(e.click(),new Promise(r=>setTimeout(r, 50)))),Promise.resolve());}",
        )
        dl_vc_all = gr.DownloadButton("打包下载VC")
        dl_vc_all.click(lambda: download_all_wavs(vcs, hash), outputs=[dl_vc_all])

    @gr.render(inputs=[lines, project])
    def render_lines(_lines, _project):
        logging.debug("re-render wavs version with active project:", _project, wavs)
        task_list = _lines
        # print(hash, task_list[0], _wavs, "\n")
        for task in task_list:
            # print(task)
            idx = f'{task["id"]:03}'  # 003 087
            wav_url = idx in wavs.keys() and wavs[idx] or None
            vc_url = idx in vcs.keys() and vcs[idx] or None
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        id = gr.Text(
                            idx,
                            visible=False,
                        )
                        gr.Text(
                            f'{task["id"]} {task["actor"]}',
                            label="metadata",
                            show_label=False,
                            container=False,
                            scale=1,
                            min_width=100,
                        )

                        text = gr.Textbox(
                            task["text"],
                            show_label=False,
                            container=False,
                            scale=6,
                        )
                        # seed area
                        _seed = gr.Textbox(
                            None,
                            container=False,
                            show_label=False,
                            min_width=80,
                            interactive=True,
                            scale=0,
                        )
                        gr.Button("\U0001f3b2", min_width=1, scale=0).click(
                            lambda: random.randint(1, 1e8), outputs=_seed
                        )
                        gr.Button("X", min_width=2, scale=0).click(
                            lambda: None, outputs=_seed
                        )
                    with gr.Row():
                        gr.Text(
                            f'{task["emo_1"]} {task["emo_2"]} V: {float(task["V"])*100:.1f} A: {float(task["A"])*100:.1f} D: {float(task["D"])*100:.1f}',
                            show_label=False,
                            container=False,
                            scale=2,
                        )
                        actors = load_actor(task["actor"], _project)
                        # print("actors:", actors)
                        actor = gr.Dropdown(
                            choices=actors,
                            value=actors[0],
                            show_label=False,
                            container=False,
                            scale=2,
                        )
                        refrences = load_refrence(
                            _project,
                            actors[0],  # use parsed actorname than original
                            [task["V"], task["A"], task["D"]],
                            emo_kw=f'{task["emo_1"]}{task["emo_2"]}',  # for kwmatch
                            text=task["text"],  # for asr match
                        )
                        ref_ctl = gr.Dropdown(
                            choices=refrences,
                            value=refrences[0],
                            show_label=False,
                            container=False,
                            scale=6,
                        )
                        gen_btn = gr.Button(
                            "生成",
                            scale=0,
                            variant="primary",
                            elem_classes=["gen-btn"],
                            elem_id=idx if wav_url else "",
                        )
                        preview_btn = gr.Button(
                            value="预览输出",
                            scale=0,
                        )
                    with gr.Row():
                        vc_actors = load_vc_actor(_project, task["actor"])
                        gr.Text(
                            "VC音色选择 [声调，模型] ",
                            show_label=False,
                            container=False,
                            scale=2,
                            min_width=100,
                        )
                        tone_key = gr.Dropdown(
                            choices=[
                                f"{i:+d}" if i != 0 else "0" for i in range(5, -6, -1)
                            ],
                            value=load_keytone_from_actor(vc_actors[0]),
                            show_label=False,
                            container=False,
                        )
                        vc_actor = gr.Dropdown(
                            show_label=False,
                            container=False,
                            choices=vc_actors,
                            value=vc_actors[0],
                            scale=8,
                        )
                        vc_btn = gr.Button(
                            "生成VC",
                            scale=0,
                            variant="primary",
                            elem_classes=["vc-btn"],
                            elem_id=idx if vc_url else "",
                        )
                        preview_vc_btn = gr.Button(
                            value="预览VC",
                            scale=0,
                        )

                with gr.Column(scale=0):
                    preview_audio = gr.Audio(
                        # container=False,
                        label="预览",
                        show_download_button=True,
                        show_share_button=False,
                        sources=[],
                        scale=0,
                        value=wav_url,
                        type="filepath",
                    )

                # to preserve idx variable
                def reset_audio(id=idx):
                    return id in wavs and gr.Audio(wavs[id]) or None

                gen_btn.click(
                    start_inference,
                    inputs=[
                        project,
                        output_dir,
                        actor,
                        text,
                        ref_ctl,
                        id,
                        _seed,
                        seed,
                    ],
                    outputs=[preview_audio],
                )
                vc_btn.click(reset_audio, outputs=preview_audio).then(
                    start_vc,
                    inputs=[project, vc_actor, preview_audio, id, text, tone_key],
                    outputs=[preview_audio],
                )
                preview_vc_btn.click(
                    lambda id=idx: gr.Audio(vcs[id]) if id in vcs else None,
                    outputs=preview_audio,
                )
                vc_actor.change(
                    load_keytone_from_actor, inputs=vc_actor, outputs=tone_key
                )
                gr.on(
                    triggers=[ref_ctl.focus, ref_ctl.select],
                    fn=play_ref_audio,
                    inputs=[project, actor, ref_ctl],
                    outputs=[preview_audio],
                )
                gr.on(
                    triggers=[vc_actor.focus, vc_actor.select],
                    fn=play_vc_ref_audio,
                    inputs=[project, vc_actor],
                    outputs=[preview_audio],
                )
                gr.on(
                    triggers=[preview_audio.clear, preview_btn.click],
                    fn=reset_audio,
                    outputs=preview_audio,
                )
            with gr.Row():
                gr.HTML("<hr/>")


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=9884)
