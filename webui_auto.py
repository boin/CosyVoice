import glob
import json
import logging
import os
import subprocess
from hashlib import md5
from pathlib import Path
from zipfile import ZipFile

import gradio as gr
import openpyxl

from tools.auto_ttd import load_actor, load_projects, load_refrence
from tools.emo_dialog_parser import dialog_parser

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def download_all_wavs(wavs, hash=hash):
    logging.debug(f"download all calle with:{wavs}, {hash}")
    fn = f"/tmp/gradio/{hash}.zip"
    with ZipFile(fn, "w") as zip:
        for f in wavs:
            if not Path(str(wavs[f])).is_file():
                continue
            zip.write(wavs[f], f"{f}.wav")
    zip.close()
    logging.info(f"zipfile {fn} writed. size {(Path(fn).lstat()).st_size}")
    return gr.DownloadButton(label="点击下载", value=fn, variant="stop")


def load_wav_cache(project, hash):
    # data/240915_有声书_殓葬禁忌/古装_师父,GZJ_灵异/output/outputs/359d487835f93a92122e54b1a105d19e/359d487835f93a92122e54b1a105d19e-2.wav
    pattern = f"data/{project}/*/*/*/*/{hash}*.wav"
    files = glob.glob(pattern)
    wavs = {}
    for f in files:
        idx = Path(f).stem
        wavs[idx] = f
    # print(pattern, wavs)
    return wavs


def play_ref_audio(project, actor, voice):
    audio_path = (
        Path("./data") / project / actor / "train" / f'{voice.split(" ")[0]}.wav'
    )
    return audio_path


def start_inference(
    project_name, output_path, actor, text, voice, id: str, r_seed, wavs
):
    """开始推理
    Args:
        project_name (str): 项目名称
        output_path (str): 输出路径
        text (str): 推理文本
        voice (str): 选定音色
        id (str): uniqid
        r_seed (str): 随机种子
    """
    # 240915_有声书_殓葬禁忌  旁白_脸红_002_507828_并且在峨眉危难之际自动出现，化解危机。  Says: 我，是一名殓葬师！ {}
    if not text:
        raise gr.Error("no text.")
    if not voice:
        raise gr.Error("no voice.")
    mode = "zero_shot"
    epoch = 0
    r_seed = r_seed[id] if id in r_seed else ""
    pre_model_path = Path("pretrained_models/CosyVoice-300M")
    output_path = Path(f"data/{project_name}/{actor}/{output_path}")
    train_list = output_path / "train" / "temp2" / "data.list"
    utt2data_list = Path(train_list).with_name("utt2data.list")
    llm_model = output_path / "models" / f"epoch_{epoch}_whole.pt"
    flow_model = pre_model_path / "flow.pt"
    hifigan_model = pre_model_path / "hift.pt"
    res_dir = output_path / "outputs" / id.rpartition("-")[0]
    res_dir.mkdir(exist_ok=True, parents=True)
    json_path = str(Path(res_dir) / "tts_text.json")
    with open(json_path, "wt", encoding="utf-8") as f:
        json.dump({voice: [text]}, f)
    logging.info(
        f"call cosyvoice/bin/inference.py {project_name} {mode} => {actor} {voice} says: {text} with r_seed {r_seed}, result to {res_dir}"
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
        str(r_seed),
        "--file_name",
        str(id),
    ]
    subprocess.run(
        cmd,
        env=dict(
            os.environ,
            PYTHONIOENCODING="UTF-8",
            PYTHONPATH="./:./third_party/Matcha-TTS:./third_party/AcademiCodec",
        ),
    )
    output_path = str(Path(res_dir) / f"{id}.wav")
    wavs[id] = output_path
    return output_path, wavs


with gr.Blocks(fill_width=True) as demo:
    wavs = gr.State({"v": 0})
    rseed = gr.State({})
    projects = load_projects()
    hash = ""
    lines = []

    def upload_textbook(text_url: str, project: str, wavs: dict):
        global hash, lines
        hash = md5(
            Path(text_url).read_bytes()
        ).hexdigest()  # save file hash for future use
        workbook = openpyxl.load_workbook(text_url, read_only=True)
        rows = [row for row in workbook.active.rows]
        if not len(rows) > 0:
            raise gr.Error("Empty document")
        lines = list(dialog_parser(rows))
        new_wavs = load_wav_cache(project, hash)  # merge old wavs with new hash values
        wavs["v"] += 1
        for k in new_wavs.keys():
            wavs[k] = new_wavs[k]
        return wavs

    # wavs.value = upload_textbook(
    #     "data/Ch001_天命，将至_QC.xlsx", projects[-1], wavs.value
    # )

    wavs.change(lambda x: print(f"wavs changed.{x}\n"), inputs=wavs)
    rseed.change(lambda x: print(f"rseed changed.{x}\n"), inputs=rseed)

    with gr.Row():
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
        output_dir = gr.Textbox("output", label="输出路径")
        upload = gr.File(label="上传台词本", file_types=["text", ".xlsx"])
        upload.upload(upload_textbook, inputs=[upload, project, wavs], outputs=[wavs])
    with gr.Row():
        gen_all = gr.Button("一键推理", variant="primary")
        gen_all.click(
            None,
            js="()=>{let g=document.querySelectorAll('.gen-btn');g.forEach(x=>!x.id&&x.click())}",
        )
        re_gen_all = gr.Button("全部重推理", variant="primary")
        re_gen_all.click(None,
            js="()=>{let g=document.querySelectorAll('.gen-btn');g.forEach(x=>x.click())}",
        )
        dl_all = gr.DownloadButton("打包下载")
        dl_all.click(
            lambda x: download_all_wavs(x, hash), inputs=[wavs], outputs=[dl_all]
        )

    @gr.render(inputs=[wavs, project])
    def render_lines(_wavs, _project):
        print("re-render wavs version with active project:", _project, _wavs["v"])
        task_list = lines
        # print(hash, task_list[0], _wavs, "\n")
        for task in task_list:
            idx = f'{hash}-{task["id"]}'
            wav_url = idx in _wavs.keys() and _wavs[idx] or None
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        id = gr.Text(
                            f'{hash}-{task["id"]}',
                            visible=False,
                        )
                        gr.Text(
                            f'{task["id"]} {task["actor"]}',
                            label="metadata",
                            show_label=False,
                            container=False,
                            scale=0,
                        )

                        text = gr.Textbox(
                            task["text"],
                            show_label=False,
                            container=False,
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
                        gr.Text(
                            f'{task["emo_1"]} {task["emo_2"]} [ V: {task["V"]} A: {task["A"]} D: {task["D"]} ]',
                            show_label=False,
                            container=False,
                        )
                        actors = load_actor(task["actor"], _project)
                        # print("actors:", actors)
                        actor = gr.Dropdown(
                            choices=actors,
                            value=actors[0],
                            show_label=False,
                            container=False,
                        )
                        refrences = load_refrence(
                            _project,
                            actors[0],  # use parsed actorname than original
                            [task["V"], task["A"], task["D"]],
                            emo_kw=f'{task["emo_1"]}{task["emo_2"]}',
                        )
                        ref_ctl = gr.Dropdown(
                            choices=refrences,
                            value=refrences[0],
                            show_label=False,
                            container=False,
                        )
                with gr.Column(scale=0):
                    preview_audio = gr.Audio(
                        # container=False,
                        label="输出预览",
                        show_download_button=True,
                        show_share_button=False,
                        sources=[],
                        scale=0,
                        value=wav_url,
                    )

                    # for preserve wav_url , not working with lambdas!
                    """
                        In a gr.render, if a variable in a loop is used inside an event listener function,
                        that variable should be "frozen" via setting it to itself as a default argument in the function header.
                        See how we have task=task in both mark_done and delete. This freezes the variable to its "loop-time" value.
                        https://www.gradio.app/guides/dynamic-apps-with-render-decorator
                    """
                    def reset_preview(wav=wav_url):
                        return gr.Audio(value=wav)

                    preview_audio.clear(reset_preview, outputs=[preview_audio])
                gen_btn.click(
                    start_inference,
                    inputs=[project, output_dir, actor, text, ref_ctl, id, rseed, wavs],
                    outputs=[preview_audio, wavs],
                )
                gr.on(
                    triggers=[ref_ctl.focus, ref_ctl.select],
                    fn=play_ref_audio,
                    inputs=[project, actor, ref_ctl],
                    outputs=[preview_audio],
                )
                preview_btn.click(reset_preview, outputs=[preview_audio])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=9884)
