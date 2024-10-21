import glob
import json
import logging
import os
import random
import torch
import subprocess
from hashlib import md5
from pathlib import Path
from zipfile import ZipFile

import gradio as gr
import openpyxl

from tools.auto_ttd import load_actor, load_projects, load_refrence, load_vc_actor
from tools.emo_dialog_parser import dialog_parser

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# global vars
wavs = {"v": 0}
vcs = {"v": 0}
projects = load_projects()
hash = ""
device = torch.cuda.is_available() and torch.cuda.get_device_name(0) or "CPU"


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


def download_all_vc_wavs():
    # TODO
    return gr.Info("敬请期待")


def load_wav_cache(project, hash: str, type="gen") -> list[str]:
    """
    # data/.outputs/240915_有声书_殓葬禁忌/cosy/359d487835f93a92122e54b1a105d19e/359d487835f93a92122e54b1a105d19e-2.wav
    Args:
        project (_type_): 240915_有声书_殓葬禁忌
        hash (bool): _description_
        type (str, optional): gen 或者 vc. Defaults to "gen".
    Returns:
        list[str]:  pathes of matched wavs
    """
    pattern = (
        f"data/.outputs/{project}/cosy/*/{hash}*.wav"
        if type == "gen"
        else f"data/.outputs/{project}/vc/*/{hash}*.wav"
    )
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


def play_vc_ref_audio(project, actor):
    # TODO
    gr.Info("尽请期待")
    return None


def upload_textbook(text_url: str, project: str):
    global hash, wavs, vcs
    hash = md5(Path(text_url).read_bytes()).hexdigest()  # save file hash for future use
    workbook = openpyxl.load_workbook(text_url, read_only=True)
    rows = [row for row in workbook.active.rows]
    if not len(rows) > 0:
        raise gr.Error("Excel文档为空！")
    lines = list(dialog_parser(rows))
    all_wavs = load_wav_cache(project, hash)  # merge old wavs with new hash values
    all_vcs = load_wav_cache(project, hash, "vc")  # also vc
    wavs["v"] += 1
    vcs["v"] += 1
    # 合并所有 WAV 和 VC
    wavs.update(all_wavs)
    vcs.update(all_vcs)
    return lines


def start_inference(
    project_name, output_path, actor, text, voice, id: str, r_seed, wavs=wavs
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
    pre_model_path = Path("pretrained_models/CosyVoice-300M")
    output_path = Path(f"data/{project_name}/{actor}/{output_path}")
    train_list = output_path / "train" / "temp2" / "data.list"
    utt2data_list = Path(train_list).with_name("utt2data.list")
    llm_model = output_path / "models" / f"epoch_{epoch}_whole.pt"
    llm_model = pre_model_path / "llm.pt"  # just using default pt
    flow_model = pre_model_path / "flow.pt"
    hifigan_model = pre_model_path / "hift.pt"
    res_dir = Path(f"data/.outputs/{project_name}/cosy/") / id.rpartition("-")[0]
    res_dir.mkdir(exist_ok=True, parents=True)
    json_path = str(Path(res_dir) / "tts_text.json")
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
        str(r_seed),
        "--file_name",
        str(id),
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
    output_path = str(Path(res_dir) / f"{id}.wav")
    wavs[id] = output_path
    return output_path


def start_vc(project: str, actor: str, audio_path: str, tone_key):
    # TODO
    gr.Info("尽请期待")
    print(audio_path)
    return None
    from gradio_client import Client

    client = Client("http://ttd-server:7866/")
    result = client.predict(
        0,  # float (numeric value between 0 and 2333) in '请选择说话人id' Slider component
        audio_path,  # str (filepath on your computer (or URL) of file) in '输入待处理音频文件' File component
        tone_key,  # float  in '变调(整数, 半音数量, 升八度12降八度-12)' Number component
        None,  # str (filepath on your computer (or URL) of file) in 'F0曲线文件, 可选, 一行一个音高, 代替默认F0及升降调' File component
        "rmvpe",  # str  in '选择音高提取算法,输入歌声可用pm提速,harvest低音好但巨慢无比,crepe效果好但吃GPU,rmvpe效果最好且微吃GPU' Radio component
        None,  # str  in '特征检索库文件路径,为空则使用下拉的选择结果' Textbox component
        None,  # str (Option from: ['assets/indices/HM_IVF211_Flat_nprobe_1_HM_v2.index', 'assets/indices/LMC1_IVF569_Flat_nprobe_1_LMC1_v2.index', 'assets/indices/LMC_IVF569_Flat_nprobe_1_LMC_v2.index', 'assets/indices/ZSYD_1_IVF1509_Flat_nprobe_1_ZSYD_1_v2.index', 'assets/indices/ZSYD_2_IVF819_Flat_nprobe_1_ZSYD_2_v2.index', 'assets/indices/guliang-LHY_IVF778_Flat_nprobe_1_guliang-LHY_v2.index', 'assets/indices/hufa-DRE_IVF825_Flat_nprobe_1_hufa-DRE_v2.index', 'assets/indices/hufa-DRE_IVF829_Flat_nprobe_1_hufa-DRE_v2.index', 'assets/indices/longwang_songzhi_IVF816_Flat_nprobe_1_longwang_songzhi_v2.index', 'logs/HM/added_IVF211_Flat_nprobe_1_HM_v2.index']) in '自动检测index路径,下拉式选择(dropdown)' Dropdown component
        0.75,  # float (numeric value between 0 and 1) in '检索特征占比' Slider component
        3,  # float (numeric value between 0 and 7) in '>=3则使用对harvest音高识别的结果使用中值滤波，数值为滤波半径，使用可以削弱哑音' Slider component
        0,  # float (numeric value between 0 and 48000) in '后处理重采样至最终采样率，0为不进行重采样' Slider component
        0.25,  # float (numeric value between 0 and 1) in '输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络' Slider component
        0.33,  # float (numeric value between 0 and 0.5) in '保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，调低加大保护力度但可能降低索引效果' Slider component
        api_name="/infer_convert",
    )
    print(result)


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
            js="()=>{let g=document.querySelectorAll('.gen-btn');g.forEach(x=>!x.id&&x.click())}",
        )
        re_gen_all = gr.Button("全部重推理", variant="primary")
        re_gen_all.click(
            None,
            js="()=>{let g=document.querySelectorAll('.gen-btn');g.forEach(x=>x.click())}",
        )
        dl_all = gr.DownloadButton("打包下载推理")
        dl_all.click(lambda: download_all_wavs(wavs, hash), outputs=[dl_all])
        vc_all = gr.Button("一键VC", variant="primary")
        vc_all.click(
            None,
            js="()=>{return false;let g=document.querySelectorAll('.vc-btn');g.forEach(x=>!x.id&&x.click())}",
        )
        re_vc_all = gr.Button("全部重VC", variant="primary")
        re_vc_all.click(
            None,
            js="()=>{return false;let g=document.querySelectorAll('.vc-btn');g.forEach(x=>x.click())}",
        )
        dl_vc_all = gr.DownloadButton("打包下载VC")
        dl_vc_all.click(lambda: download_all_vc_wavs(wavs, hash), outputs=[dl_vc_all])

    @gr.render(inputs=[lines, project])
    def render_lines(_lines, _project):
        logging.debug(
            "re-render wavs version with active project:", _project, wavs["v"]
        )
        task_list = _lines
        # print(hash, task_list[0], _wavs, "\n")
        for task in task_list:
            # print(task)
            idx = f'{hash}-{task["id"]}'
            wav_url = idx in wavs.keys() and wavs[idx] or None
            vc_url = idx in vcs.keys() and vcs[idx] or None
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
                            emo_kw=f'{task["emo_1"]}{task["emo_2"]}',
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
                        vc_actors = load_vc_actor(actor, _project)
                        gr.Text(
                            "VC音色选择 [声调，模型] ",
                            show_label=False,
                            container=False,
                            scale=2,
                            min_width=100,
                        )
                        tone_key = gr.Dropdown(
                            choices=[(i, str(i)) for i in range(-5, 5)],
                            value="0",
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
                        _seed if _seed.value else seed,
                    ],
                    outputs=[preview_audio],
                )
                vc_btn.click(reset_audio, outputs=preview_audio).then(
                    start_vc,
                    inputs=[project, vc_actor, tone_key, preview_audio],
                    outputs=[preview_audio],
                )
                preview_vc_btn.click(lambda: vc_url, outputs=preview_audio)
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
