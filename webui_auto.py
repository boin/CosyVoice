import json
import random
import subprocess
from pathlib import Path
from tools.emo_dialog_parser import dialog_parser
from tools.auto_ttd import load_refrence, load_actor, load_projects

import logging
import gradio as gr

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def upload_textbook(text_url):
    texts = Path(text_url).read_text().splitlines()
    if not len(texts) > 0:
        return "Empty lines"
    lines = []
    for text in texts:
        if not text:
            continue
        text = dialog_parser(text)
        lines.append(text)
    return lines


def start_inference(project_name, output_path, text, voice, id, r_seed):
    if not text:
        raise gr.Error("no text.")
    if not voice:
        raise gr.Error("no voice.")
    mode = "zero-shot"
    epoch = 0
    pre_model_path = Path("pretrained_models/CosyVoice-300M")
    output_path = Path(f"data/{project_name}/{output_path})")
    train_list = output_path / "train" / "temp2" / "data.list"
    utt2data_list = Path(train_list).with_name("utt2data.list")
    llm_model = output_path / "models" / f"epoch_{epoch}_whole.pt"
    flow_model = pre_model_path / "flow.pt"
    hifigan_model = pre_model_path / "hift.pt"
    res_dir = output_path / "outputs" / str(random.randint(1e16, 1e17))
    res_dir.mkdir(exist_ok=True, parents=True)
    voice = voice.split(" - ")[1]  # spkr1 - voice1 => voice1
    if not voice:
        raise "empty voice."
    json_path = str(Path(res_dir) / "tts_text.json")
    with open(json_path, "wt", encoding="utf-8") as f:
        json.dump({voice: [text]}, f)

    logging.info(
        f"call cosyvoice/bin/inference.py {mode} => {voice} says: {text} with r_seed {r_seed}"
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
    ]
    subprocess.run(cmd)
    output_path = str(Path(res_dir) / f"{voice}_0.wav")
    return output_path, gr.Button(link=output_path, variant="stop")


with gr.Blocks(fill_width=True) as demo:
    s = gr.State(upload_textbook("data/第一章 天命，将至_final.txt"))
    projects = load_projects()
    project_name = gr.State(projects[-1] if len(projects) > 0 else "")
    rseed = gr.State({})
    # print(s)

    with gr.Row():
        project = gr.Dropdown(choices=projects, value=project_name.value, label="项目名称")
        project.change(lambda x: x, project, project_name)
        gr.Button("刷新").click(
            lambda: {"__type__": "update", "choices": load_projects()},
            inputs=[],
            outputs=[project],
        )
        output_dir = gr.Textbox("output", label="输出路径")
        upload = gr.File(label="上传台词本", file_types=["text"])
        upload.upload(upload_textbook, inputs=[upload], outputs=[s])

    @gr.render(inputs=s)
    def render_lines(task_list):
        # print(task_list)
        # complete = [task for task in task_list if task["complete"]]
        # incomplete = [task for task in task_list if not task["complete"]]
        # gr.Markdown(f"### Incomplete Lines ({len(incomplete)})")
        for task in task_list:
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        id = gr.Text(task["id"], render=False)
                        gr.Text(
                            f'{task["id"]} {task["actor"]}',
                            label="metadata",
                            show_label=False,
                            container=False,
                            scale=0,
                        )

                        # gr.Text(task['id'], show_label=False, container=False, scale=0)
                        # gr.Text(task['actor'], show_label=False, container=False, scale=0)
                        # gr.Text(f"{task['emo_chn']} {task['emo_eng']}", show_label=False, container=False, scale=0)
                        # gr.Text(task['id'])
                        # gr.Text(task['id'])

                        text = gr.Textbox(
                            task["text"], show_label=False, container=False
                        )
                        gen_btn = gr.Button("生成", scale=0, variant="primary")
                        download_btn = gr.Button("下载", scale=0)
                        # done_btn.click(lambda: False, None, [s])

                    with gr.Row():
                        gr.Text(
                            f'{task["emo_chn"]}  [ V: {task["V"]} A: {task["A"]} D: {task["D"]} ]',
                            show_label=False,
                            container=False,
                        )
                        actors = load_actor(task["actor"], project_name.value)
                        print("actors:", actors)
                        gr.Dropdown(
                            choices=actors,
                            value=actors[0],
                            show_label=False,
                            container=False,
                        )
                        refrences = (
                            load_refrence(
                                project_name.value,
                                actors[0], #use parsed actorname than original
                                [task["V"], task["A"], task["D"]],
                                emo_kw=task["emo_chn"],
                            ),
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
                        show_download_button=False,
                        show_share_button=False,
                        sources=[],
                        scale=0,
                    )
                gen_btn.click(
                    start_inference,
                    inputs=[project_name, output_dir, text, ref_ctl, id, rseed],
                    outputs=[preview_audio, download_btn],
                )
                # download_btn.click(lambda: gr.Info("WIP"))

        gr.Button("一键三连", variant="primary").click(lambda: None)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=9883)
