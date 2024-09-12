import json
import os
import random
import subprocess
from pathlib import Path
from tools.emo_dialog_parser import dialog_parser
from tools.auto_tdd import load_refrence, load_actor
import logging
import gradio as gr
import psutil
from gradio_log import Log

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


with gr.Blocks(fill_width=True) as demo:
    s = gr.State(upload_textbook("data/lines.txt"))
    project_name = gr.State("test")
    # print(s)

    with gr.Row():
        project = gr.Textbox("test", label="项目名称")
        project.change(lambda x: x, project, project_name)
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

                        gr.Textbox(task["text"], show_label=False, container=False)
                        done_btn = gr.Button("生成", scale=0, variant="primary")
                        download_btn = gr.Button("下载",scale=0)
                        #done_btn.click(lambda: False, None, [s])

                    with gr.Row():
                        gr.Text(
                            f'{task["emo_chn"]} ( {task["emo_eng"]} ) [ V: {task["V"]} A: {task["A"]} D: {task["D"]} ]',
                            show_label=False,
                            container=False,
                        )
                        actor_ctl = gr.Dropdown(
                            choices=load_actor(task["actor"], project_name.value),
                            show_label=False,
                            container=False,
                        )
                        ref_ctl = gr.Dropdown(
                            choices=load_refrence(
                                task["actor"],
                                [task["V"], task["A"], task["D"]],
                                project_name.value,
                            ),
                            show_label=False,
                            container=False,
                        )
                with gr.Column(scale=0):
                    preview_audio = gr.Audio(
                        #container=False,
                        label="输出预览",
                        show_download_button=False,
                        show_share_button=False,
                        sources=[],
                        scale=0,
                    )
        gr.Button("一键三连", variant="primary")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=9883, inbrowser=False)