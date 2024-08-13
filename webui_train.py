import json
import os
import gradio as gr
import subprocess
from pathlib import Path

def data_path(path, base):
    return Path(f'./data/{base}/{path}')

def log(data):
    print(data)
    return data

from gradio_log import Log
def get_docker_logs():
    base_path = "./logs/train.log"
    log_path = (base_path) 
    return log_path

def preprocess(project_input_dir, train_input_path, val_input_path, output_path, pre_model_path):
    for state, input_path in zip(['train', 'val'], [data_path(train_input_path, project_input_dir), data_path(val_input_path, project_input_dir)]):
        temp1 = data_path(output_path, project_input_dir)/state/'temp1'
        temp2 = data_path(output_path, project_input_dir)/state/'temp2'
        try:
            temp1.mkdir(parents=True)
            temp2.mkdir(parents=True)
        except Exception as e:
            pass

        print('processing state', state, 'with input_path: ', project_input_dir, input_path, 'temp_path:', temp1, temp2)

        #subprocess.run([r'.\py311\python.exe', 'local/prepare_data.py', 
        out = subprocess.run([r'python3', 'local/prepare_data.py', 

                        '--src_dir', input_path, 
                        '--des_dir', str(temp1)], 
                        #capture_output= True
                        )
        if out.returncode == 0:
            log(f"{state} 数据初始化完成")
        else:
            return log(f"{state} 数据初始化出错 {out}")

        #subprocess.run([r'.\py311\python.exe', 'tools/extract_embedding.py', 
        out = subprocess.run([r'python3', 'tools/extract_embedding.py', 
                        '--dir', str(temp1), 
                        '--onnx_path', "pretrained_models/CosyVoice-300M/campplus.onnx"
                        ],
                        #capture_output= True
                        )
        if out.returncode == 0:
            log(f"{state} 导出 embeddeding 完成")
        else:
            return log(f"{state} 导出 embeddeding 出错 {out}")

        #subprocess.run([r'.\py311\python.exe', 'tools/extract_speech_token.py', 
        out = subprocess.run([r'python3', 'tools/extract_speech_token.py', 
                        '--dir', str(temp1), 
                        '--onnx_path', "pretrained_models/CosyVoice-300M/speech_tokenizer_v1.onnx"
                        ],
                        #capture_output= True
                        )
        if out.returncode == 0:
            log(f"{state} 导出分词 token 完成")
        else:
            return log(f"{state} 导出分词 token 出错 {out}")        

        #subprocess.run([r'.\py311\python.exe', 'tools/make_parquet_list.py', 
        out = subprocess.run([r'python3', 'tools/make_parquet_list.py', 
                        '--num_utts_per_parquet', '100',
                        '--num_processes', '1',
                        '--src_dir', str(temp1),
                        '--des_dir', str(temp2),
                        ],
                        #capture_output= True
                        )
        if out.returncode == 0:
            log(f"{state} 导出 parquet 列表完成")
        else:
            return log(f"{state} 导出 parquet 列表出错 {out}")
    return '预处理全部完成，可以开始训练'

def refresh_voice(project_input_dir, output_path):
    content = (data_path(output_path, project_input_dir)/'train'/'temp1'/'utt2spk').read_text()
    voices = []
    for item in content.split('\n'):
        voices.append(item.split(' ')[0])
    return gr.Dropdown(choices=voices)

    
def train(project_input_dir, output_path, pre_model_path, thread_num, max_epoch):
    output_path = data_path(output_path, project_input_dir)
    train_list = os.path.join(output_path, 'train', 'temp2', 'data.list')
    val_list = os.path.join(output_path, 'val', 'temp2', 'data.list')
    model_dir = Path(f'{output_path}/models')
    model_dir.mkdir(exist_ok=True, parents=True)

    out = subprocess.run([r'torchrun', '--nnodes', '1', '--nproc_per_node', '1', '--rdzv_id', '1986', '--rdzv_backend', "c10d", '--rdzv_endpoint', "localhost:0", 
                    'cosyvoice/bin/train.py',
                    '--train_engine','torch_ddp',
                    '--config','conf/cosyvoice.yaml', 
                    '--max_epoch', str(max_epoch),
                    '--train_data', train_list, '--cv_data', val_list, 
                    '--model','llm', '--checkpoint', os.path.join(pre_model_path, 'llm.pt'), 
                    '--model_dir', str(model_dir), '--tensorboard_dir', str(model_dir),                    
                    '--ddp.dist_backend', 'nccl', '--num_workers', str(thread_num), '--prefetch', '200','--pin_memory', 
                    '--deepspeed_config', './conf/ds_stage2.json', '--deepspeed.save_states', 'model+optimizer', 
                    ], env=dict(os.environ, PYTHONIOENCODING="UTF-8", PYTHONPATH="./:./third_party/Matcha-TTS:./third_party/AcademiCodec"))
    if out.returncode == 0:
        return f"训练完成"
    else:
        return log(f"训练出错啦 {out}")


def inference(mode, project_input_dir, output_path, epoch, pre_model_path, text, voice):
    output_path = data_path(output_path, project_input_dir)
    train_list = os.path.join(output_path, 'train', 'temp2', 'data.list')
    utt2data_list = Path(train_list).with_name('utt2data.list')
    llm_model = os.path.join(output_path, 'models', f'epoch_{epoch}_whole.pt')
    flow_model = os.path.join(pre_model_path, 'flow.pt')
    hifigan_model = os.path.join(pre_model_path, 'hift.pt')

    res_dir = Path(output_path)/'outputs'
    res_dir.mkdir(exist_ok=True, parents=True)

    json_path = str(Path(res_dir)/'tts_text.json')
    with open(json_path, 'wt', encoding='utf-8') as f:
        json.dump({voice:[text]}, f)

    #subprocess.run([r'.\pyr11\python.exe', 'cosyvoice/bin/inference.py', 
    subprocess.run([r'python3', 'cosyvoice/bin/inference.py', 
      '--mode', mode,
      '--gpu', '0', '--config', 'conf/cosyvoice.yaml',
      '--prompt_data', train_list, 
      '--prompt_utt2data', str(utt2data_list), 
      '--tts_text', json_path,
      '--llm_model', llm_model, 
      '--flow_model', flow_model,
      '--hifigan_model', hifigan_model, 
      '--result_dir', str(res_dir)])
    output_path = str(Path(res_dir)/f'{voice}_0.wav')
    return output_path
    

with gr.Blocks() as demo:
    with gr.Group():
        with gr.Accordion("项目根目录，点此展开项目目录文件规则", open=False):
            gr.HTML(value=f'<pre>{Path("data/README").read_text()}</pre>', show_label=False)
        project_input_dir = gr.Text(container=True ,value="test", show_label=False, info="项目数据根目录，在TeamSpace/TTD-Space/CosyVoice_Train/目录下新建，训练数据和模型输出都此文件夹下")
    with gr.Row():
        output_dir = gr.Text(label='模型输出文件夹',value="output",info="预处理与训练最终会输出在项目根目录的本文件夹下，没有会自动新建，一般不用改")
        pretrained_model_path = gr.Text('pretrained_models/CosyVoice-300M', label='预训练模型文件夹', info='可选 300M-SFT/330M-Insturct 一般不用改')
    with gr.Tab('训练'):
        with gr.Row():
            train_input_path = gr.Text(label='训练集目录名',value="train", info="需要自己按要求创建并存放数据，一般不用改")
            val_input_path = gr.Text(label='测试集目录名',value="val", info="需要自己按要求创建并存放数据，一般不用改")
        preprocess_btn = gr.Button('开始预处理（提取训练集音色数据，如果只是要新增推理的音色，只点这个就行了）', variant='primary')
        with gr.Row():
            max_epoch = gr.Number(value=100, interactive=True, precision=0, label="训练总轮次",info="1-1000")
            thread_num = gr.Number(value=1, interactive=True, precision=0, label="训练线程数量",info="每次+1康康，爆显存杀手")
        train_btn = gr.Button('开始训练（如果要把训练数据来影响底层模型，可以用训练的方式）', variant='primary')
        status = gr.Text(label='状态')
    with gr.Tab('推理'):
        with gr.Row():
            voices = gr.Dropdown(label='音色列表', info="根据训练集的数据，在上一步预处理中生成，点右侧刷新")
            refresh = gr.Button('刷新音色列表', variant='primary')
            mode = gr.Dropdown(choices=['sft', 'zero_shot'], label='推理模式', value="sft", info="SFT模型（SFT）和3秒复刻模型（zero-shot）可选，都试试")
            epoch = gr.Number(interactive=True, precision=0, label='模型轮次ID', info="使用模型输出文件夹中训练第？轮次的模型")
        text = gr.Text(label="输入文字")
        inference_btn = gr.Button('开始推理', variant='primary')
        out_audio = gr.Audio()
    Log(get_docker_logs(), dark=True, xterm_font_size=12, render=bool(get_docker_logs()))

    preprocess_btn.click(preprocess, inputs=[project_input_dir, train_input_path, val_input_path, output_dir, pretrained_model_path], outputs=status)
    train_btn.click(train, inputs=[project_input_dir, output_dir, pretrained_model_path, thread_num, max_epoch], outputs=status)
    inference_btn.click(inference, inputs=[mode, project_input_dir, output_dir, epoch, pretrained_model_path, text, voices], outputs=out_audio)
    refresh.click(refresh_voice, inputs=[project_input_dir, output_dir], outputs=voices)

demo.launch(server_name='0.0.0.0',server_port=9883,inbrowser=True)