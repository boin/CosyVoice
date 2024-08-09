import json
import os
import gradio as gr
import subprocess
from pathlib import Path

def data_path(path, base):
    return Path(f'./data/{base}/{path}')

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
                        '--des_dir', str(temp1)])
        if (out.returncode == 0):
            yield f"{state} 数据初始化完成"
        else:
            return f"{state} 数据初始化出错 {out}"

        #subprocess.run([r'.\py311\python.exe', 'tools/extract_embedding.py', 
        out = subprocess.run([r'python3', 'tools/extract_embedding.py', 
                        '--dir', str(temp1), 
                        '--onnx_path', "pretrained_models/CosyVoice-300M/campplus.onnx"
                        ])
        if (out.returncode == 0):
            yield f"{state} 导出embeddeding完成"
        else:
            return f"{state} 导出embeddeding出错 {out}"

        #subprocess.run([r'.\py311\python.exe', 'tools/extract_speech_token.py', 
        out = subprocess.run([r'python3', 'tools/extract_speech_token.py', 
                        '--dir', str(temp1), 
                        '--onnx_path', "pretrained_models/CosyVoice-300M/speech_tokenizer_v1.onnx"
                        ])
        if (out.returncode == 0):
            yield f"{state} 导出分词token完成"
        else:
            return f"{state} 导出分词token出错 {out}"

        #subprocess.run([r'.\py311\python.exe', 'tools/make_parquet_list.py', 
        out = subprocess.run([r'python3', 'tools/make_parquet_list.py', 
                        '--num_utts_per_parquet', '100',
                        '--num_processes', '1',
                        '--src_dir', str(temp1),
                        '--des_dir', str(temp2),
                        ])
        if (out.returncode == 0):
            yield f"{state} 导出parquet列表完成"
        else:
            return f"{state} 导出parquet列表出错 {out}"

    return '预处理全部完成，可以训练'

def refresh_voice(project_input_dir, output_path):
    content = (data_path(output_path, project_input_dir)/'train'/'temp1'/'utt2spk').read_text()
    voices = []
    for item in content.split('\n'):
        voices.append(item.split(' ')[0])
    return gr.Dropdown(choices=voices)

    
def train(project_input_dir, output_path, pre_model_path):
    output_path = data_path(output_path, project_input_dir)
    train_list = os.path.join(output_path, 'train', 'temp2', 'data.list')
    val_list = os.path.join(output_path, 'val', 'temp2', 'data.list')
    model_dir = Path(output_path)/'models'
    model_dir.mkdir(exist_ok=True, parents=True)

    subprocess.run([r'torchrun', '--nnodes', '1', '--nproc_per_node', '1', '--rdzv_id', '1986', '--rdzv_backend', "c10d", '--rdzv_endpoint', "localhost:0", 
                    'cosyvoice/bin/train.py',
                    '--train_engine','torch_ddp',
                    '--config','conf/cosyvoice.yaml', 
                    '--train_data', train_list, '--cv_data', val_list, 
                    '--model','llm', '--checkpoint', os.path.join(pre_model_path, 'llm.pt'), 
                    '--model_dir', str(model_dir), '--tensorboard_dir', str(model_dir),                    
                    '--ddp.dist_backend', 'nccl', '--num_workers', '1', '--prefetch', '100','--pin_memory', 
                    '--deepspeed_config', './conf/ds_stage2.json', '--deepspeed.save_states', 'model+optimizer', 
                    ])
    return 'Train done!'



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
    pretrained_model_path = gr.Text('pretrained_models/CosyVoice-300M', label='预训练模型文件夹')
    output_dir = gr.Text(label='模型输出文件夹，输出在项目目录下',value="output")
    project_input_dir = gr.Text(label='项目目录名（数据根目录）',value="test")
    with gr.Tab('训练'):
        train_input_path = gr.Text(label='训练集目录名',value="train")
        val_input_path = gr.Text(label='测试集目录名',value="val")
        preprocess_btn = gr.Button('预处理（提取训练集音色数据）', variant='primary')
        train_btn = gr.Button('开始训练', variant='primary')
        status = gr.Text(label='状态')
    with gr.Tab('推理'):
        with gr.Row():
            voices = gr.Dropdown(label='音色列表')
            refresh = gr.Button('刷新音色列表', variant='primary')
            mode = gr.Dropdown(choices=['sft模型', 'zero_shot（3秒复刻模型）'], label='Mode')
            epoch = gr.Number(value=8, interactive=True, precision=0, label='使用训练第？轮次的模型')
        text = gr.Text()
        inference_btn = gr.Button('开始推理', variant='primary')
        out_audio = gr.Audio()

    preprocess_btn.click(preprocess, inputs=[project_input_dir, train_input_path, val_input_path, output_dir, pretrained_model_path], outputs=status)
    train_btn.click(train, inputs=[project_input_dir, output_dir, pretrained_model_path], outputs=status)
    inference_btn.click(inference, inputs=[mode, project_input_dir, output_dir, epoch, pretrained_model_path, text, voices], outputs=out_audio)
    refresh.click(refresh_voice, inputs=[project_input_dir, output_dir], outputs=voices)

demo.launch(server_name='0.0.0.0',server_port=9883,inbrowser=True)