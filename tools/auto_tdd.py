from pathlib import Path

def load_refrence(actor, emo, project_name):
    print('load refrence called:', actor, emo)
    root_dir = f"./data/{project_name}"

    content = Path(
        f'{root_dir}/output/train/temp1/utt2spk'
    ).read_text()
    voices = []
    for item in content.split("\n"):
        if (item): 
            [ voice, spkr ] = item.split(" ")
            voices.append(f'{spkr} - {voice}')
    return voices

def load_actor(actor: str, project_name):
    #print('load actor called:', actor, project_name)
    root_dir = f"./data/{project_name}"
    content = Path(
         f'{root_dir}/output/train/temp1/spk2utt'
    ).read_text()
    spks = []
    for item in content.split("\n"):
        if (item): 
            spkr = item.split(" ")[0]
            spks.append(f'{spkr}')

    print('globing:', content, 'got:', spks)
    return spks