import json
import os
import random
import subprocess
from pathlib import Path
from tools.emo_dialog_parser import dialog_parser
from tools.auto_tdd import load_refrence, load_actor

import gradio as gr
import psutil
from gradio_log import Log


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
                            f'{task["emo_chn"]} ({task["emo_eng"]})',
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
    """lines = <旁白001><旁白><冷静_Rationality><V: 0.4, A: 0.3, D: 0.5>我，是一名殓葬师！很多人听说我的职业后，都会误以为我的工作和入殓师差不多。但其实，这两者有着天差地别。<Pause: 0.8s></旁白>

<旁白002><旁白><冷静_Rationality><V: 0.4, A: 0.3, D: 0.5>入殓师的工作，是为死者化妆，着装。他们负责的，是让死者体面的走完阳世间的最后一段路。而殓葬师，则是负责让死者殓入棺材，葬进陵墓。<Pause: 0.8s></旁白>

<旁白003><旁白><沉思地_contemplative><V: 0.5, A: 0.4, D: 0.5>而我们的职责，是让死者走上该走的路。熟话说，人死入土，进而尘归尘，土归土。但是，就如世间有许多人不愿死亡、抗拒死亡一样。<Pause: 0.6s></旁白>

<旁白004><旁白><庄重_vow><V: 0.3, A: 0.3, D: 0.6>也有许多死者不愿入土，抗拒入棺。而这些年，我所遇到的拒抗入棺的死者，方式也全都千奇百怪。<Pause: 0.6s></旁白>

<旁白005><旁白><惊讶_Surprised><V: 0.4, A: 0.6, D: 0.4>有死后尸身重愈千钧的，有已入棺却又诈尸而起的，也有守尸夜里尸传哭声，引黑猫相聚，最终尸变的……！然而这些，都尚且只能称之为奇怪罢了！<Pause: 0.7s></旁白>

<旁白006><旁白><恐怖_Terrified><V: 0.2, A: 0.9, D: 0.3>我这辈子遇到的最恐怖的，还是我师父死时所发生的——九龙压残尸，十鬼阻阴路！那件事，也引领了我正式走上殓葬师之路。<Pause: 0.8s></旁白>

<旁白007><旁白><紧张_Nervous><V: 0.3, A: 0.7, D: 0.5>事情，还要从我师父八十岁那年的七月十五中元节说起。众所周知，七月半，鬼门开。群鬼入世，生人回避。恰好，我就是七月十五生的人。而且还是七月十五凌晨子时出生。<Pause: 0.8s></旁白>

<旁白008><旁白><悲伤_Sadness><V: 0.3, A: 0.5, D: 0.4>据我师父说，我的母亲是他在乱葬岗遇到的一具没有入土的怀孕女尸。和死人打交道的，都阴气重，福德浅，都讲究要积阴德。于是，我师父打着积阴德的打算，想将我母亲殓葬入土。<Pause: 0.7s></旁白>

<旁白009><旁白><惊奇_Surprise><V: 0.4, A: 0.7, D: 0.4>至于我，师父认为我当时已经死定了。可就在他将我母亲抱入新挖的坟坑，刚准备覆土之际，日子正好由七月十四转入七月十五。子时过半，鬼节来临。<Pause: 0.8s></旁白>

<旁白010><旁白><惊奇_Surprise><V: 0.5, A: 0.8, D: 0.3>而我，也在这鬼门大开之际，撑起了我母亲的肚土，让师父发现我还活着。最后还是师父他捡了根锋利的树枝，划开了我母亲的肚皮，才让我出生了。<Pause: 0.7s></旁白>

<旁白011><旁白><欣喜_Elated><V: 0.4, A: 0.6, D: 0.5>而师父，也给我取名——肖魏魃！鬼是刚死之人，而婴是刚转生之鬼。<Pause: 0.6s></旁白>

<旁白012><旁白><神秘_Mysterious><V: 0.3, A: 0.5, D: 0.4>据师父说，他将我从娘胎里接出来时，我没哭，而是在笑。一会儿笑得狂，一会儿又笑得狠，一会儿又笑得厉。他认为，这是初生的我，引吸了从鬼门关内出来的鬼怪。<Pause: 0.7s></旁白>

<旁白013><旁白><神秘_Mysterious><V: 0.3, A: 0.5, D: 0.4>并且那些鬼怪试图占据我的身体。情急之下，他替我取了这个名字。肖是随他姓，而魏与魃都带着‘鬼’字。其目的，就是让鬼也误认为我是个鬼。<Pause: 0.7s></旁白>

<旁白014><旁白><平静_Serenity><V: 0.4, A: 0.3, D: 0.5>总之，我活了下来，并平平安安长到了这么大。可是，我身边的人却没有这么好运。<Pause: 0.6s></旁白>

<旁白015><旁白><无奈_pout><V: 0.3, A: 0.4, D: 0.4>别看我是农历七月十五凌晨12点，鬼门大开之际出生。但我的八字，其实是不错的。是辛未年丙申月丙寅年戊子时！五行齐不缺，且中正平和。<Pause: 0.7s></旁白>

<旁白016><旁白><沉思_Pensiveness><V: 0.4, A: 0.3, D: 0.4>从八字上来看，应该利官近贵，多勤而多有获。命有一妻，贤良温德，周全一生。但事实却是，我身边的人总是多灾多难。<Pause: 0.7s></旁白>

<旁白017><旁白><悲伤_Sadness><V: 0.3, A: 0.4, D: 0.4>尤其是我师父。自打我有记忆以来，他总是多病，唯有在中元节那天会好一些。但恰恰是师父八十岁那一年，截然不同。从那年年初开始，师父表现出了一个八十岁老人家不该有的强健身姿。<Pause: 0.6s></旁白>

<旁白018><旁白><惊讶_Surprised><V: 0.5, A: 0.7, D: 0.5>并且在短短的半年内，四处奔走，殓葬了十名不愿入棺的死者。要知道，我们这一行，一年里都不一定能处理十宗拒葬者。而那，也让我误以为师父的疾运走完了。<Pause: 0.8s></旁白>

<旁白019><旁白><紧张_Nervous><V: 0.3, A: 0.6, D: 0.5>直到中元节那一天。我记得很清楚，那也是子时刚过。睡梦中的我，被一阵悉悉嗦嗦的声音吵醒。我所在的地方，是个很小的村子。住的房子也不大，只一间卧室。<Pause: 0.7s></旁白>

<旁白020><旁白><担忧_Apprehension><V: 0.3, A: 0.5, D: 0.4>我从小就和师父同屋分床而睡。当我睁开双眼的时候，只见到师父坐在他的床上，双眼直愣愣地看着自己的身前。那时候，我的意识才刚刚恢复了少许。<Pause: 0.8s></旁白>

<旁白021><旁白><恐惧_Fearful><V: 0.2, A: 0.6, D: 0.4>只是隐隐约约听到师父说了一句：<Pause: 0.8s></旁白>

<台词001><师傅><庄重_vow><V: 0.3, A: 0.5, D: 0.6>“天规法度，幽冥刑责，众生难违。一切全看天意，你我就此约定！”<Pause: 0.6s></师傅>

<旁白022><旁白><焦虑_Anxiety><V: 0.3, A: 0.7, D: 0.4>我从小跟师父学习殓葬，怪事也见过不少。自然知道，师父不会无缘无故有这种表现。惊讶也罢，惶恐也好。总之师父说的话，让我瞬间被激得彻底惊醒了。<Pause: 0.6s></旁白>

<旁白023><旁白><惊恐_Terrified><V: 0.2, A: 0.9, D: 0.4>我下意识下床，想走到师父身边。却不料，师父却也正好转头看向了我，朝着招了招手。我当即跑了过去。还没来得及问师父到底怎么了，他便先开口了。<Pause: 0.7s></旁白>

<台词002><师傅><冷静_Rationality><V: 0.3, A: 0.5, D: 0.6>“肖儿，师父的天命要来了！”<Pause: 0.8s></师傅>

    for line in lines.splitlines():
        if not line: continue
        print(line, dialog_parser(line))
"""
