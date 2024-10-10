import re


def dialog_parser(str: str):
    return dialog_parser_excel(str)


def dialog_parser_text(str: str):
    """lines
    [
        { #: line <旁白001><旁白><冷静_Rationality><V: 0.46, A: 0.32, D: 0.55>我，是一名殓葬师！很多人听说我的职业后，都会误以为我的工作和入殓师差不多。但其实，这两者有着天差地别。<Pause: 0.8s></旁白>
            id: 旁白001,
            actor: "旁白"
            emo_chn: "冷静",
            emo_eng: "Rationality",
            V: 0.46,
            A: 0.32,
            D: 0.55,
            text: "...."
        }
    ]
    """
    partern = r"\<(?P<id>.*\d{3})\>\<(?P<actor>.+)\>\<(?P<emo_chn>.+)_(?P<emo_eng>.+?)\>\<V: (?P<V>0\.\d\d), A: (?P<A>0.\d\d), D: (?P<D>0.\d\d)\>(?P<text>.+?)\<\/(\2)\>"
    match = re.search(partern, str)
    return match.groupdict()


def dialog_parser_excel(rows: list):
    """
    角色	情绪        V       A       D       内容	                                                                                       Pause(s)
    旁白	坚定,自豪	0.482	0.601	0.672	我，是一名殓葬师！很多人听说我的职业后，都会误以为我的工作和入殓师差不多，但其实，这两者有着天差地别。	1.2
    """
    headers = rows.pop(0) #pop-out header
    headers = ['id', "actor", "emo_1", "emo_2", "V", "A", "D", "text", "contorl"]
    for id, row in enumerate(rows): 
        if not ( row[0].value or row[1].value or row[2].value or row[3].value or row[4].value or row[5].value ):
            print(f"Excel 文件第{id}行, 有内容为空，跳过")
            continue
        #print('processing line {}'.format(id))
        emos = row[1].value.split(",")
        emos = emos[:2] if len(emos) > 1 else [emos[0], ""] #make sure emos has 2 elements
        content = [id+1, row[0].value ] + emos + [row[2].value, row[3].value, row[4].value, row[5].value]
        #print(content)
        yield dict(zip(headers, content))