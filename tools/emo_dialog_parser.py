import re

def dialog_parser(str: str):
    """lines
    [
        { #: line <旁白001><旁白><冷静_Rationality><V: 0.4, A: 0.3, D: 0.5>我，是一名殓葬师！很多人听说我的职业后，都会误以为我的工作和入殓师差不多。但其实，这两者有着天差地别。<Pause: 0.8s></旁白>
            id: 001,
            type: "旁白"
            emo_chn: "冷静",
            emo_eng: "Rationality",
            vad: [4,3,5],
            text: "...."
        }
    ]
    """
    partern = r"\<(?P<id>.*\d{3})\>\<(?P<actor>.+)\>\<(?P<emo_chn>.+)_(?P<emo_eng>.+?)\>\<V: (?P<V>0\.\d), A: (?P<A>0.\d), D: (?P<D>0.\d)\>(?P<text>.+?)\<\/(\2)\>"
    match = re.search(partern, str)
    return match.groupdict()
