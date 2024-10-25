import re
import logging

import numpy as np

"""
tree_obj = {
 [v1,a1,d1] = AU_Key,
 [v2,a2,d2] = AU2_Key,
}

tree_obj.keys() = (
    [v1, a1, d1],
    [v2, a2, d2]
)

KDTree.search(tree_obj.keys(), [v, a, d]) 
    => nearest( [v, a, d], [v, a, d] )
"""


def find_nearest_match_euclidean(target, coordinates, count):
    """找到最近的匹配坐标，优先级为欧式距离"""

    vad_tree = coordinates
    vad = target

    d = ((vad_tree - vad) ** 2).sum(axis=1)  # compute distances
    ndx = d.argsort()  # indirect sort

    result = vad_tree[ndx[:count]].tolist()
    return result


def find_nearest_match_adv(target, coordinates, count):
    """找到最近的匹配坐标，优先级为ADV，根据 a、d、v 的优先级计算综合距离"""
    weight_a = 100  # a 的权重
    weight_d = 10  # d 的权重
    weight_v = 1  # v 的权重

    distances = []
    for coord in coordinates:
        distance = (
            weight_a * abs(target[1] - coord[1])
            + weight_d * abs(target[2] - coord[2])
            + weight_v * abs(target[0] - coord[0])
        )
        distances.append((distance, coord))
    # 按距离排序
    distances.sort(key=lambda x: x[0])

    # 提取前 top_n 个结果
    nearest_coords = [coord for _, coord in distances[:count]]
    return nearest_coords


# 主要入口，取得最相近的VAD count结果
def findNearestVAD(vad: [str or int, str or int, str or int], voices: [str], count=10):
    tree_obj = init_vad_tree(voices)  # text.splitlines()
    # print(tree_obj)
    vad_tree = np.array(list(tree_obj.keys()))
    # print(vad_tree)
    if len(vad_tree) < 1:
        return []
    vad = [int(vad[0]), int(vad[1]), int(vad[2])]
    logging.debug(f"vad: {vad}")

    # result = find_nearest_match_adv(vad, vad_tree, count)  # based from ADV
    result = find_nearest_match_euclidean(vad, vad_tree, count)  # based from euclidean

    logging.debug(f"与 VAD: {vad} 最相近的{count}个 VAD :  {result}")

    result = [tree_obj[tuple(idx)] for idx in result]
    logging.debug(result)
    return result


def init_vad_tree(voice_list):
    #  旁白_震惊_3_154799_将那手下撞飞十米之后，才堪堪落地。
    pattern = re.compile(r".+_(?P<vad>\d{6})_.+")
    tree_obj = {}
    for voice in voice_list:
        if voice:
            # logging.debug(voice, pattern.match(voice).all)
            vad = pattern.match(voice) and pattern.match(voice).group(1)  # \d6 vad
            if vad:
                arr_idx = tuple(int(vad[n : n + 2]) for n in range(0, len(vad), 2))
                # print(arr_idx, type(arr_idx))
                tree_obj[arr_idx] = voice
    return tree_obj


def findNearestASR(target: str, strings: list[str]) -> str | None:
    """
    查找给定字符串列表中与目标字符串包含的感叹号或问号相匹配的最近字符串。

    参数:
    target (str): 目标字符串，可能包含感叹号或问号。
    strings (list[str]): 字符串列表，函数将在其中查找匹配的字符串。

    返回:
    str | None: 如果找到匹配的字符串，则返回该字符串；如果没有找到匹配，则返回 None。
    """

    exclamation = "！"  # 感叹号
    question = "？"  # 问号

    for asr in strings:
        # 如果目标和当前字符串都包含感叹号，则返回当前字符串
        if exclamation in target and exclamation in asr:
            return asr
        # 如果目标和当前字符串都包含问号，则返回当前字符串
        if question in target and question in asr:
            return asr

    return None  # 如果没有找到匹配，返回 None


def findNearestKW(target: str, strings: list[str]) -> str:
    """
        首先计算每个字符串与目标字符串的交集大小，然后筛选出交集大小相同的字符串，最后计算这些字符串的顺序匹配分数，返回分数最高的字符串。
    Args:
        target (str): 待匹配的keyword
        strings (list[str]): 匹配数组
    """

    def char_intersection_count(s1, s2):
        return len(set(s2) & set(s1.split("_")[1]))  # biz_logic

    def order_match_score(s1, s2):
        s1 = s1.split("_")[1]  # biz_logic
        score = 0
        index = 0
        for char in s1:
            if char in s2[index:]:
                score += 1
                index = s2.index(char, index) + 1  # Move index to next position
        return score

    # Step 1: Calculate intersection counts
    intersection_counts = {s: char_intersection_count(s, target) for s in strings}

    # Step 2: Find max intersection count
    max_count = max(intersection_counts.values())
    candidates = [s for s, count in intersection_counts.items() if count == max_count]

    # Step 3: Find the best match based on order match score
    best_match = None
    best_score = -1

    for candidate in candidates:
        score = order_match_score(candidate, target)
        if score > best_score:
            best_score = score
            best_match = candidate

    return best_match


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    text = """对白_担忧_1_349151_小兰，等会儿看到了树枝，你尽快帮我取出来。.wav
对白_讽刺_1_474596_没事，想要生活过得去，总得头上带点绿，不是吗？.wav
对白_讽刺_2_806630_一时间，惨叫声此起彼伏。.wav
旁白_不相信_1_753477_连村霸也都有了文化。.wav
旁白_不相信_2_558207_惹得我是一头雾水的。.wav
旁白_信任_1_974408_我和师父做的虽然是死人的生意，但信的是天命，是自然。.wav
旁白_傲慢_1_807280_总认为啊，天大地大，自己最大。.wav
旁白_傲慢_2_250841_因为鬼婆婆的关系，他们根本就没有见识过唐雅兰的真本事。.wav
旁白_冷静_10_694738_成了被猎豹捕食的小绵羊。.wav
旁白_冷静_11_809617_周泰的那些手下们，没有一个是唐雅兰的一合之敌。.wav
旁白_冷静_12_125472_想将我母亲殓葬入土。.wav
旁白_冷静_13_180152_再看唐雅兰。.wav
旁白_冷静_14_089244_神色平静，目光冷淡。.wav
旁白_冷静_15_382048_打倒了这一群人，仿佛只是踩死了几只蚂蚁而已。.wav
旁白_冷静_16_431721_没费她多少力气，也没有撩动她的心弦。.wav
旁白_冷静_1_678417_那就是，向弱者挥刀。.wav
旁白_冷静_2_297170_这样的性格，地位又高，很自然成了周泰的用来欺压而立威的对象。.wav
旁白_冷静_3_688544_而且到了新时代了。.wav
旁白_冷静_4_705679_至今，鬼婆婆和唐雅兰早就搬离了原地，住进了村委会。.wav
旁白_冷静_5_222825_虽然周泰只是村霸，.wav
旁白_冷静_6_229707_所以也一直忍着。.wav
旁白_冷静_7_518481_因此啊，在唐雅兰朝着周泰一行人冲去之后，.wav
旁白_冷静_8_249612_至于我，师父认为我当时已经死定了。.wav
旁白_冷静_9_319890_反观周泰那些身材高大的手下们，.wav
旁白_哀愁_1_779693_但俗话说得好，因果循环，天心难策。.wav
旁白_哀愁_2_863824_这才不过刚刚开端而已。.wav
旁白_夸赞_1_611972_村子里的村干部解决不了的问题，或是民事纠纷，往往，都要请鬼婆婆出马。.wav
旁白_害怕_1_484498_我开始害怕，我想逃走，但却怎么都逃不掉。.wav
旁白_害怕_2_363796_我看到了她身边的那个孩子，那孩子在对我微笑，似乎对我十分亲近。.wav
旁白_平静_1_809446_村霸，我们那个时代都不少。.wav
旁白_平静_2_617917_不信鬼神，不信因果，不信报应。.wav
旁白_平静_4_977001_所以，时至今日，我们村所在的一大片区域。.wav
旁白_平静_5_290092_不管是村民们，以及我们村周边的村子。.wav
旁白_平静_6_147158_我笑呵呵地说出一句。.wav
旁白_平静_7_264884_鬼婆婆身份特殊。.wav
旁白_平静_8_712866_可也恰恰是因为身份特殊，所以性子，淡泊、孤僻。.wav
旁白_平静_9_636993_她清冷地朝着我看了过来，并朝着我走来。.wav
旁白_悬疑_1_040437_对于鬼神之事，都比较看重。.wav
旁白_悬疑_2_333815_而这类人，大部分都有一个特点。.wav
旁白_悬疑_3_799600_生养唐雅兰的鬼婆婆，是一位落花洞女。.wav
旁白_悬疑_4_950196_却又有一些，巫蛊祭祀的手段。.wav
旁白_惊奇_1_305105_一分钟。从唐雅兰出手到现在，.wav
旁白_惊奇_2_316629_不过只有短短的一分钟而已，.wav
旁白_愤怒_1_339685_周泰每次欺压鬼婆婆，.wav
旁白_愤怒_2_233723_总是以鬼婆婆，宣传迷信活动为借口，合理且合法的欺压她。.wav
旁白_担忧_1_370565_却成了对我这一生影响最大的劫难。.wav
旁白_敬畏_1_157178_是山神之妻，让人畏惧。.wav
旁白_敬畏_2_725204_而师傅也给我取名，肖魏魃。.wav
旁白_敬畏_3_574723_子时过半，鬼节来临。.wav
旁白_沉思_1_594443_可我母亲的身边又的确没有什么树枝。.wav
旁白_焦虑_1_922370_眼见到唐雅兰快冲到他面前了，.wav
旁白_焦虑_2_323531_我怎么都没有想到，.wav
旁白_紧张_1_420327_同时，他狠狠挥动铁铲，朝着唐雅兰的头，狠狠拍去。.wav
旁白_紧张_2_824757_拍向唐雅兰的铲子，又猛又快。.wav
旁白_紧张_3_769471_铲子拍下之际，.wav
旁白_紧张_4_687043_唐雅兰狠挥左手，掌呈刀势，.wav
旁白_紧张_5_781319_周泰吓到了，唐雅兰可没有。.wav
旁白_紧张_6_677830_打断铁铲之后，唐雅兰踏出右脚，.wav
旁白_紧张_7_190423_一记冲拳直冲周泰小腹。.wav
旁白_讥讽_1_033476_我可算是逮着机会反驳他了。.wav
旁白_讥讽_2_243138_以前这小子总是冷不丁地冒出这么些句子。.wav
旁白_讥讽_3_689840_怎么立威？怎么抬高自己？.wav
旁白_轻蔑_1_864448_而村霸最爱干的是什么？.wav
旁白_轻蔑_2_135382_除了赚钱，无疑就是树立自己的威信与地位！.wav
旁白_鄙视_1_929691_周泰就是我们村的村霸。只要有钱赚，什么事，都要参合一脚。.wav
旁白_震惊_1_102026_我看得出来，周泰一点力都没留。是铁了心真要杀人了，.wav
旁白_震惊_2_149386_只可惜，他的速度快，唐雅兰的速度更快，他的力量大，唐雅兰的力量更大。.wav
旁白_震惊_3_843071_旋身摆臂，把周泰如甩铅球一般甩了出去。.wav
旁白_震惊_4_194272_以迅雷不及掩耳之速，率先反砍到周泰手中的铁铲的把手之上。.wav
旁白_震惊_5_552552_“嘭！”一声爆响，铁铲把手应声而断。.wav
旁白_震惊_6_989150_周泰，也随之狠狠一怔。.wav
旁白_震惊_7_642578_农村的铁铲把手都是硬乔木做的，.wav"""
    # text = re.sub(r"[VAD]", lambda x: "%02d" % (random.randint(0, 99)), text)

    import pprint

    pprint.pp(findNearestVAD([12, 28, 50], text.splitlines(), 5))
