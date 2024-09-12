import numpy as np
from scipy.spatial import KDTree
import re


def init_vad_tree(voice_list):
    #  旁白_震惊_3_154799_将那手下撞飞十米之后，才堪堪落地。
    pattern = re.compile(r'.+_(?P<vad>\d{6})_.+')
    for voice in voice_list:
        if voice:
            vad = pattern.match(pattern).groupdict()['vad']