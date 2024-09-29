#!/bin/bash
# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8
export PYTHONPATH=../../../:../../../third_party/Matcha-TTS:$PYTHONPATH

[ ! -L cosyvoice ] && ln -s ../../../cosyvoice .
[ ! -L tools ] && ln -s ../../../tools .

chmod +x tools/*.py
