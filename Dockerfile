FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /opt/CosyVoice
COPY . /opt/CosyVoice

RUN apt-get update -y
RUN apt-get -y install python3-dev cmake python3-pip git socat curl ffmpeg wget vim
#RUN git clone --recursive https://github.com/v3ucn/CosyVoice_For_Windows.git CosyVoice
RUN pip3 install -r requirements.txt 
#RUN pip3 install torchaudio==2.0.2 funasr
RUN echo '#!/bin/bash\npython3 webui.py &\nsocat TCP4-LISTEN:8001,fork TCP4:127.0.0.1:8000\n' >> /opt/CosyVoice/service.sh
RUN chmod u+x /opt/CosyVoice/service.sh

#CMD ["sleep","infinity"]
CMD ["./service.sh" ]