FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /opt/CosyVoice
COPY . /opt/CosyVoice

RUN apt-get update -y
RUN apt-get -y install git curl ffmpeg wget vim locales
RUN locale-gen en_US en_US.UTF-8
RUN pip3 install -r requirements.txt
RUN pip3 install -U numpy==1.26.4
RUN mkdir -p logs
RUN echo '#!/bin/bash\npython3 -u webui.py 2>&1 | stdbuf -oL -eL tee -i logs/ui.log &\n  python3 -u webui_train.py 2>&1 | stdbuf -oL -eL tee -i logs/train.log ' >> /opt/CosyVoice/service.sh
RUN chmod u+x /opt/CosyVoice/service.sh
#CMD ["sleep","infinity"]
CMD ["./service.sh" ]