FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /opt/CosyVoice
COPY . /opt/CosyVoice

RUN apt-get update -y
RUN apt-get -y install git curl ffmpeg wget vim
#RUN git clone --recursive https://github.com/v3ucn/CosyVoice_For_Windows.git CosyVoice
RUN pip3 install -r requirements.txt 
#RUN pip3 install torchaudio==2.0.2 funasr
RUN echo '#!/bin/bash\npython3 webui.py &\npython3 webui_train.py' >> /opt/CosyVoice/service.sh
RUN chmod u+x /opt/CosyVoice/service.sh
#CMD ["sleep","infinity"]
CMD ["./service.sh" ]