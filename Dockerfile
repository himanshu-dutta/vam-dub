FROM nvcr.io/nvidia/pytorch:24.01-py3

RUN pip install -U numpy==1.23.4
RUN pip install git+https://github.com/aliutkus/speechmetrics#egg=speechmetrics

WORKDIR /home
COPY visual-acoustic-matching /home/visual-acoustic-matching

WORKDIR /home/visual-acoustic-matching
RUN python -m pip install -e .
RUN pip install -U numpy==1.23.4
RUN pip install matplotlib astropy decord
RUN pip uninstall opencv opencv-python opencv-python-headless -y
RUN pip install opencv-python==4.8.0.74
RUN pip3 install torch torchvision torchaudio
RUN pip uninstall torchtext -y
RUN pip install torchtext
RUN pip uninstall torchmetrics -y
RUN pip install torchmetrics==0.6.0
RUN pip install fastapi uvicorn
RUN pip install python-multipart
RUN pip install transformers datasets metrics
RUN pip install streamlit

COPY docker-requirements/torchmetrics_imports.py /usr/local/lib/python3.10/dist-packages/torchmetrics/utilities/imports.py

WORKDIR /home
CMD [ "/bin/bash" ]