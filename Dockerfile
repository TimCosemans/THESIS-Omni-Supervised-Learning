FROM tensorflow/tensorflow:latest-gpu

RUN pip install --upgrade pip

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY src src
COPY setup.py setup.py
RUN pip install .
