FROM python:3.10 AS builder

RUN mkdir -p app
WORKDIR app

RUN apt update && apt install -y libsm6 libxext6 ffmpeg libfontconfig1 libxrender1 libgl1-mesa-glx git
RUN pip3 install --upgrade pip

COPY requirements.txt .
COPY usmail.pt .
COPY app.py .

RUN pip3 install -r requirements.txt



ENTRYPOINT ["python3"]
CMD ["app.py"]


