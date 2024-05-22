FROM python:3.11.4-slim

WORKDIR /quang
COPY . /quang

RUN pip3 install -r requirement.txt

RUN conda install -c conda-forge faiss-gpu

CMD ["python", "app.py"]
