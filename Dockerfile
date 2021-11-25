FROM pytorch/pytorch

RUN pip install ray

RUN mkdir -p /home/app

COPY . /home/app

CMD ["python", "/home/app/main.py"]