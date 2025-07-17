FROM python:3.12

RUN pip install torch torchtext

COPY run.py .
COPY models.py .
COPY tokens.json .
COPY model_weights.pt .

CMD ["python", "run.py"]