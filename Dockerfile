FROM python:3.9-slim-buster

RUN pip install --upgrade pip
RUN apt-get update

COPY requirements.txt app.py ./
RUN pip install -r requirements.txt

COPY models ./models/
COPY data ./data/
COPY modules ./modules/

EXPOSE 7000
# 7000:5000
CMD ["python3", "app.py"]