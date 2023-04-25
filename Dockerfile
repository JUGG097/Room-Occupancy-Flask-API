FROM python:3.9-slim-buster

RUN pip install --upgrade pip
RUN apt-get update

COPY requirements.txt app.py ./
COPY models ./models/
COPY modules ./modules/

RUN pip install -r requirements.txt

EXPOSE 7000
# 7000:5000
CMD ["python3", "app.py"]