
FROM tensorflow/tensorflow:2.6.1

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1

WORKDIR /app

COPY requirements.txt .

RUN /usr/bin/python3 -m pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn

COPY . .

EXPOSE 8000

CMD ["gunicorn", "-c", "gunicorn_conf.py", "app:app"]


