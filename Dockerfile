FROM python:3
ADD . /app
WORKDIR /app
RUN pip install -r /app/requirements.txt
CMD cd /app/app && python3 app.py
