FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

#COPY requirements.txt .
#RUN pip3 install --no-cache-dir -r requirements.txt

COPY starter.py .

ENTRYPOINT ["python3", "starter.py"]
