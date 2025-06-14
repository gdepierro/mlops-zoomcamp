FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app

COPY ["./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0:9696", "predict:app"]

