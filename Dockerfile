FROM python:3.14-slim
WORKDIR /srv/gnomon
COPY pyproject.toml README.md ./
COPY gnomon ./gnomon
RUN pip install --no-cache-dir .
ENV GNOMON_DATA=/var/lib/gnomon
EXPOSE 8737
CMD ["uvicorn", "gnomon.api:app", "--host", "0.0.0.0", "--port", "8737", "--workers", "1"]
