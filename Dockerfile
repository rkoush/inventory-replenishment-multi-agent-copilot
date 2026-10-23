FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
# Bake the synthetic DB and policy index into the image so containers start fast
RUN python -m scripts.generate_data && python -m rag.build_index
EXPOSE 8000
CMD ["python", "-m", "mcp_servers.inventory_server"]
