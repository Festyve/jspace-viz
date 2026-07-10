# Deploy jspace-viz as a website (e.g. a Hugging Face Space, SDK: docker).
# The free CPU tier comfortably runs the gpt2 preset with its prebaked lens.
FROM python:3.12-slim
WORKDIR /app
COPY . .
# CPU-only torch wheel (the default pulls multi-GB CUDA deps on Linux)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -e .
# HF Spaces provides a writable /tmp; model + lens download there on first boot.
ENV HF_HOME=/tmp/hf
EXPOSE 7860
CMD ["jspace-viz", "--preset", "gpt2", "--host", "0.0.0.0", "--port", "7860"]
