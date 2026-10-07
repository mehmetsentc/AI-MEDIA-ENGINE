# AI-MEDIA-ENGINE image worker runtime.
# Build from the repository root:
#   docker build -f deploy/worker/Dockerfile -t ai-media-engine-image-worker:runtime .
#
# This image is runtime only. Qwen weights are not copied in.
# They are mounted at /models (warm volume) or staged from R2 at start.
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

# This runtime image ships torch on Debian Python and leaves the PEP 668 marker in place.
# Removing that marker is the upstream image's own container install path.
RUN rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED \
    && python3 -m pip install --no-cache-dir --disable-pip-version-check \
      "diffusers==0.41.0" \
      "transformers==5.19.0" \
      "accelerate==1.15.0" \
      "safetensors==0.8.0" \
      pillow \
    && python3 -c "import torch,diffusers,transformers,accelerate,safetensors"

COPY src/media_engine/engines/image/qwen_remote.py /workspace/qwen_remote.py
COPY src/media_engine/engines/image/model_cache.py /workspace/model_cache.py
COPY src/media_engine/engines/image/r2_cache.py /workspace/r2_cache.py
COPY deploy/worker/worker_ready.py /workspace/worker_ready.py

RUN mkdir -p /opt/ai-media-engine \
    && touch /opt/ai-media-engine/runtime-ready \
    && python3 -c "import torch,diffusers,transformers,accelerate,safetensors,pathlib; assert pathlib.Path('/usr/local/bin/python3').exists()"

ENV HF_HOME=/models/hf-cache \
    HUGGINGFACE_HUB_CACHE=/models/hf-cache \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    WORKER_PORT=8080

EXPOSE 8080
WORKDIR /workspace
CMD ["python3", "/workspace/worker_ready.py"]
