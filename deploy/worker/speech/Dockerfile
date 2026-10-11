# Speech verification image. Build from the repository root:
#   docker build -f deploy/worker/speech/Dockerfile -t ai-media-engine-speech-worker:unverified .
#
# Do not retag or replace the Qwen-Image worker. That image stays at
# deploy/worker/Dockerfile and
# ghcr.io/mehmetsentc/ai-media-engine-worker@sha256:be2c96a09947b15ff1fd8fb1d2aaa4692dcc9c06083ef1486674ac4fbc4a907c
#
# qwen-tts 0.1.1 requires transformers==4.57.3 and accelerate==1.12.0.
# The image worker pins transformers==5.19.0 and accelerate==1.15.0.
# This file is a separate build. It has not been built or pushed.
#
# PyPI ctranslate2 4.8.2 does not publish a wheel whose filename names CUDA.
# The first GPU session must observe CUDA support before accepting a transcript.
# Weights are not copied into this image. They mount at /models/speech.
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

RUN rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED \
    && apt-get update \
    && apt-get install -y --no-install-recommends sox \
    && rm -rf /var/lib/apt/lists/*

COPY deploy/worker/speech/requirements.txt /tmp/speech-requirements.txt
RUN python3 -m pip install --no-cache-dir --disable-pip-version-check \
      -r /tmp/speech-requirements.txt \
    && python3 -c "import importlib.metadata as meta; assert meta.version('faster-whisper')=='1.2.1'; assert meta.version('qwen-tts')=='0.1.1'; assert meta.version('transformers')=='4.57.3'"

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    SPEECH_WEIGHTS=/models/speech

WORKDIR /workspace
CMD ["python3", "-c", "import importlib.metadata as meta; print(meta.version('faster-whisper'), meta.version('qwen-tts'))"]
