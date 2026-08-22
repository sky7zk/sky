FROM --platform=linux/amd64 pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    DOSERAD_BATCH_SIZE=1

RUN groupadd --system user && useradd --create-home --no-log-init --system --gid user user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/
RUN python -m pip install --no-cache-dir --no-color --requirement /opt/app/requirements.txt

COPY --chown=user:user app.py inference.py /opt/app/
COPY --chown=user:user doserad_v2 /opt/app/doserad_v2

USER user

LABEL org.grand-challenge.api-method="invoke"

ENTRYPOINT ["python", "app.py"]
