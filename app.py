"""Grand Challenge invoke server for DoseRAD photon dose on CT."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Response, status
import uvicorn

import inference


RUNTIME: dict[str, inference.RuntimeModel] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Loading before /health returns 200 keeps model-loading time outside the
    # measured invoke duration.
    RUNTIME["model"] = inference.load_runtime_model()
    yield
    RUNTIME.clear()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    code = status.HTTP_200_OK if "model" in RUNTIME else status.HTTP_503_SERVICE_UNAVAILABLE
    return Response(status_code=code)


@app.post("/invoke")
async def invoke():
    inference.run(RUNTIME["model"])
    return Response(status_code=status.HTTP_201_CREATED)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4743)

