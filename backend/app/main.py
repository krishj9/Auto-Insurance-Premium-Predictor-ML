from contextlib import asynccontextmanager
import logging
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api.routes import PredictionError, router, set_prediction_service
from backend.app.services.predictor import PredictionService


log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    service = PredictionService()
    try:
        service.load()
        log.info("Prediction service loaded successfully.")
    except FileNotFoundError as exc:
        log.warning("Prediction service started without champion artifacts: %s", exc)
    set_prediction_service(service)
    try:
        yield
    finally:
        set_prediction_service(None)


app = FastAPI(title="Auto Insurance Premium API", lifespan=lifespan)


@app.exception_handler(PredictionError)
async def prediction_error_handler(_request: Request, exc: PredictionError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
            }
        },
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)
