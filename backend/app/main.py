from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.predict import router as predict_router
from services.model import initialize_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        initialize_model()
    except FileNotFoundError as exc:
        raise RuntimeError(str(exc)) from exc

    yield


app = FastAPI(title="Medical Image Classifier Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    # Allow access from your LAN IP, e.g. http://192.168.x.x:3000
    allow_origin_regex=r"http://192\.168\.\d{1,3}\.\d{1,3}:3000",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router)


@app.get("/")
def read_root():
    return {"message": "Medical image classifier backend is running"}


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "medical-image-classifier-backend",
    }
