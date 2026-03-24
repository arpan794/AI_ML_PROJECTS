from fastapi import FastAPI
from .routes import router

app = FastAPI(title="Image Caption Generator")

app.include_router(router)