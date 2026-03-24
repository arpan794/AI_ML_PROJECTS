from fastapi import APIRouter, UploadFile
from ..ml.pipeline import ImageCaptionPipeline

router = APIRouter()

pipeline = ImageCaptionPipeline()

@router.post("/generate-caption")
async def generate_caption(file: UploadFile):

    image_bytes = await file.read()

    caption = pipeline.run(image_bytes)

    return {"caption": caption}