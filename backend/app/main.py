from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .detection import detect_auto_color_blocks, detect_objects

app = FastAPI(title="CountSnap API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/detect")
async def detect(
    image: UploadFile = File(...),
    mode: str = Form("color"),
    min_area: int = Form(900),
    threshold: int = Form(0),
    blur: int = Form(7),
    invert: bool = Form(True),
) -> dict:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="请上传图片文件。")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="上传的图片是空文件。")

    try:
        if mode in {"auto_color_blocks", "color"}:
            return detect_auto_color_blocks(
                image_bytes=image_bytes,
                min_area=min_area,
            )

        return detect_objects(
            image_bytes=image_bytes,
            min_area=min_area,
            threshold=threshold,
            blur=blur,
            invert=invert,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
