from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import base64
import os
import traceback
import nest_asyncio

from layoutgenerator import TextToLayoutPipeline
from src.visualization import Visualizer
from src.generators.copy_generator import generate_copies

nest_asyncio.apply()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발/테스트는 * (운영 땐 도메인 지정)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATASET_NAME = "cardnews"
pipeline = TextToLayoutPipeline(dataset=DATASET_NAME)
visualizer = Visualizer(dataset=DATASET_NAME)

@app.post("/api/cardnews")
async def generate_cardnews(
    query: str = Form(...),
    images: list[UploadFile] = File([]),
    logo: UploadFile = File(None),
):
    try:
        images_dir = "src/images"
        os.makedirs(images_dir, exist_ok=True)
        image_filenames = []
        for file in images:
            filename = file.filename
            filepath = os.path.join(images_dir, filename)
            with open(filepath, "wb") as f:
                f.write(await file.read())
            image_filenames.append(filename)

        logo_data = None
        if logo:
            logo_path = os.path.join(images_dir, logo.filename)
            with open(logo_path, "wb") as f:
                f.write(await logo.read())
            with open(logo_path, "rb") as f:
                logo_data = f.read()

        # 1. 레이아웃 생성
        results = pipeline.run(user_text=query, num_images=len(image_filenames))

        # 2. 카피(문구) 생성
        layout_lists = [list(sample.keys()) for sample in pipeline.map_labels_to_bboxes(results)]
        copy_result = generate_copies(layout_lists, query)

        # 3. 시각화(완성본 이미지 생성)
        visualizer.visualize(
            results,
            copy=copy_result,
            image_filenames=image_filenames,
            logo_data=logo_data, 
            show_bbox=False
        )

        output_image_path = "output/output_poster.png"
        if not os.path.exists(output_image_path):
            return JSONResponse(status_code=500, content={"status_message": "이미지 생성 실패"})

        with open(output_image_path, "rb") as img_file:
            img_bytes = img_file.read()
        image_base64 = base64.b64encode(img_bytes).decode("utf-8")
        return {
            "image_base64": image_base64,
            "status_message": "성공",
        }
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"status_message": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=7860)
