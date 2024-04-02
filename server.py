import shutil
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from crossvals.translate.translate import TranslateCrossValidator
from pydantic import BaseModel
from crossvals.healthcare.healthcare import HealthcareCrossval
from fastapi import UploadFile, File, HTTPException
app = FastAPI()

# Enable all cross-origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

translate_crossval = TranslateCrossValidator()
healthcare_crossval = HealthcareCrossval(netuid = 31, topk = 1)
# healthcare_crossval.run_background_thread()

class TranlsateItem(BaseModel):
    text: str
    source_lang: str = "en"
    target_lang: str = "es"
    timeout: int = None

@app.get("/")
def read_root():
    return translate_crossval.run("Hello, how are you?")


@app.post("/translate/")
def tranlsate_item(item: TranlsateItem):
    
    translate_crossval.setLang(item.source_lang, item.target_lang)
    if item.timeout:
        translate_crossval.setTimeout(item.timeout)
    return translate_crossval.run(item.text)



class ImageUpload(BaseModel):
    file: UploadFile = File(...)

# @app.post("/healthcare/")
# async def analyze_healthcare_image(image: ImageUpload):
#     print(image.file.filename)
#     file_location = f"images/{image.file.filename}"
#     with open(file_location, "wb+") as file_object:
#         file_object.write(await image.file.read())
#     result = healthcare_crossval.run(file_location)
#     return {"result": result}

@app.post("/healthcare/")
async def upload_image(image: UploadFile = File(...)):
    try:
        # Save the file to disk or process it
        with open(f"{image.filename}", "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
            result = healthcare_crossval.run(image.filename)
            # print(result)
        # You can process the file here, and then return a response
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))