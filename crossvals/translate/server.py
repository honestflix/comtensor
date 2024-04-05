from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from crossvals.translate.translate import TranslateCrossValidator
from pydantic import BaseModel

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
