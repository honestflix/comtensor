from fastapi import FastAPI
from crossvals.translate.translate import TranslateCrossValidator
from pydantic import BaseModel
app = FastAPI()
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
