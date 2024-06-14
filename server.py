from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import bittensor as bt

from crossvals.prompting.text import PromtingCrossValidator
from crossvals.image_alchemy.alchemy import ImageAlchemyCrossVal
from crossvals.sybil.sybil import SybilCrossVal
from crossvals.openkaito.openkaito import OpenkaitoCrossVal
from crossvals.itsai.itsai import ItsaiCrossVal
# from crossvals.niche.niche import NicheCrossVal
from crossvals.wombo.wombo import WomboCrossVal
from crossvals.wombo.protocol import ImageGenerationClientInputs
from crossvals.fractal.fractal import FractalCrossVal
from crossvals.llm_defender.llm_defender import LLMDefenderCrossVal
from crossvals.subvortex.subvortex import SubvortexCrossVal
from crossvals.cortex.cortex import CortexCrossVal
from crossvals.snporacle.snporacle import SnporacleCrossVal
from crossvals.compute.compute import ComputeCrossVal
from crossvals.bitagent.bitagent import BitagentCrossVal
from crossvals.omegalabs.omegalabs import OmegalabsCrossVal
from crossvals.vision.vision import VisionCrossVal
from crossvals.omron.omron import OmronCrossVal
from crossvals.sturdy.sturdy import SturdyCrossVal

from fastapi import UploadFile, File, HTTPException, Body
import asyncio
from pydantic import BaseModel
from typing import List, Annotated

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TranlsateItem(BaseModel):
    text: str
    source_lang: str = "en"
    target_lang: str = "es"
    timeout: int = None
class PropmtItem(BaseModel):
    roles: list[str] = ["user", "assistant"]
    messages: list[str] = [
        "What's the weather like today?",
        "The weather is sunny with a high of 25 degrees.",
        "Could you set a reminder for me to take my umbrella tomorrow?",
        "Reminder set for tomorrow to take your umbrella.",
        "Thank you! What time is my first meeting tomorrow?",
        "Your first meeting tomorrow is at 9 AM.",
        "Can you play some music?",
        "Playing your favorite playlist now.",
        "How's the traffic to work?",
        "Traffic is light, it should take about 15 minutes to get to work."
    ]

class ImageItem(BaseModel):
    imageText: str
class SybilItem(BaseModel):
    sources: str
    query: str

class OpenkaitoItem(BaseModel):
    query: str

class ItsaiItem(BaseModel):
    texts: List[str]

# class NicheItem(BaseModel):
#     model_name: str
#     prompt: str

class WomboItem(BaseModel):
    watermark: bool
    prompt: str

class FractalItem(BaseModel):
    query: str

class AudiogenItem(BaseModel):
    type: str
    prompt: str

class LLMDefenderItem(BaseModel):
    analyzer: str

class TranscriptionItem(BaseModel):
    type: str
    audio_url: str
    audio_sample: bytes

class CortexItem(BaseModel):
    type: str
    provider: str
    prompt: str

class ComputeItem(BaseModel):
    difficulty: int

class OmegalabsItem(BaseModel):
    query: str

class VisionItem(BaseModel):
    type: str
    prompt: str
    width: Optional[int] = 1024
    height: Optional[int] = 1024
    init_image: Optional[str] = ''

@app.post("/image-alchemy/", tags=["Mainnet"])
def image_generate(item: ImageItem):
    return imagealchemy_crossval.run(item.imageText)

@app.post("/sybil/", tags=["Mainnet"])
def sybil_search(item: SybilItem):
    return sybil_crossval.run({'sources': item.sources, 'query': item.query})

@app.post("/openkaito/", tags=["Mainnet"])
async def openkaito_search(item: OpenkaitoItem):
    return await openkaito_crossval.run(item.query)

@app.post("/itsai/", tags=["Mainnet"])
async def llm_detection(item: ItsaiItem):
    return await itsai_crossval.run(item.texts)

# @app.post("/niche/", tags=["Mainnet"])
# def niche_generation(item: NicheItem):
#     return niche_crossval.run(item)

@app.post("/wombo/", tags=["Mainnet"])
async def generate(item: WomboItem):
    print(item)
    return await wombo_crossval.run(ImageGenerationClientInputs(prompt=item.prompt, watermark=item.watermark))

@app.post("/fractal", tags=["Mainnet"])
def fractal_research(item: FractalItem):
    return fractal_crossval.run(item.query)

@app.post("/llm-defender", tags=["Mainnet"])
def llm_defender(item: LLMDefenderItem):
    return llmdefender_crossval.run({"analyzer": item.analyzer})

@app.post("/subvortex/", tags=["Mainnet"])
async def subvortex_calc():
    return await subvortex_crossval.run()

@app.post("/cortex", tags=["Mainnet"])
async def cortex_api(item: CortexItem):
    return await cortex_crossval.run(item)

@app.post("/snporacle/", tags=["Mainnet"])
def snporacle():
    return snporacle_crossval.run()

@app.post("/compute/", tags=["Mainnet"])
async def compute(item: ComputeItem):
    return compute_crossval.run(item)

@app.post("/bitagent/", tags=["Mainnet"])
def bitagent():
    return bitagent_crossval.run()

@app.get("/omegalist/", tags=["Mainnet"])
async def omega_list():
    return await omegalabs_crossval.get_topic()

@app.post("/omegalabs/", tags=["Mainnet"])
async def omegalabs(item: OmegalabsItem):
    return await omegalabs_crossval.run(item)

@app.post("/vision/", tags=["Mainnet"])
async def vision(item: VisionItem):
    return await vision_crossval.run(item)

@app.post("/omron/", tags=["Mainnet"])
def omron_func():
    return omron_crossval.run()

@app.post("/sturdy", tags=["Mainnet"])
async def sturdy():
    return await sturdy_crossval.run()

@app.post("/prompting/", tags=["Mainnet"])
async def prompting(item: PropmtItem):
    return await promtingCrossval.run(item)

subtensor = bt.subtensor(network = "local")
promtingCrossval = PromtingCrossValidator(subtensor=subtensor)
imagealchemy_crossval = ImageAlchemyCrossVal(subtensor=subtensor)
sybil_crossval = SybilCrossVal(subtensor=subtensor)
openkaito_crossval = OpenkaitoCrossVal(subtensor=subtensor)
itsai_crossval = ItsaiCrossVal(subtensor=subtensor)
# niche_crossval = NicheCrossVal(subtensor=subtensor)
wombo_crossval = WomboCrossVal(subtensor=subtensor)
fractal_crossval = FractalCrossVal(subtensor=subtensor)
llmdefender_crossval = LLMDefenderCrossVal(subtensor=subtensor)
subvortex_crossval = SubvortexCrossVal(subtensor=subtensor)
snporacle_crossval = SnporacleCrossVal(subtensor=subtensor)
compute_crossval = ComputeCrossVal(subtensor=subtensor)
bitagent_crossval = BitagentCrossVal(subtensor=subtensor)
omegalabs_crossval = OmegalabsCrossVal(subtensor=subtensor)
cortex_crossval = CortexCrossVal(subtensor=subtensor)
vision_crossval = VisionCrossVal(subtensor=subtensor)
omron_crossval = OmronCrossVal(subtensor=subtensor)
sturdy_crossval = SturdyCrossVal(subtensor=subtensor)