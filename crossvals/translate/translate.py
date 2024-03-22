import os
import bittensor as bt
from base.synapse_based_crossval import SynapseBasedCrossval

import bittensor as bt
import pydantic
from typing import List

class Translate(bt.Synapse):
    source_texts: List[str] = pydantic.Field(..., allow_mutation=False)
    translated_texts: List[str] = []
    source_lang: str = pydantic.Field(..., allow_mutation=False)
    target_lang: str = pydantic.Field(..., allow_mutation=False)
    required_hash_fields: list[str] = pydantic.Field(  ["source_texts", "source_lang", "target_lang"], allow_mutation = False)
class TranlsateCrossValidator(SynapseBasedCrossval):
    def __init__(self, netuid = 2, wallet_name = 'NI', wallet_hotkey = 'ni', network = "finney", topk = 1):
        super().__init__(netuid, wallet_name, wallet_hotkey, network, topk)
        print([item['uid'] for item in self.top_miners])
        self.dendrite = bt.dendrite( wallet = self.wallet )
    def forward(self, text):
        translate_synapse = Translate(
            source_texts = [text],
            source_lang = "en",
            target_lang = "es"
        )
        axons = [self.metagraph.axons[i['uid']] for i in self.top_miners]
        responses = self.dendrite.query(
                    axons,
                    # Construct a scraping query.
                    translate_synapse,
                    # All responses have the deserialize function called on them before returning.
                    deserialize = False, 
                    timeout = 60
                )
        return responses
    def run(self, text):
        response = self.forward(text)
        return response


translate_crossval = TranlsateCrossValidator()
print(translate_crossval.run("Hello, how are you?"))