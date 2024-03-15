from abc import ABC, abstractmethod
from crossval import BaseCrossval
import bittensor as bt
class SynapseBasedCrossval(BaseCrossval):
    def __init__(self, netuid = 1, wallet_name = None, wallet_hotkey = None, network = "finney", topk = 10):
        super().__init__(netuid, wallet_name, wallet_hotkey, network, topk)
    
    @abstractmethod
    def run(self):
        ...
    
    @abstractmethod
    def forward(self):
        ...
