import bittensor as bt
import asyncio
import os
from constants import BASE_DIR
from base.commit_based_crossval import CommitBasedCrossval
from huggingface_hub import snapshot_download
class HealthcareCrosscal(CommitBasedCrossval):
    def __init__(self, netuid = 1, wallet_name = None, wallet_hotkey = None, network = "finney", topk = 10, subtensor = None):
        super().__init__(netuid, wallet_name, wallet_hotkey, network, topk, subtensor)
        self.local_dir = os.path.join("crossvals/healthcare/model")
        self.cache_dir = os.path.join(BASE_DIR, "crossvals/healthcare/cache")
        print([item['uid'] for item in self.top_miners])
    def run(self):
        commits = []
        for miner_axon in self.top_miners:
            commit = self.getCommit(miner_axon=miner_axon)
            print(f"Commit from miner {miner_axon['uid']}: {commit}")
            commits.append(commit)
        for commit in commits:
            asyncio.run(self.download_model(commit))
            
    async def download_model(self, commit):
        try:
            repo_id = commit.split(' ')[0]
            commit_hash = commit.split(' ')[1]
            snapshot_download(repo_id = repo_id, revision = commit_hash, local_dir = self.local_dir, cache_dir = self.cache_dir)
            bt.logging.success(f"Model downloaded successfully")
        except Exception as e:
            bt.logging.error(f"Error occured while downloading model: {e}")

if __name__ == "__main__":
    healthcareCrossval = HealthcareCrosscal(netuid = 31, topk=1)
    healthcareCrossval.run()

    