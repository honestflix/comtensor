# Bittensor Subnet 2 - Bittranslate

We just added translate feature of bittensor to comtensor.

## Running translate.py script locally

1. Install dependancies

    ```bash
    pip install -r crossvals/translate/requirements.txt
    ```

2. Check if you got validator keys.

    You need validator keys to query bittensor miners. Miners have blacklist_fn to block various attacs due to unnecessary requests.

    If you have keys, you need to replace in `translate.py` code. (`wallet_name`, `wallet_hotkey`)

    ```python

    class TranslateCrossValidator(SynapseBasedCrossval):
        def __init__(self, netuid = 2, wallet_name = 'NI', wallet_hotkey = 'ni', network = "finney", topk = 1):
            ...
    ```

3. Running script.
    ```bash
    python crossvals/translate/translate.py
    ```

## Running http server using fastapi

```bash
uvicorn crossvals.translate.server:app --reload
```

-------

### How to test

```bash
curl -X 'POST' \
    'http://127.0.0.1:8000/translate/' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "text": "In this example, Item is a Pydantic model that defines the structure of the request body. FastAPI automatically validates incoming requests to ensure they match the model.",
    "target_lang": "zh"
    }'

```