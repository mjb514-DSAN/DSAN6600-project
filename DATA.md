# Data

## Source
This project uses DEC (Dense Earnings Call dataset), released with:

Ding Yu, Zhuo Liu, Hangfeng He. Same Company, Same Signal: The Role of Identity in Earnings Call Transcripts. Findings of ACL 2025. [ACL Anthology](https://aclanthology.org/2025.findings-acl.946/)

- Code and labels: [Github Repo of Paper](https://github.com/piqueyd/Same-Company-Same-Signal)
- Transcripts and Embeddings: Google Drive folder linked from that repo's README

## What it Contains

`DEC.json`
- Contains Dict of 1800 transcripts

`Embeddings/openai/DEC.npz`
- Contains OpenAI transcript embeddings with matching ID

`Embeddings/openai/DEC2RandomTicker.npz`
- Contains the control: one random vector per company (identity, no content)

`Embeddings/openai/DECRandomAll.npz`
- Coontains control: one random vector per call (no identity, no content)

`DEC.csv`
- Contains 1800 rows x 251 columns: sector, ticker, call date, before / after market flag, returns, rolling log volatility labels

## Size
1800 earnings call from 90 US companies. 20 calls per company from 2019-2023. 1,255 calls were released before market open and 545 after the close.

All files share the same 1,800 IDs.

## License and Usage
From the repository's `LICENSE-DATA`:
- Labels, metadata, embeddings: CC BY 4.0. They can be used and shared with attribution to the paper
- Transcript text: Third party material from Seeking Alpha. Use is limited to academic research and reproduction of results. Because of this, no data files are committed to this repository

## How to Get the Data
1. Download `DEC.json` and `Embeddings` folder from the Google Drive folder linked in the repo README
2. Download `DEC.csv` from `SameCompanySameSignal/dataset/DEC.csv`
3. Running `notebooks/EDA.ipynb` created the cleaned table `data/processed/dec_clean.csv`

## Label Definitions
There are two volatility columns `lv{w}_future_{k}` and `lv{w}_past_{k}`. These are rolling log volatilities over a w day window of daily returns ending on day k.

The fully post-call 3 day target is `lv3_future_3`