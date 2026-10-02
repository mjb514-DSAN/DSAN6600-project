# Check in 1: Earnings Calls and Volatility Surprise

## Problem Framing and Scope
My project is an extension of "Same Company, Same Signal: The Role of Identity in Earnings Call Transcripts" by Ding Yu, Zhuo Liu, Hangfeng He. Their research used text embedding models to predict future volatility of the company's stock. They concluded that transcript based models do not perform better than simply using the company's average past volatility. Additionally, they found that the transcript embeddings mainly encode which company is speaking. I wanted to expand on this framework and test whether there is additional signal from these calls that would indicate changes in volatility.

### Question
The question that this problem attempts to answer: Do earnings call transcripts predict post earnings volatility beyond the company's own history? 

### Motivation
I chose this project because I have prior work experience in Quantitative Finance at Bank of America where I was tasked with using text embedding models to categorize transactions. I wanted to combine my experience with NLP and the materials of this course to attempt to understand if models in the financial markets could be improved, such as the volatility text embedding model used in the paper.

### Task
This task will involve a scalar regression from text, using MSE as the evaluation metric. The transcripts serve as inputs, and the target is volatility surprise. Volatility surprise is the actual 3 day log volatility minus the company's average from past calls, adjusted by quarter. 

### Success Criteria
Success in this project is to see if it's possible to beat the company baseline out of sample on 2023 (baseline produced an MSE of 0.525 on the quarter adjusted surprise). Additionally, success would be beating the random-ticker embeddings, which carry company identity but no content.

### Scope
- Floor: MLP heads on the provided embeddings with time based evaluation
- Target: my own Q&A aware embeddings and inputs with company identity removed
- If possible: Fine tuning a pretrained encoder; an option-implied volatility baseline if WRDS data access is available

## Data
This project uses DEC (Dense Earnings Call dataset), released by Yu et al. (2025). The data contains 1800 calls from 90 US companies (20 calls per company) across 11 sectors. The calls span from 2019-2023. The data from these calls includes three parts that join on a shared call ID:
- Transcripts (DEC.json): full call text, with a median of 9,400 words
- Embeddings (.npz): the paper's OpenAI embeddings (1,800 x 3072), plus two controls: random-ticker which only has the identity of the company and random-all which has neither identity nor content from the calls.
- Labels (DEC.csv): sector, call date, if the call was before open or after close, daily returns around each call, and volatility labels


### Access and Licence
The transcripts and embeddings come from the paper's Google Drive; DEC.csv comes from their Github. Steps to access the data are in [DATA.md](DATA.md)

Labels, metadata, and embeddings are released under CC BY 4.0. Transcripts are third party material (sourced from Seeking Alpha) and limited to academic use, so no data is committed to this repository (data/ is gitignored)

### Data Cleaning
The following issues were resolved during the data cleaning process
- Fixed typos of sector names so that they were standardized
- 8 PepsiCo transcripts contained the call text more than once so I removed the extra call text
- Flagged short calls (under 4,000 words)
- Used call dates for all train / validation / test splits
- Saved a cleaned table in: 
   `data/processed/dec_clean.csv`
- Ensured there were no missing values, exactly 20 calls per company, no duplicate IDs, and no empty transcripts

### Limitations of the Data
- Only uses large cap companies with 5 years of full coverage. Companies were selected from the top holdings of sector ETFs. Firms that shrank, were acquired, or had gaps are excluded. This is a form of survivorship bias, and results of this project may not generalize to smaller companies
- 70% of calls were released before market open. Release timing determines which trading day is the first post call day, and after close calls are more volatile on average
- The provided embeddings were computed on raw text, so they include the PepsiCo duplicates 

## EDA and Data Audit

### Label Verification
- Recomputed the volatility labels from raw daily returns and matched the stored values. The label is the log of the standard deviation over a window ending on day k. So `lv3_future_1` mixes pre and post call days, and `lv3_future_3` is the true 3 day post call target

### Summary Stats and Samples
- Transcripts: Median of 9,400 words (lower quartile  of 8,550 and upper quartile of 10,550). The range is 2,200 to 27,800.
- Target: mean 3 day log volatility is -4.14 with a standard deviation of 0.77. 
- Representative Sample: Sampled a portion of the Q&A from a Wells Fargo call to show what the model reads

### Key Distributions and Patterns
- Event study: Volatility jumps about 1.7x once the window includes the first post call day
- Company average volatilities differ, but post call volatilities from the same vompany vary even more. Knowing the company helps the model, but most of the variation is quarter to quarter.
- Baseline: The company's past average reproduces the paper's baseline (MSE of 0.54) but explains only 9% of the variance of volatility. In this project, I will try to determine what of the 91% is the noise floor and what can actually be modeled.
- Mean surprise volatility (the additional volatility compared to the company average) moves by quarter. It was positive in early 2020 and 2022, and negative in the 3rd quarter of 2021. This is why I am using a quarter adjustment.
- Using the embeddings that the OpenAI model produced, calls from the same company are much more similar than calls across different companies
- Utilities are the calmest sector while tech is the most volatile. Additionally, after close calls are more volatile on average than before open calls.

### Artifacts, Biases, and Failures
- Because the 3 day label measures spread around the window mean, steady directional moves score as calm. For example, Wells Fargo rose roughly 1.8% three days in a row (total of up 5.4%) and was labeled as the calmest call. 
- Outliers have mixed causes: I identified a few possible causes for some of the outliers that I found. COVID and oil shocks starting mid quarter (ORCL March 2020), noisy early baselines (JPM 2019), and some that look driven by the call itself (CRM August 2020).
- Distribution Shifts: The validation year 2022 is more volatile than the training or test.
- Potential failures of the model: The model learns company identity instead of call content or learns the time period


## Early Probe of the Data
I ran non-neural models on the on the same split and metrics that the neural models will use to train:
- Train: 2019-2021
- Validation: 2022
- Test: 2023

The baseline models were as follows, with the metric being test MSE:
- Global average of all companies: .624
- Company's past average: .541
- Company average + pre-call volatility: .564

The predicting the company's average volatility is the best model.

Additionally, I ran a logistic regression that trained a classifier on the 2019-2021 call embeddings to guess which company or sector each call came from and tested it on 2023 calls. It predicted the company with 1.00 accuracy and sector with .99 accuracy.

These models show that the embeddings almost perfectly encode which company is speaking. 

## Evaluation Plan*
### Splits
Training data will be 2019-2021. This is 990 calls. The validation will be 2022, 360 calls. Lastly, I will test on 2023, 360 calls.

### Target
The target is the quarter adjusted volatility surprise. 

### Metrics
I will use MSE on the surprise, compared with the zero surprise baseline.

### What I count as a Win
A win in this project would be to beat the zero-surprise baseline and the random-ticker embeddings. This means the model would go beyond just identify the company and actually capture signal from the call content.

# Plan for Check in 2
For check in 2, I want to build an MLP on top of the OpenAI model. The arichtecture will be 3,072 - 64 - 1. One hidden layer with ReLU and dropout, and a single output the predicted surprise volatility. I will run it on all three embedding sets and compare them using MSE.

Additionally, I want to remove company identity from the input. A potential way that I could do this is use a call's embedding minus the average of that company's past call embeddings. That will capture what's different about this call compared with how the company usually sounds.

## AI Disclosure
I used Claude in the following ways:
- Research: Checking if any papers had been written expanding on the original paper. Additionally, used it to understand different ways I could compare the calls (discussed the application of cosine similarity)
- Debugging: Debugging code when analyzing the transcripts
- Title Names and Markdown Documentation: Used it for Markdown formatting in check-in-1.md (adding backticks and changing section titles to be more accurate)


## References

Yu, D., Liu, Z., & He, H. (2025). Same Company, Same Signal: The Role of Identity in Earnings Call Transcripts. *Findings of the Association for Computational Linguistics: ACL 2025.* https://aclanthology.org/2025.findings-acl.946/