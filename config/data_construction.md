# Document scores and source populations

The numerical artifacts contain token-average likelihood-difference scores for held-out query documents. Query documents are excluded from both the audited fine-tuning update and checkpoint validation. Source participation refers to contribution to that update.

## Data sources

| Domain | Source unit | Document | Dataset |
|---|---|---|---|
| Reddit | Account | Comment | [Pushshift Reddit comments](https://huggingface.co/datasets/fddemarco/pushshift-reddit-comments), Baumgartner et al. (2020) |
| CC News | Web domain | News article | [CC News](https://huggingface.co/datasets/vblagoje/cc_news), Common Crawl; Hamborg et al. (2017) |
| Enron | Mailbox owner | Sent email | [CMU Enron Email Dataset](https://www.cs.cmu.edu/~enron/), May 7, 2015 release; Klimt and Yang (2004) |
| Chromium | Developer | Code-review comment | [ConvoKit Chromium corpus](https://convokit.cornell.edu/documentation/chromium.html), Meyers et al. (2018), Chang et al. (2020) |

This package distributes numerical scores, not the original document text or source identifiers. Original datasets retain their respective terms. The Chromium corpus is distributed under CC BY 4.0; its comments are processed as described below.

## Query construction

Each source uses at most 16 eligible query documents. Query bags are identical across all aggregation methods and source-role partitions.

- **Reddit:** each account contributes text from two communities. Participating accounts supply eight training documents per community, and query bags contain eight documents from each community. Views retain 8–511 content tokens and append EOS.
- **CC News:** each query score uses the first at most 511 content tokens of an article, with EOS appended. An article is excluded if a candidate segment among its first 2,044 content tokens equals a complete token input in the training-candidate or validation pool, including unused training candidates. This is exact-input exclusion. Fixed segments contain 511 tokens, with a final remainder retained only when it contains at least eight tokens. Paragraph segments use tokenizer-aligned newline boundaries: take the earliest boundary within 8–511 tokens that leaves zero or at least eight tokens; otherwise use the remaining suffix if it fits, or a 511-token block shortened if needed to leave at least eight tokens. EOS is included in segment comparisons. All candidate segments are checked before the first-block query view is scored.
- **Enron:** within each mailbox, the canonical sender is the most frequent parsed From address containing the surname inferred from the mailbox name, with lexical tie-breaking. Only that sender's messages in the owner's mailbox are retained; repeated canonical addresses define one source. Line-ending-normalized bodies are deduplicated within sources, and exact bodies shared by multiple sources are removed from all of them. Sources need at least four remaining documents. Query views retain at most 511 content tokens and append EOS. Exact input matches against update and validation documents are excluded before applying the query budget.
- **Chromium:** sources are developer accounts and documents are comments. Quote-prefixed lines are removed. Views contain 65–257 content tokens, corresponding to 64–256 next-token predictions, without appended EOS. Each participating source contributes 16 training documents. Each source's 16 query documents are drawn from its 32 eligible queries.

CC News has 2,087 sources with fewer than 16 queries and a median bag size of six. Enron has 51 sources with fewer than 16 queries and a median bag size of 16. Every Reddit and Chromium source has 16 queries.

## Models and scores

The initial model in every domain is `EleutherAI/gpt-neo-125m`. Fine-tuning uses Adam, learning rate 5e-5, batch size eight, betas (0.9, 0.999), epsilon 1e-8, and zero weight decay. There are 200 warmup updates followed by linear decay over ten epochs. Documents are not packed; gradient accumulation and clipping are not used. Training uses bfloat16 autocast and float32 optimizer state.

The updated checkpoint is the epoch endpoint with minimum validation token-average negative log-likelihood, breaking ties in favor of the earlier endpoint. Validation uses held-out documents from participating sources.

| Domain | Training documents/epoch | Prediction tokens/epoch | Validation documents | Checkpoint epoch | Updates |
|---|---:|---:|---:|---:|---:|
| Reddit | 1,344 | 62,965 | 336 | 3 | 504 |
| CC News | 283,370 | 95,590,331 | 17,191 | 9 | 318,798 |
| Enron | 42,917 | 10,742,290 | 2,636 | 5 | 26,825 |
| Chromium | 800 | 84,258 | 200 | 2 | 200 |

For identical predicted tokens and contexts, the document score is the mean of `log p_updated(token | context) - log p_initial(token | context)`. Target and initial models use identical token inputs and prediction masks. The four data files preserve these scores without rounding.

Running `reproduce.py` recomputes reference tails, aggregation, calibration, and evaluation from these scores. Training checkpoints and raw-text processing are not required for this calculation and are not included in this package.
