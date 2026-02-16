# Computational Psycholinguistics — Assignment 3

This repo contains my **report**, **analysis script**, and **generated outputs** for Assignment 3 on Word Processing and Frequency Ordered Bin Search (FOBS).

## Contents

* `script.py` - main Python script to reproduce all results (plots, tables, summary).
* `report.pdf` - final report.
* `outputs/` - generated figures, tables, and logs from running `script.py`.
* `word-processing.pdf` - assignment document.

## Dataset (NOT included)

The dataset files are not included in this repository.
First, download the English “Natural Stories corpus” via the link provided below:
• Corpus Repository: https://github.com/languageMIT/naturalstories
Place the required dataset files in the **same folder as `script.py`** before running.

Required files:

* `processed_RTs.tsv`
* `freqs-1.tsv`
* `all_stories_gpt3.csv`

## How to run

1. Install dependencies:

   ```bash
   pip install numpy pandas matplotlib scipy statsmodels nltk
   ```

2. Run the script:

   ```bash
   python script.py
   ```

3. Outputs will be written to:

   * `outputs/plots/` (all figures)
   * `outputs/tables/` (CSV tables)
   * `outputs/logs/summary.txt` (key statistics + model comparisons)
   * `outputs/all_plots.pdf` (all plots combined)



* The script downloads required NLTK resources automatically on first run (WordNet + POS tagger).
* GPT-3 surprisal is computed as **−log(p)** using `logprob` from `all_stories_gpt3.csv`.
