import os
import re
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, ttest_ind
import statsmodels.formula.api as smf

# ---------- NLTK setup (standard lemmatizer + POS tagger) ----------
def ensure_nltk():
    import nltk

    # corpora needed for WordNet lemmatizer
    for pkg in ["wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)

    # tagger name differs across NLTK versions
    tagger_candidates = [
        ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
        ("averaged_perceptron_tagger",     "taggers/averaged_perceptron_tagger"),
    ]
    for pkg, path in tagger_candidates:
        try:
            nltk.data.find(path)
            return
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
                nltk.data.find(path)
                return
            except Exception:
                pass

    raise RuntimeError(
        "NLTK POS tagger not found even after download. "
        "Try running: python -c \"import nltk; nltk.download('averaged_perceptron_tagger_eng')\""
    )


def penn_to_wn_pos(penn_tag: str):
    # Map Penn Treebank tag -> WordNet tag
    if not penn_tag:
        return "n"
    t = penn_tag[0].upper()
    if t == "J":
        return "a"  # adj
    if t == "V":
        return "v"
    if t == "N":
        return "n"
    if t == "R":
        return "r"  # adv
    return "n"

def clean_token(w: str) -> str:
    w = str(w)
    w = w.strip()
    # keep apostrophes in contractions; remove other punctuation
    w = re.sub(r"[^A-Za-z']+", "", w)
    return w.lower()

def is_punct_or_empty(w: str) -> bool:
    cw = clean_token(w)
    return cw == ""

# ---------- Plot helpers ----------
def save_fig(fig, outpath: Path):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

def scatter_with_binned_line(x, y, xlabel, ylabel, title, outpath: Path, bins=30):
    # robust binning: quantiles
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) == 0:
        return
    try:
        df["qbin"] = pd.qcut(df["x"], bins, duplicates="drop")
        b = df.groupby("qbin", as_index=False).agg(x_mean=("x", "mean"), y_mean=("y", "mean"))
    except Exception:
        # fallback: equal-width bins
        df["qbin"] = pd.cut(df["x"], bins=bins)
        b = df.groupby("qbin", as_index=False).agg(x_mean=("x", "mean"), y_mean=("y", "mean"))

    fig = plt.figure(figsize=(7, 5))
    plt.scatter(df["x"], df["y"], s=10, alpha=0.12)
    plt.plot(b["x_mean"], b["y_mean"], linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    save_fig(fig, outpath)

def pred_vs_obs_plot(obs, pred, title, outpath: Path):
    df = pd.DataFrame({"obs": obs, "pred": pred}).dropna()
    fig = plt.figure(figsize=(7, 5))
    plt.scatter(df["obs"], df["pred"], s=10, alpha=0.15)
    plt.xlabel("Observed mean RT (ms)")
    plt.ylabel("Predicted mean RT (ms)")
    plt.title(title)
    save_fig(fig, outpath)

# ---------- Main pipeline ----------
def main():
    root = Path(".").resolve()
    RT_PATH = root / "processed_RTs.tsv"
    FREQ1_PATH = root / "freqs-1.tsv"
    GPT3_PATH = root / "all_stories_gpt3.csv"

    out_dir = root / "outputs"
    plots_dir = out_dir / "plots"
    tables_dir = out_dir / "tables"
    logs_dir = out_dir / "logs"
    for d in (plots_dir, tables_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ---------------- Load RTs (mean RT per word position) ----------------
    rt = pd.read_csv(RT_PATH, sep="\t")
    # mean RT over subjects per (item, zone, word)
    token_means = (
        rt.groupby(["item", "zone", "word"], as_index=False)["RT"]
          .mean()
          .rename(columns={"RT": "mean_RT"})
    )
    token_means["token_code"] = token_means["item"].astype(str) + "." + token_means["zone"].astype(str) + ".word"

    # ---------------- Load unigram frequencies (freqs-1.tsv) ----------------
    f1 = pd.read_csv(
        FREQ1_PATH,
        sep="\t",
        header=None,
        names=["token_code", "order", "token", "ngram_freq", "context_freq"],
    )
    # keep rows for words (not "context" records)
    f1["part"] = f1["token_code"].astype(str).str.split(".").str[-1]
    f1 = f1[f1["part"] == "word"].copy()
    unigram = f1[["token_code", "ngram_freq"]].rename(columns={"ngram_freq": "unigram_freq"})

    # ---------------- Load GPT-3 probabilities (all_stories_gpt3.csv) ----------------
    # Expected columns: story, token, logprob (and maybe others)
    g = pd.read_csv(GPT3_PATH)
    if not {"story", "token", "logprob"}.issubset(set(g.columns)):
        raise ValueError("all_stories_gpt3.csv must contain columns: story, token, logprob")

    # Align to RT indexing:
    # - RT items are 1-indexed; GPT story is 0-indexed => item = story + 1
    # - zone is token index within each story (1-indexed)
    g["item"] = g["story"].astype(int) + 1
    g["zone"] = g.groupby("story").cumcount() + 1
    # token in GPT file may have leading spaces => strip-left for matching RT word text
    g["word_gpt"] = g["token"].astype(str).str.lstrip()
    g["gpt3_prob"] = np.exp(g["logprob"].astype(float))
    g["gpt3_surprisal"] = -g["logprob"].astype(float)  # since logprob is ln(p)

    gpt3 = g[["item", "zone", "gpt3_prob", "gpt3_surprisal", "word_gpt"]].copy()

    # ---------------- Merge all ----------------
    df = (
        token_means
        .merge(unigram, on="token_code", how="left")
        .merge(gpt3, on=["item", "zone"], how="left")
    )

    # ---------------- Features ----------------
    df["word_clean"] = df["word"].apply(clean_token)
    df["is_empty_or_punct"] = df["word"].apply(is_punct_or_empty)

    # Fill unigram freq
    df["unigram_freq_filled"] = df["unigram_freq"].fillna(1).replace(0, 1).astype(float)
    df["log_freq"] = np.log(df["unigram_freq_filled"])
    df["neg_log_freq"] = -np.log(df["unigram_freq_filled"])

    # Fill GPT-3 probability (avoid log(0))
    minpos = df.loc[df["gpt3_prob"].notna() & (df["gpt3_prob"] > 0), "gpt3_prob"].min()
    eps = float(minpos / 2) if pd.notna(minpos) else 1e-12
    df["gpt3_prob_filled"] = df["gpt3_prob"].fillna(eps)
    df.loc[df["gpt3_prob_filled"] <= 0, "gpt3_prob_filled"] = eps
    df["gpt3_surprisal_filled"] = -np.log(df["gpt3_prob_filled"])

    df["word_len"] = df["word_clean"].str.len().fillna(0).astype(int)

    # Exclude empty/punct tokens for most analyses
    df_main = df[~df["is_empty_or_punct"]].copy()

    # ---------------- PART I: correlations + plots ----------------
    # correlations
    r_len_freq = pearsonr(df_main["word_len"], df_main["log_freq"])
    r_len_rt = pearsonr(df_main["word_len"], df_main["mean_RT"])
    r_freq_rt = pearsonr(df_main["log_freq"], df_main["mean_RT"])

    # plots
    scatter_with_binned_line(
        df_main["word_len"], df_main["mean_RT"],
        "Word length (characters)", "Mean RT per word position (ms)",
        "Part I: Word length vs Mean RT",
        plots_dir / "partI_length_vs_rt.png",
        bins=30,
    )

    scatter_with_binned_line(
        df_main["log_freq"], df_main["mean_RT"],
        "log(Unigram frequency)", "Mean RT per word position (ms)",
        "Part I: Unigram frequency vs Mean RT",
        plots_dir / "partI_logfreq_vs_rt.png",
        bins=30,
    )

    scatter_with_binned_line(
        df_main["gpt3_surprisal_filled"], df_main["mean_RT"],
        "GPT-3 surprisal (-log p)", "Mean RT per word position (ms)",
        "Part I (extra): GPT-3 surprisal vs Mean RT",
        plots_dir / "partI_gpt3surp_vs_rt.png",
        bins=30,
    )

    # ---------------- PART II: Hypothesis 1 ----------------
    # Model 1: mean_RT ~ -log(freq) + word_len
    m_freq = smf.ols("mean_RT ~ neg_log_freq + word_len", data=df_main).fit()

    # Model 2: mean_RT ~ GPT3_surprisal + word_len
    m_gpt3 = smf.ols("mean_RT ~ gpt3_surprisal_filled + word_len", data=df_main).fit()

    # predicted vs observed plots
    pred_vs_obs_plot(df_main["mean_RT"], m_freq.predict(df_main),
                     "Part II H1: Frequency model (predicted vs observed)",
                     plots_dir / "partII_H1_pred_vs_obs_freq.png")

    pred_vs_obs_plot(df_main["mean_RT"], m_gpt3.predict(df_main),
                     "Part II H1: GPT-3 surprisal model (predicted vs observed)",
                     plots_dir / "partII_H1_pred_vs_obs_gpt3.png")

    # ---------------- PART II: Hypothesis 2 (content vs function) ----------------
    # Use POS tagging (standard) to classify content vs function.
    ensure_nltk()
    import nltk
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import wordnet as wn

    # POS tag each unique word type (saves time)
    types = df_main["word_clean"].dropna().unique().tolist()
    # POS tagger expects token list
    tagged = nltk.pos_tag(types)

    pos_map = {w: tag for w, tag in tagged}
    df_main["penn_pos"] = df_main["word_clean"].map(pos_map).fillna("")

    # content words: nouns/verbs/adjs/advs
    def is_content_from_penn(tag: str) -> bool:
        if not tag:
            return False
        return tag.startswith(("NN", "VB", "JJ", "RB"))

    df_main["is_content"] = df_main["penn_pos"].apply(is_content_from_penn)
    df_main["is_function"] = ~df_main["is_content"]

    content = df_main[df_main["is_content"]].copy()
    function = df_main[df_main["is_function"]].copy()

    # 4 models:
    # Content: freq vs GPT3
    mc_freq = smf.ols("mean_RT ~ neg_log_freq + word_len", data=content).fit()
    mc_gpt3 = smf.ols("mean_RT ~ gpt3_surprisal_filled + word_len", data=content).fit()
    # Function: freq vs GPT3
    mf_freq = smf.ols("mean_RT ~ neg_log_freq + word_len", data=function).fit()
    mf_gpt3 = smf.ols("mean_RT ~ gpt3_surprisal_filled + word_len", data=function).fit()

    # Plot comparison: coefficient dots (optional but useful)
    coef_rows = []
    for name, model in [
        ("content_freq", mc_freq),
        ("content_gpt3", mc_gpt3),
        ("function_freq", mf_freq),
        ("function_gpt3", mf_gpt3),
    ]:
        for term in model.params.index:
            coef_rows.append((name, term, float(model.params[term])))
    coef_df = pd.DataFrame(coef_rows, columns=["model", "term", "coef"])
    coef_df.to_csv(tables_dir / "partII_H2_coefficients.csv", index=False)

    fig = plt.figure(figsize=(8, 5))
    # simple dot plot per term
    terms = ["neg_log_freq", "gpt3_surprisal_filled", "word_len"]
    models = ["content_freq", "content_gpt3", "function_freq", "function_gpt3"]
    for term in terms:
        sub = coef_df[coef_df["term"] == term].set_index("model").reindex(models).reset_index()
        plt.plot(range(len(models)), sub["coef"], marker="o", linestyle="-", label=term)
    plt.xticks(range(len(models)), models, rotation=20, ha="right")
    plt.ylabel("Coefficient")
    plt.title("Part II H2: Coefficients across content/function models")
    plt.legend()
    save_fig(fig, plots_dir / "partII_H2_coef_compare.png")

    # ---------------- PART III: FOBS Hypothesis 1 (lemma/root frequency) ----------------
    # Standard lemmatization with WordNetLemmatizer + POS tag mapping
    lemmatizer = WordNetLemmatizer()

    def lemma_word(word_clean: str, penn: str) -> str:
        if not word_clean:
            return ""
        wn_pos = penn_to_wn_pos(penn)
        return lemmatizer.lemmatize(word_clean, pos=wn_pos)

    df_main["lemma"] = df_main.apply(lambda r: lemma_word(r["word_clean"], r["penn_pos"]), axis=1)
    df_main["lemma_len"] = df_main["lemma"].str.len().fillna(0).astype(int)

    # Lemma frequency = sum of unigram freqs across UNIQUE surface forms in lemma family
    # (use median freq per surface form across positions, then sum by lemma)
    surface_type = (
        df_main.groupby("word_clean", as_index=False)
               .agg(surface_freq=("unigram_freq_filled", "median"),
                    surface_len=("word_len", "first"),
                    surface_lemma=("lemma", "first"),
                    surface_lemma_len=("lemma_len", "first"))
    )
    lemma_table = (
        surface_type.groupby("surface_lemma", as_index=False)
                    .agg(lemma_freq=("surface_freq", "sum"),
                         lemma_len=("surface_lemma_len", "first"),
                         n_forms=("word_clean", "count"))
                    .rename(columns={"surface_lemma": "lemma"})
    )

    df_main = df_main.merge(lemma_table[["lemma", "lemma_freq", "lemma_len", "n_forms"]],
                            on="lemma", how="left")

    df_main["lemma_freq_filled"] = df_main["lemma_freq"].fillna(1).replace(0, 1).astype(float)
    df_main["neg_log_lemma_freq"] = -np.log(df_main["lemma_freq_filled"])

    # Compare models:
    # Surface: mean_RT ~ -log(surface_freq) + word_len
    m_surface = smf.ols("mean_RT ~ neg_log_freq + word_len", data=df_main).fit()
    # --- SAFETY: ensure lemma_len exists ---
    if "lemma_len" not in df_main.columns:
        df_main["lemma_len"] = df_main["lemma"].astype(str).str.len().fillna(0).astype(int)
    # Lemma: mean_RT ~ -log(lemma_freq) + lemma_len
    m_lemma = smf.ols("mean_RT ~ neg_log_lemma_freq + lemma_len", data=df_main).fit()

    # Save lemma table
    lemma_table.to_csv(tables_dir / "partIII_H1_lemma_table.csv", index=False)

    # ---------------- PART III: FOBS Hypothesis 2 (pseudo-affixed vs real-affixed) ----------------
    # The assignment wants words matched by length & frequency.
    # We'll do:
    # 1) Build candidate sets for suffix "-er" (common in assignment examples: driver/finger).
    # 2) Real-affix heuristic: word endswith "er" AND stem exists in WordNet as a lemma (any POS).
    # 3) Pseudo-affix heuristic: word endswith "er" AND stem does NOT exist in WordNet.
    # 4) Then greedily match pseudo to real by closest (length, logfreq).
    #
    # If not enough words exist for "-er", we also try "-ly" (adverbs) similarly.

    def in_wordnet_lemma(w: str) -> bool:
        if not w:
            return False
        return len(wn.synsets(w)) > 0

    def build_affix_candidates(suffix: str):
        # use types present in df_main
        types_df = surface_type.copy()
        types_df = types_df[types_df["word_clean"].str.endswith(suffix, na=False)].copy()
        types_df["stem"] = types_df["word_clean"].str[:-len(suffix)]
        # require non-trivial stem
        types_df = types_df[types_df["stem"].str.len() >= 3].copy()
        # compute stem in wordnet
        types_df["stem_in_wordnet"] = types_df["stem"].apply(in_wordnet_lemma)
        # also require the full word exists in wordnet (to avoid typos)
        types_df["word_in_wordnet"] = types_df["word_clean"].apply(in_wordnet_lemma)
        types_df = types_df[types_df["word_in_wordnet"]].copy()

        real = types_df[types_df["stem_in_wordnet"]].copy()
        pseudo = types_df[~types_df["stem_in_wordnet"]].copy()
        return real, pseudo

    def greedy_match(real_df, pseudo_df, k=5):
        # match pseudo->real by minimizing weighted distance in (len, logfreq)
        if len(real_df) == 0 or len(pseudo_df) == 0:
            return pd.DataFrame()

        real_df = real_df.copy()
        pseudo_df = pseudo_df.copy()
        real_df["logf"] = np.log(real_df["surface_freq"].astype(float))
        pseudo_df["logf"] = np.log(pseudo_df["surface_freq"].astype(float))

        real_used = set()
        pairs = []
        for _, p in pseudo_df.sort_values(["surface_len", "logf"]).iterrows():
            best = None
            best_score = None
            for _, r in real_df.iterrows():
                if r["word_clean"] in real_used:
                    continue
                # distance: prioritize length matching, then frequency
                dlen = abs(int(p["surface_len"]) - int(r["surface_len"]))
                dlogf = abs(float(p["logf"]) - float(r["logf"]))
                score = 2.0 * dlen + 1.0 * dlogf
                if best_score is None or score < best_score:
                    best_score = score
                    best = r
            if best is not None:
                real_used.add(best["word_clean"])
                pairs.append({
                    "pseudo_word": p["word_clean"],
                    "pseudo_len": int(p["surface_len"]),
                    "pseudo_freq": float(p["surface_freq"]),
                    "real_word": best["word_clean"],
                    "real_len": int(best["surface_len"]),
                    "real_freq": float(best["surface_freq"]),
                    "score": float(best_score),
                })
            if len(pairs) >= k:
                break
        return pd.DataFrame(pairs)

    # Try "-er" first
    real_er, pseudo_er = build_affix_candidates("er")
    pairs_er = greedy_match(real_er, pseudo_er, k=5)

    # If not enough, try "-ly"
    pairs_ly = pd.DataFrame()
    if len(pairs_er) < 5:
        real_ly, pseudo_ly = build_affix_candidates("ly")
        pairs_ly = greedy_match(real_ly, pseudo_ly, k=5 - len(pairs_er))

    pairs = pd.concat([pairs_er, pairs_ly], ignore_index=True)

    # Ensure we include the example finger/driver if present
    # If both exist and not already included, add them as a pair (even if heuristic disagrees).
    def add_forced_pair(pseudo_word, real_word):
        nonlocal pairs
        if pseudo_word in surface_type["word_clean"].values and real_word in surface_type["word_clean"].values:
            if not ((pairs["pseudo_word"] == pseudo_word) & (pairs["real_word"] == real_word)).any():
                p = surface_type[surface_type["word_clean"] == pseudo_word].iloc[0]
                r = surface_type[surface_type["word_clean"] == real_word].iloc[0]
                forced = pd.DataFrame([{
                    "pseudo_word": pseudo_word,
                    "pseudo_len": int(p["surface_len"]),
                    "pseudo_freq": float(p["surface_freq"]),
                    "real_word": real_word,
                    "real_len": int(r["surface_len"]),
                    "real_freq": float(r["surface_freq"]),
                    "score": np.nan,
                }])
                pairs = pd.concat([pairs, forced], ignore_index=True)

    add_forced_pair("finger", "driver")

    # Save chosen pairs
    pairs.to_csv(tables_dir / "partIII_H2_matched_pairs.csv", index=False)

    # Build token-level dataset for the chosen words (all token positions)
    chosen_words = set(pairs["pseudo_word"].dropna().tolist() + pairs["real_word"].dropna().tolist())
    subset = df_main[df_main["word_clean"].isin(chosen_words)].copy()

    # label pseudo vs real
    pseudo_set = set(pairs["pseudo_word"].dropna().tolist())
    subset["affix_type"] = np.where(subset["word_clean"].isin(pseudo_set), "pseudo-affix", "real-affix")
    subset["is_pseudo"] = (subset["affix_type"] == "pseudo-affix").astype(int)

    # Summary table per word
    summary_subset = (
        subset.groupby("word_clean", as_index=False)
              .agg(mean_RT=("mean_RT", "mean"),
                   word_len=("word_len", "first"),
                   unigram_freq=("unigram_freq_filled", "median"),
                   gpt3_surprisal=("gpt3_surprisal_filled", "mean"),
                   affix_type=("affix_type", "first"))
    )
    summary_subset.to_csv(tables_dir / "partIII_H2_word_summary.csv", index=False)

    # Statistical test: pseudo vs real (token positions)
    if subset["affix_type"].nunique() == 2:
        real_rt = subset[subset["affix_type"] == "real-affix"]["mean_RT"]
        pseudo_rt = subset[subset["affix_type"] == "pseudo-affix"]["mean_RT"]
        ttest = ttest_ind(pseudo_rt, real_rt, equal_var=False)
    else:
        ttest = None

    # Regression controlling for length & frequency
    if len(subset) > 0:
        affix_model = smf.ols("mean_RT ~ is_pseudo + neg_log_freq + word_len", data=subset).fit()
    else:
        affix_model = None

    # Plot: per-word mean RT bars (simple)
    if len(summary_subset) > 0:
        fig = plt.figure(figsize=(9, 5))
        # sort by affix type then RT
        ss = summary_subset.sort_values(["affix_type", "mean_RT"])
        plt.bar(range(len(ss)), ss["mean_RT"])
        plt.xticks(range(len(ss)), ss["word_clean"], rotation=45, ha="right")
        plt.ylabel("Mean RT (ms)")
        plt.title("Part III H2: Mean RT for matched pseudo-affix vs real-affix words")
        save_fig(fig, plots_dir / "partIII_H2_bar_means.png")

    # Plot: pseudo vs real RT distributions (scatter)
    if len(subset) > 0:
        fig = plt.figure(figsize=(7, 5))
        x = np.where(subset["affix_type"] == "pseudo-affix", 1, 0)
        plt.scatter(x + np.random.uniform(-0.05, 0.05, size=len(x)), subset["mean_RT"], s=10, alpha=0.2)
        plt.xticks([0, 1], ["real-affix", "pseudo-affix"])
        plt.ylabel("Mean RT (ms)")
        plt.title("Part III H2: RT distribution (real vs pseudo)")
        save_fig(fig, plots_dir / "partIII_H2_rt_distribution.png")

    # ---------------- Save combined plot PDF (optional but handy) ----------------
    # Put all .pngs into a single PDF
    pngs = sorted(plots_dir.glob("*.png"))
    if pngs:
        from matplotlib.backends.backend_pdf import PdfPages
        pdf_path = out_dir / "all_plots.pdf"
        with PdfPages(pdf_path) as pdf:
            for p in pngs:
                img = plt.imread(p)
                fig = plt.figure(figsize=(8.27, 11.69))
                plt.imshow(img)
                plt.axis("off")
                plt.title(p.name, fontsize=10)
                pdf.savefig(fig)
                plt.close(fig)

    # ---------------- Write a text summary (for your submission) ----------------
    summary_path = logs_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== PART I: Correlations ===\n")
        f.write(f"r(length, log freq)  = {r_len_freq.statistic:.6f}, p={r_len_freq.pvalue:.3e}\n")
        f.write(f"r(length, mean RT)   = {r_len_rt.statistic:.6f}, p={r_len_rt.pvalue:.3e}\n")
        f.write(f"r(log freq, mean RT) = {r_freq_rt.statistic:.6f}, p={r_freq_rt.pvalue:.3e}\n\n")

        f.write("=== PART II: Hypothesis 1 (Regression comparison) ===\n")
        f.write("Model A: mean_RT ~ -log(freq) + word_len\n")
        f.write(f"  AdjR2={m_freq.rsquared_adj:.6f}, AIC={m_freq.aic:.2f}\n")
        f.write("Model B: mean_RT ~ GPT3_surprisal + word_len\n")
        f.write(f"  AdjR2={m_gpt3.rsquared_adj:.6f}, AIC={m_gpt3.aic:.2f}\n\n")

        f.write("=== PART II: Hypothesis 2 (Content vs Function) ===\n")
        f.write(f"Content tokens:  {len(content)}\n")
        f.write(f"Function tokens: {len(function)}\n")
        f.write("Content models:\n")
        f.write(f"  freq AdjR2={mc_freq.rsquared_adj:.6f}, AIC={mc_freq.aic:.2f}\n")
        f.write(f"  gpt3 AdjR2={mc_gpt3.rsquared_adj:.6f}, AIC={mc_gpt3.aic:.2f}\n")
        f.write("Function models:\n")
        f.write(f"  freq AdjR2={mf_freq.rsquared_adj:.6f}, AIC={mf_freq.aic:.2f}\n")
        f.write(f"  gpt3 AdjR2={mf_gpt3.rsquared_adj:.6f}, AIC={mf_gpt3.aic:.2f}\n\n")

        f.write("=== PART III: Hypothesis 1 (Surface vs Lemma frequency) ===\n")
        f.write("Surface: mean_RT ~ -log(surface_freq) + word_len\n")
        f.write(f"  AdjR2={m_surface.rsquared_adj:.6f}, AIC={m_surface.aic:.2f}\n")
        f.write("Lemma:   mean_RT ~ -log(lemma_freq) + lemma_len\n")
        f.write(f"  AdjR2={m_lemma.rsquared_adj:.6f}, AIC={m_lemma.aic:.2f}\n\n")

        f.write("=== PART III: Hypothesis 2 (Pseudo vs Real affix) ===\n")
        f.write(f"Matched pairs saved to: {tables_dir / 'partIII_H2_matched_pairs.csv'}\n")
        if ttest is not None:
            f.write(f"Welch t-test (pseudo vs real): t={ttest.statistic:.6f}, p={ttest.pvalue:.3e}\n")
        else:
            f.write("Welch t-test: not computed (need both groups).\n")
        if affix_model is not None:
            f.write("\nRegression: mean_RT ~ is_pseudo + -log(freq) + word_len\n")
            f.write(f"is_pseudo coef={affix_model.params.get('is_pseudo', np.nan):.6f}, "
                    f"p={affix_model.pvalues.get('is_pseudo', np.nan):.3e}\n")

        f.write("\n=== WHERE TO LOOK ===\n")
        f.write(f"Plots:  {plots_dir}\n")
        f.write(f"Tables: {tables_dir}\n")
        f.write(f"All plots PDF: {out_dir / 'all_plots.pdf'}\n")

    print(f"\nDone. Outputs written to: {out_dir}")
    print(f"Summary: {summary_path}")

if __name__ == "__main__":
    main()
