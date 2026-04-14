""" # =========================================
# TOKEN ANALYSIS FOR LLM DATASET (PARQUET)
# =========================================

import pandas as pd
import tiktoken
import matplotlib.pyplot as plt
import os

# =========================================
# 1. LOAD DATASET
# =========================================

FILE_PATH = "test-00000-of-00001.parquet"  # ganti jika perlu

if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"File tidak ditemukan: {FILE_PATH}")

df = pd.read_parquet(FILE_PATH)

print("\n=== DATASET LOADED ===")
print("Jumlah data:", len(df))
print("\nKolom dataset:")
print(df.columns)

print("\nContoh data:")
print(df.head())


# =========================================
# 2. DETEKSI KOLOM OTOMATIS
# =========================================

# Coba deteksi kolom umum
possible_question_cols = ["question", "input", "prompt"]
possible_answer_cols = ["answer", "output", "response"]
possible_reasoning_cols = ["reasoning", "rationale", "explanation"]

def find_column(possible_list, df_cols):
    for col in possible_list:
        if col in df_cols:
            return col
    return None

question_col = find_column(possible_question_cols, df.columns)
answer_col = find_column(possible_answer_cols, df.columns)
reasoning_col = find_column(possible_reasoning_cols, df.columns)

print("\n=== DETEKSI KOLOM ===")
print("Question column :", question_col)
print("Answer column   :", answer_col)
print("Reasoning column:", reasoning_col)

if question_col is None or answer_col is None:
    raise ValueError("Kolom question/answer tidak ditemukan. Cek nama kolom dataset kamu.")


# =========================================
# 3. SETUP TOKENIZER (GPT)
# =========================================

encoding = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    if pd.isna(text):
        return 0
    return len(encoding.encode(str(text)))


# =========================================
# 4. HITUNG TOKEN
# =========================================

print("\nMenghitung token...")

df["question_tokens"] = df[question_col].apply(count_tokens)
df["answer_tokens"] = df[answer_col].apply(count_tokens)

if reasoning_col:
    df["reasoning_tokens"] = df[reasoning_col].apply(count_tokens)
else:
    df["reasoning_tokens"] = 0


# =========================================
# 5. TOTAL TOKEN
# =========================================

df["total_tokens"] = (
    df["question_tokens"] +
    df["answer_tokens"] +
    df["reasoning_tokens"]
)


# =========================================
# 6. STATISTIK
# =========================================

print("\n=== STATISTIK TOKEN ===")

print(f"Rata-rata question tokens : {df['question_tokens'].mean():.2f}")
print(f"Rata-rata answer tokens   : {df['answer_tokens'].mean():.2f}")
print(f"Rata-rata reasoning tokens: {df['reasoning_tokens'].mean():.2f}")
print(f"Rata-rata total tokens    : {df['total_tokens'].mean():.2f}")

print("\nTotal token keseluruhan:")
print(df[["question_tokens", "answer_tokens", "reasoning_tokens"]].sum())


# =========================================
# 7. DISTRIBUSI TOKEN
# =========================================

plt.figure()
plt.hist(df["total_tokens"], bins=50)
plt.title("Distribusi Total Token")
plt.xlabel("Jumlah Token")
plt.ylabel("Frekuensi")
plt.savefig("token_distribution.png")
plt.close()

print("\nGrafik distribusi disimpan: token_distribution.png")


# =========================================
# 8. SAMPLE TERPANJANG (WORST CASE)
# =========================================

df_sorted = df.sort_values(by="total_tokens", ascending=False)

print("\n=== TOP 5 DATA TERPANJANG ===")
print(df_sorted[[question_col, answer_col, "total_tokens"]].head(5))


# =========================================
# 9. SIMULASI LATENT / COMPRESSED CoT
# =========================================

# Asumsi: reasoning dipadatkan 50%
COMPRESSION_RATIO = 0.5

df["compressed_reasoning_tokens"] = df["reasoning_tokens"] * COMPRESSION_RATIO

df["compressed_total_tokens"] = (
    df["question_tokens"] +
    df["answer_tokens"] +
    df["compressed_reasoning_tokens"]
)

print("\n=== SIMULASI LATENT CoT ===")
print(f"Rata-rata token original  : {df['total_tokens'].mean():.2f}")
print(f"Rata-rata token compressed: {df['compressed_total_tokens'].mean():.2f}")

reduction = (
    (df["total_tokens"].mean() - df["compressed_total_tokens"].mean()) /
    df["total_tokens"].mean()
) * 100

print(f"Efisiensi pengurangan token: {reduction:.2f}%")


# =========================================
# 10. SIMPAN HASIL
# =========================================

OUTPUT_FILE = "hasil_token_analysis.csv"
df.to_csv(OUTPUT_FILE, index=False)

print(f"\nHasil disimpan ke: {OUTPUT_FILE}")


# =========================================
# DONE
# =========================================

print("\n=== SELESAI ===")

# =========================================
# 11. PROMPT MODES
# =========================================

def build_prompt(question, mode="no_cot"):
    if mode == "no_cot":
        return f"Answer the following question:\n{question}"

    elif mode == "cot":
        return f"Answer step-by-step:\n{question}"

    elif mode == "latent_cot":
        return (
            f"Solve the problem step-by-step internally, "
            f"but only output the final answer.\nQuestion:\n{question}"
        )

    else:
        raise ValueError("Mode tidak dikenal")
    
# =========================================
# 12. SIMULASI OUTPUT MODEL
# =========================================

def simulate_model_output(row, mode):
    question = row[question_col]
    answer = row[answer_col]
    reasoning = row.get(reasoning_col, "")

    if mode == "no_cot":
        return answer

    elif mode == "cot":
        return f"{reasoning}\nFinal Answer: {answer}"

    elif mode == "latent_cot":
        compressed_reasoning = str(reasoning)[:int(len(str(reasoning)) * 0.3)]
        return f"{compressed_reasoning} ... Final Answer: {answer}"
    
# =========================================
# 13. TOKEN PER MODE
# =========================================

print("\n=== MENGHITUNG TOKEN PER MODE ===")

modes = ["no_cot", "cot", "latent_cot"]

results = {}

for mode in modes:
    print(f"\nProcessing mode: {mode}")

    outputs = df.apply(lambda row: simulate_model_output(row, mode), axis=1)
    tokens = outputs.apply(count_tokens)

    results[mode] = tokens.mean()

    print(f"Rata-rata token ({mode}): {tokens.mean():.2f}")
    
# =========================================
# 14. PERBANDINGAN EFISIENSI
# =========================================

print("\n=== PERBANDINGAN MODE ===")

no_cot = results["no_cot"]
cot = results["cot"]
latent = results["latent_cot"]

print(f"No CoT      : {no_cot:.2f}")
print(f"Full CoT    : {cot:.2f}")
print(f"Latent CoT  : {latent:.2f}")

reduction_vs_cot = ((cot - latent) / cot) * 100

print(f"\nEfisiensi Latent vs CoT: {reduction_vs_cot:.2f}%")

# =========================================
# HITUNG REDUKSI
# =========================================

reduction_cot_to_latent = ((results["cot"] - results["latent_cot"]) / results["cot"]) * 100
increase_no_cot_to_cot = ((results["cot"] - results["no_cot"]) / results["no_cot"]) * 100

print("\n=== ANALISIS EFISIENSI ===")
print(f"Kenaikan token (No CoT → CoT): {increase_no_cot_to_cot:.2f}%")
print(f"Pengurangan token (CoT → Latent CoT): {reduction_cot_to_latent:.2f}%")

# =========================================
# VISUALISASI PERBANDINGAN (IMPROVED)
# =========================================

import matplotlib.pyplot as plt

labels = list(results.keys())
values = list(results.values())

plt.figure()

bars = plt.bar(labels, values)

# Tambahkan angka di atas bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.1f}", 
             ha='center', va='bottom')

plt.title("Perbandingan Penggunaan Token (No CoT vs CoT vs Latent CoT)")
plt.xlabel("Metode")
plt.ylabel("Rata-rata Token")

plt.savefig("comparison_modes.png")
plt.show() """

# =========================================
# TOKEN ANALYSIS FOR LLM DATASET (PARQUET)
# Latent Chain-of-Thought — Token Reduction
# Author : Gema Satya Danera (2404130070)
# =========================================

import pandas as pd
import tiktoken
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import re

# =========================================
# 1. LOAD DATASET
# =========================================

FILE_PATH = "train-00000-of-00001.parquet"

if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"File tidak ditemukan: {FILE_PATH}")

df = pd.read_parquet(FILE_PATH)

print("\n=== DATASET LOADED ===")
print("Jumlah data:", len(df))
print("\nKolom dataset:")
print(df.columns.tolist())
print("\nContoh data:")
print(df.head(2))


# =========================================
# 2. DETEKSI KOLOM OTOMATIS
# =========================================

possible_question_cols  = ["question", "input", "prompt", "problem"]
possible_answer_cols    = ["answer", "output", "response", "solution"]
possible_reasoning_cols = ["reasoning", "rationale", "explanation", "steps"]

def find_column(possible_list, df_cols):
    for col in possible_list:
        if col in df_cols:
            return col
    return None

question_col  = find_column(possible_question_cols,  df.columns)
answer_col    = find_column(possible_answer_cols,    df.columns)
reasoning_col = find_column(possible_reasoning_cols, df.columns)

print("\n=== DETEKSI KOLOM ===")
print("Question column :", question_col)
print("Answer column   :", answer_col)
print("Reasoning column:", reasoning_col)

if question_col is None or answer_col is None:
    raise ValueError("Kolom question/answer tidak ditemukan. Cek nama kolom dataset.")


# =========================================
# 3. SETUP TOKENIZER
# =========================================

encoding = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    """Hitung jumlah token dari sebuah string."""
    if pd.isna(text) or text is None:
        return 0
    return len(encoding.encode(str(text)))


# =========================================
# 4. HITUNG TOKEN PER KOLOM
# =========================================

print("\nMenghitung token kolom dasar...")

df["question_tokens"]  = df[question_col].apply(count_tokens)
df["answer_tokens"]    = df[answer_col].apply(count_tokens)
df["reasoning_tokens"] = (
    df[reasoning_col].apply(count_tokens) if reasoning_col else 0
)


# =========================================
# 5. STATISTIK DASAR
# =========================================

print("\n=== STATISTIK TOKEN ===")
print(f"Rata-rata question tokens : {df['question_tokens'].mean():.2f}")
print(f"Rata-rata answer tokens   : {df['answer_tokens'].mean():.2f}")
print(f"Rata-rata reasoning tokens: {df['reasoning_tokens'].mean():.2f}")


# =========================================
# 6. SIMULASI OUTPUT PER MODE
# =========================================
# ┌─────────────────────────────────────────────────────────────┐
# │  PERBAIKAN UTAMA ada di bagian ini.                         │
# │                                                             │
# │  BUG LAMA: latent_cot masih menyertakan teks reasoning      │
# │  (meski dipotong 30%), sehingga token-nya hampir sama       │
# │  dengan full CoT.                                           │
# │                                                             │
# │  FIX BARU: Latent CoT = hanya jawaban akhir (final answer). │
# │  Proses berpikir dilakukan secara IMPLISIT di ruang laten   │
# │  model — TIDAK dieksternalisasi ke teks sama sekali.        │
# └─────────────────────────────────────────────────────────────┘

def extract_final_answer(answer_text: str) -> str:
    """
    Ambil hanya bagian jawaban akhir yang singkat.
    Strategi:
      1. Cari pola 'The answer is ...' / 'Final Answer: ...'
      2. Jika tidak ada, ambil kalimat/baris terakhir yang mengandung angka/kata kunci.
      3. Fallback: ambil maksimal 20 kata pertama dari answer.
    """
    text = str(answer_text).strip()

    # Pola 1 — kalimat eksplisit jawaban akhir
    patterns = [
        r"(?:the answer is|final answer[:\s]+|answer[:\s]+|therefore[,\s]+(?:the answer is)?)\s*(.+?)(?:\.|$)",
        r"(?:=|equals?)\s*([\d\.,\-]+)",
        r"\$\s*([\d\.,]+)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip().rstrip(".")
            # Batasi maksimal 15 kata
            words = candidate.split()
            return " ".join(words[:15])

    # Pola 2 — baris terakhir yang tidak kosong
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines:
        last_line = lines[-1]
        words = last_line.split()
        return " ".join(words[:15])

    # Fallback — 15 kata pertama
    words = text.split()
    return " ".join(words[:15])


def simulate_model_output(row, mode: str) -> str:
    """
    Simulasi teks OUTPUT model untuk setiap mode.

    no_cot    : Jawaban langsung tanpa penalaran bertahap.
    cot       : Penalaran eksplisit + jawaban akhir (standar CoT).
    latent_cot: HANYA jawaban akhir. Reasoning terjadi secara implisit
                di dalam representasi laten model — tidak ditulis ke output.
    """
    question  = str(row[question_col])
    answer    = str(row[answer_col])
    reasoning = str(row[reasoning_col]) if reasoning_col else ""

    if mode == "no_cot":
        # ── Output: pertanyaan + jawaban langsung (tanpa langkah) ──
        return f"Q: {question}\nA: {answer}"

    elif mode == "cot":
        # ── Output: pertanyaan + seluruh rantai reasoning + jawaban ──
        return f"Q: {question}\nReasoning: {reasoning}\nFinal Answer: {answer}"

    elif mode == "latent_cot":
        # ── Output: HANYA jawaban akhir yang ringkas ──
        # Reasoning dilakukan secara laten (di dalam hidden state model),
        # TIDAK dieksternalisasi ke token teks.
        final_answer = extract_final_answer(answer)
        return f"Q: {question}\nA: {final_answer}"

    else:
        raise ValueError(f"Mode tidak dikenal: {mode}")


# =========================================
# 7. HITUNG TOKEN PER MODE
# =========================================

print("\n=== MENGHITUNG TOKEN PER MODE ===")

modes   = ["no_cot", "cot", "latent_cot"]
results = {}

for mode in modes:
    outputs       = df.apply(lambda row: simulate_model_output(row, mode), axis=1)
    token_series  = outputs.apply(count_tokens)
    results[mode] = {
        "mean"  : token_series.mean(),
        "median": token_series.median(),
        "max"   : token_series.max(),
        "min"   : token_series.min(),
        "series": token_series,
    }
    print(f"\n[{mode.upper()}]")
    print(f"  Rata-rata : {results[mode]['mean']:.2f}")
    print(f"  Median    : {results[mode]['median']:.2f}")
    print(f"  Min–Max   : {results[mode]['min']} – {results[mode]['max']}")

# Tambahkan kolom token per mode ke dataframe
df["tokens_no_cot"]    = results["no_cot"]["series"]
df["tokens_cot"]       = results["cot"]["series"]
df["tokens_latent_cot"]= results["latent_cot"]["series"]


# =========================================
# 8. ANALISIS EFISIENSI
# =========================================

no_cot_mean = results["no_cot"]["mean"]
cot_mean    = results["cot"]["mean"]
latent_mean = results["latent_cot"]["mean"]

overhead_cot       = ((cot_mean - no_cot_mean) / no_cot_mean) * 100
reduction_cot_lat  = ((cot_mean - latent_mean) / cot_mean)    * 100
reduction_nocot_lat= ((no_cot_mean - latent_mean) / no_cot_mean) * 100 \
                     if latent_mean < no_cot_mean else 0

print("\n=== ANALISIS EFISIENSI ===")
print(f"No CoT      : {no_cot_mean:.2f} token")
print(f"Full CoT    : {cot_mean:.2f} token")
print(f"Latent CoT  : {latent_mean:.2f} token")
print(f"\nOverhead token  No CoT  → CoT        : +{overhead_cot:.2f}%")
print(f"Reduksi token   CoT     → Latent CoT : -{reduction_cot_lat:.2f}%")

if reduction_nocot_lat > 0:
    print(f"Reduksi token   No CoT  → Latent CoT : -{reduction_nocot_lat:.2f}%")
else:
    gap = latent_mean - no_cot_mean
    print(f"Latent CoT lebih boros {gap:.2f} token dari No CoT "
          f"(karena question masih disertakan di prompt)")


# =========================================
# 9. SIMULASI LATENT (SKENARIO KOMPRESI)
# =========================================
# Ini adalah estimasi teoritis jika model hanya mengembalikan
# token jawaban murni (tanpa echo pertanyaan).

print("\n=== SKENARIO KOMPRESI LATEN (TEORITIS) ===")

COMPRESSION_RATIOS = [0.1, 0.2, 0.3]  # 10%, 20%, 30% dari token CoT

for cr in COMPRESSION_RATIOS:
    compressed = cot_mean * cr
    saved      = ((cot_mean - compressed) / cot_mean) * 100
    print(f"  CR={cr:.0%} → {compressed:.1f} token rata-rata | hemat {saved:.1f}%")


# =========================================
# 10. SIMPAN HASIL
# =========================================

OUTPUT_FILE = "hasil_token_analysis.csv"
df[[
    question_col, answer_col,
    "question_tokens", "answer_tokens", "reasoning_tokens",
    "tokens_no_cot", "tokens_cot", "tokens_latent_cot"
]].to_csv(OUTPUT_FILE, index=False)

print(f"\nHasil disimpan ke: {OUTPUT_FILE}")


# =========================================
# 11. VISUALISASI — BAR CHART PERBANDINGAN
# =========================================

labels = ["No CoT", "Full CoT\n(Standard)", "Latent CoT\n(Proposed)"]
means  = [no_cot_mean, cot_mean, latent_mean]
colors = ["#5b8dee", "#e05c5c", "#2dd68b"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor("#0d1117")
for ax in axes:
    ax.set_facecolor("#161b22")

# ── Sub-plot 1: Bar chart rata-rata token ──
bars = axes[0].bar(labels, means, color=colors, width=0.5, edgecolor="none",
                   zorder=3)
for bar, val in zip(bars, means):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{val:.1f}",
        ha="center", va="bottom",
        color="white", fontsize=10, fontweight="bold"
    )

axes[0].set_title("Rata-rata Token per Mode Inferensi",
                  color="white", fontsize=12, pad=12)
axes[0].set_ylabel("Jumlah Token (rata-rata)", color="#8b949e")
axes[0].tick_params(colors="white")
axes[0].spines[["top","right","left","bottom"]].set_color("#30363d")
axes[0].yaxis.label.set_color("#8b949e")
axes[0].set_ylim(0, max(means) * 1.2)
axes[0].grid(axis="y", color="#30363d", linewidth=0.5, zorder=0)

# Anotasi reduksi
axes[0].annotate(
    f"−{reduction_cot_lat:.1f}%",
    xy=(2, latent_mean), xytext=(1.6, (cot_mean + latent_mean) / 2),
    arrowprops=dict(arrowstyle="->", color="#2dd68b", lw=1.5),
    fontsize=10, color="#2dd68b", fontweight="bold"
)

# ── Sub-plot 2: Boxplot distribusi token ──
data_to_plot = [
    results["no_cot"]["series"].values,
    results["cot"]["series"].values,
    results["latent_cot"]["series"].values,
]
bp = axes[1].boxplot(
    data_to_plot,
    labels=["No CoT", "Full CoT", "Latent CoT"],
    patch_artist=True,
    medianprops=dict(color="white", linewidth=2),
    whiskerprops=dict(color="#8b949e"),
    capprops=dict(color="#8b949e"),
    flierprops=dict(markerfacecolor="#8b949e", marker="o", markersize=3, alpha=0.5)
)
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

axes[1].set_title("Distribusi Token per Mode",
                  color="white", fontsize=12, pad=12)
axes[1].set_ylabel("Jumlah Token", color="#8b949e")
axes[1].tick_params(colors="white")
axes[1].spines[["top","right","left","bottom"]].set_color("#30363d")
axes[1].yaxis.label.set_color("#8b949e")
axes[1].grid(axis="y", color="#30363d", linewidth=0.5, zorder=0)

# Legend
patches = [mpatches.Patch(color=c, label=l)
           for c, l in zip(colors, ["No CoT", "Full CoT", "Latent CoT"])]
fig.legend(handles=patches, loc="lower center", ncol=3,
           framealpha=0.15, labelcolor="white", fontsize=9)

plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig("comparison_modes.png", dpi=150, bbox_inches="tight",
            facecolor="#0d1117")
plt.show()
print("\nGrafik disimpan: comparison_modes.png")


# =========================================
# 12. DISTRIBUSI TOKEN — HISTOGRAM
# =========================================

fig2, ax2 = plt.subplots(figsize=(9, 4))
fig2.patch.set_facecolor("#0d1117")
ax2.set_facecolor("#161b22")

for mode, color, label in zip(
    ["no_cot", "cot", "latent_cot"],
    colors,
    ["No CoT", "Full CoT (Standard)", "Latent CoT (Proposed)"]
):
    ax2.hist(results[mode]["series"], bins=40, alpha=0.6,
             color=color, label=label, edgecolor="none")

ax2.set_title("Distribusi Jumlah Token per Mode", color="white", fontsize=12)
ax2.set_xlabel("Jumlah Token", color="#8b949e")
ax2.set_ylabel("Frekuensi", color="#8b949e")
ax2.tick_params(colors="white")
ax2.spines[["top","right","left","bottom"]].set_color("#30363d")
ax2.legend(labelcolor="white", framealpha=0.2)
ax2.grid(color="#30363d", linewidth=0.5)

plt.tight_layout()
plt.savefig("token_distribution.png", dpi=150, bbox_inches="tight",
            facecolor="#0d1117")
plt.show()
print("Grafik distribusi disimpan: token_distribution.png")


# =========================================
# RINGKASAN AKHIR
# =========================================

print("\n" + "=" * 50)
print("RINGKASAN HASIL PENELITIAN")
print("=" * 50)
print(f"  Dataset      : {FILE_PATH}  ({len(df)} sampel)")
print(f"  No CoT       : {no_cot_mean:.2f} token rata-rata")
print(f"  Full CoT     : {cot_mean:.2f} token rata-rata")
print(f"  Latent CoT   : {latent_mean:.2f} token rata-rata")
print(f"  Reduksi CoT → Latent : {reduction_cot_lat:.2f}%")
print("=" * 50)
print("SELESAI ✓")

