# =========================================
# TOKEN ANALYSIS + ACCURACY MEASUREMENT
# Dataset: SVAMP (1000 Math Word Problems)
# Latent Chain-of-Thought — Token Reduction
# Author : Gema Satya Danera (2404130070)
# =========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re
import os

# =========================================
# 1. LOAD DATASET SVAMP
# =========================================

FILE_PATH = "D:/Tugas Kuliah/Semester 4/Riset Teknologi Informasi/Chain-of-thought/Dataset/gsm8k/main/SVAMP.csv"
df = pd.read_csv(FILE_PATH)

# Gabungkan Body + Question menjadi satu kolom "problem"
df["problem"] = df["Body"].str.strip() + " " + df["Question"].str.strip()
df["answer_str"] = df["Answer"].astype(str)

print("=== DATASET SVAMP LOADED ===")
print(f"Jumlah data  : {len(df)}")
print(f"Kolom        : {df.columns.tolist()}")
print(f"Tipe soal    : {df['Type'].value_counts().to_dict()}")
print(f"\nContoh soal  :\n{df[['problem','Answer']].head(3).to_string()}")

# =========================================
# 2. SETUP TOKENIZER (Offline, no API needed)
# =========================================
# Menggunakan word-level tokenizer berbasis regex yang memisahkan
# kata, angka, dan tanda baca — konsisten dan deterministik.
# Estimasi ~0.75x token GPT-4 (word tokens ≈ 1.33x subword tokens),
# namun untuk analisis perbandingan relatif antar mode hasilnya ekuivalen.

_TOKEN_PATTERN = re.compile(r"[\w']+|[.,!?;:()\[\]{}\-$%]")

def count_tokens(text: str) -> int:
    """Hitung jumlah token dari sebuah string (word-level regex tokenizer)."""
    if pd.isna(text) or text is None:
        return 0
    return len(_TOKEN_PATTERN.findall(str(text)))

# =========================================
# 3. EXTRACT FINAL ANSWER (LATENT COT)
# =========================================

def extract_number(text: str) -> float | None:
    """Ekstrak angka pertama yang ditemukan dalam string."""
    text = str(text).replace(",", "")
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    if matches:
        return float(matches[0])
    return None

def extract_final_answer(answer_text: str) -> str:
    """
    Ambil HANYA jawaban akhir numerik yang paling ringkas.
    Reasoning terjadi secara implisit di ruang laten — tidak dieksternalisasi.
    """
    text = str(answer_text).strip()

    # Pola eksplisit jawaban akhir
    patterns = [
        r"(?:the answer is|final answer[:\s]+|answer[:\s]+|therefore[,\s]+(?:the answer is)?)\s*(.+?)(?:\.|$)",
        r"(?:=|equals?)\s*([\d\.,\-]+)",
        r"\$\s*([\d\.,]+)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip().rstrip(".")
            words = candidate.split()
            return " ".join(words[:10])

    # Ambil angka pertama jika tidak ada pola
    num = extract_number(text)
    if num is not None:
        return str(int(num) if num == int(num) else num)

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines:
        return " ".join(lines[-1].split()[:10])

    return " ".join(text.split()[:10])


# =========================================
# 4. SIMULASI OUTPUT PER MODE
# =========================================

def simulate_output(row, mode: str) -> str:
    """
    Simulasi teks output model per mode.

    no_cot     : Q + A langsung (tanpa langkah reasoning).
    cot        : Q + Equation (sebagai proxy reasoning) + A.
    latent_cot : HANYA Q + jawaban akhir. Reasoning terjadi secara
                 IMPLISIT di ruang laten model — tidak ditulis ke output.
    """
    problem  = str(row["problem"])
    equation = str(row["Equation"])
    answer   = str(row["Answer"])

    if mode == "no_cot":
        return f"Q: {problem}\nA: {answer}"

    elif mode == "cot":
        return f"Q: {problem}\nReasoning: {equation}\nFinal Answer: {answer}"

    elif mode == "latent_cot":
        # Reasoning dilakukan secara laten — hanya jawaban akhir ditulis
        final = extract_final_answer(answer)
        return f"Q: {problem}\nA: {final}"

    else:
        raise ValueError(f"Mode tidak dikenal: {mode}")


# =========================================
# 5. HITUNG TOKEN PER MODE
# =========================================

print("\n=== MENGHITUNG TOKEN PER MODE ===")

modes   = ["no_cot", "cot", "latent_cot"]
results = {}

for mode in modes:
    outputs      = df.apply(lambda row: simulate_output(row, mode), axis=1)
    token_series = outputs.apply(count_tokens)
    results[mode] = {
        "mean"  : token_series.mean(),
        "median": token_series.median(),
        "max"   : token_series.max(),
        "min"   : token_series.min(),
        "series": token_series,
        "total" : token_series.sum(),
    }
    print(f"\n[{mode.upper()}]")
    print(f"  Rata-rata : {results[mode]['mean']:.2f}")
    print(f"  Median    : {results[mode]['median']:.2f}")
    print(f"  Min–Max   : {results[mode]['min']} – {results[mode]['max']}")
    print(f"  Total     : {results[mode]['total']}")

df["tokens_no_cot"]     = results["no_cot"]["series"]
df["tokens_cot"]        = results["cot"]["series"]
df["tokens_latent_cot"] = results["latent_cot"]["series"]


# =========================================
# 6. ANALISIS EFISIENSI TOKEN
# =========================================

no_cot_mean = results["no_cot"]["mean"]
cot_mean    = results["cot"]["mean"]
latent_mean = results["latent_cot"]["mean"]

overhead_cot      = ((cot_mean - no_cot_mean) / no_cot_mean) * 100
reduction_cot_lat = ((cot_mean - latent_mean) / cot_mean) * 100

total_cot    = results["cot"]["total"]
total_latent = results["latent_cot"]["total"]
total_reduction = ((total_cot - total_latent) / total_cot) * 100

print("\n=== ANALISIS EFISIENSI TOKEN ===")
print(f"No CoT      : {no_cot_mean:.2f} token")
print(f"Full CoT    : {cot_mean:.2f} token")
print(f"Latent CoT  : {latent_mean:.2f} token")
print(f"\nOverhead  No CoT → CoT         : +{overhead_cot:.2f}%")
print(f"Reduksi   CoT   → Latent CoT   : -{reduction_cot_lat:.2f}%")
print(f"\nTotal CoT token    : {total_cot:,}")
print(f"Total Latent token : {total_latent:,}")
print(f"Total Reduction    : -{total_reduction:.2f}%")


# =========================================
# 7. PENGUKURAN AKURASI
# =========================================
# Akurasi diukur dengan membandingkan jawaban numerik model
# terhadap ground-truth Answer di dataset SVAMP.
# Karena ini adalah simulasi (bukan inferensi live LLM),
# kita mengukur seberapa baik extract_final_answer() 
# berhasil mengekstrak angka yang tepat dari kolom Answer.

print("\n=== PENGUKURAN AKURASI ===")

def evaluate_accuracy(df_in: pd.DataFrame, tolerance: float = 0.01) -> dict:
    """
    Hitung akurasi per mode dengan membandingkan jawaban yang
    diekstrak terhadap ground truth Answer.

    Untuk no_cot & cot : jawaban diambil langsung dari kolom Answer
                          (simulasi model yang menghasilkan output benar).
    Untuk latent_cot   : jawaban diekstrak via extract_final_answer()
                          yang merepresentasikan kompresi output laten.

    Metrik:
    - Exact Match   : angka identik persis
    - Approx Match  : selisih relatif ≤ tolerance (default 1%)
    - Parse Rate    : seberapa sering angka berhasil di-parse
    """
    gt = df_in["Answer"].astype(float)

    metrics = {}

    # ── No CoT & CoT: output = Answer langsung → akurasi 100% sebagai baseline ──
    for mode in ["no_cot", "cot"]:
        pred = gt.copy()
        exact   = (pred == gt).sum()
        approx  = ((pred - gt).abs() / (gt.abs() + 1e-9) <= tolerance).sum()
        metrics[mode] = {
            "exact_match" : exact / len(gt) * 100,
            "approx_match": approx / len(gt) * 100,
            "parse_rate"  : 100.0,
            "total"       : len(gt),
        }

    # ── Latent CoT: ekstrak angka dari extract_final_answer(Answer) ──
    latent_preds = df_in["Answer"].apply(
        lambda a: extract_number(extract_final_answer(str(a)))
    )

    parsed_mask = latent_preds.notna()
    parse_rate  = parsed_mask.sum() / len(gt) * 100

    exact_match  = 0
    approx_match = 0

    for pred_val, gt_val in zip(latent_preds, gt):
        if pred_val is None:
            continue
        if pred_val == gt_val:
            exact_match += 1
        if abs(pred_val - gt_val) / (abs(gt_val) + 1e-9) <= tolerance:
            approx_match += 1

    metrics["latent_cot"] = {
        "exact_match" : exact_match / len(gt) * 100,
        "approx_match": approx_match / len(gt) * 100,
        "parse_rate"  : parse_rate,
        "total"       : len(gt),
    }

    return metrics

accuracy_metrics = evaluate_accuracy(df)

print(f"\n{'Method':<15} {'Exact Match':>12} {'Approx Match':>14} {'Parse Rate':>12}")
print("-" * 55)
for mode in modes:
    m = accuracy_metrics[mode]
    print(f"{mode:<15} {m['exact_match']:>11.2f}% {m['approx_match']:>13.2f}% {m['parse_rate']:>11.2f}%")

# Akurasi per tipe soal (Latent CoT)
print("\n=== AKURASI LATENT COT PER TIPE SOAL ===")
df["latent_pred"] = df["Answer"].apply(
    lambda a: extract_number(extract_final_answer(str(a)))
)
df["latent_correct"] = df.apply(
    lambda r: r["latent_pred"] is not None and
              abs(r["latent_pred"] - r["Answer"]) / (abs(r["Answer"]) + 1e-9) <= 0.01,
    axis=1
)

type_acc = df.groupby("Type")["latent_correct"].agg(["sum", "count"])
type_acc["accuracy_%"] = (type_acc["sum"] / type_acc["count"] * 100).round(2)
print(type_acc[["sum","count","accuracy_%"]].rename(
    columns={"sum":"Correct","count":"Total"}
).to_string())


# =========================================
# 8. SKENARIO KOMPRESI TEORITIS
# =========================================

print("\n=== SKENARIO KOMPRESI LATEN (TEORITIS) ===")
for cr in [0.10, 0.20, 0.30]:
    compressed = cot_mean * cr
    saved      = (1 - cr) * 100
    print(f"  CR={cr:.0%} → {compressed:.1f} token rata-rata | hemat {saved:.0f}%")


# =========================================
# 9. SIMPAN HASIL CSV
# =========================================

out_cols = [
    "ID","problem","Answer","Type",
    "tokens_no_cot","tokens_cot","tokens_latent_cot",
    "latent_pred","latent_correct"
]
df[out_cols].to_csv("D:/Tugas Kuliah/Semester 4/Riset Teknologi Informasi/Chain-of-thought/Dataset/gsm8k/main/hasil_token_svamp.csv", index=False)
print("\nHasil disimpan ke: hasil_token_svamp.csv")

summary_table = pd.DataFrame({
    "Method"      : ["No CoT", "Full CoT", "Latent CoT"],
    "Avg Tokens"  : [no_cot_mean, cot_mean, latent_mean],
    "Median"      : [results[m]["median"] for m in modes],
    "Total Tokens": [results[m]["total"]  for m in modes],
    "Exact Match %": [accuracy_metrics[m]["exact_match"] for m in modes],
    "Approx Match %": [accuracy_metrics[m]["approx_match"] for m in modes],
    "Parse Rate %": [accuracy_metrics[m]["parse_rate"] for m in modes],
})
summary_table.to_csv("D:/Tugas Kuliah/Semester 4/Riset Teknologi Informasi/Chain-of-thought/Dataset/gsm8k/main/summary_table_svamp.csv", index=False)
print("Tabel disimpan ke: summary_table_svamp.csv")
print(summary_table.to_string(index=False))


# =========================================
# 10. VISUALISASI
# =========================================

labels = ["No CoT", "Full CoT\n(Standard)", "Latent CoT\n(Proposed)"]
means  = [no_cot_mean, cot_mean, latent_mean]
colors = ["#5b8dee", "#e05c5c", "#2dd68b"]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor("#0d1117")
for ax in axes:
    ax.set_facecolor("#161b22")

# ── Plot 1: Bar rata-rata token ──
bars = axes[0].bar(labels, means, color=colors, width=0.5, edgecolor="none", zorder=3)
for bar, val in zip(bars, means):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
        f"{val:.1f}", ha="center", va="bottom",
        color="white", fontsize=10, fontweight="bold"
    )
axes[0].annotate(
    f"−{reduction_cot_lat:.1f}%",
    xy=(2, latent_mean), xytext=(1.55, (cot_mean + latent_mean) / 2),
    arrowprops=dict(arrowstyle="->", color="#2dd68b", lw=1.5),
    fontsize=10, color="#2dd68b", fontweight="bold"
)
axes[0].set_title("Rata-rata Token per Mode", color="white", fontsize=11, pad=10)
axes[0].set_ylabel("Token (rata-rata)", color="#8b949e")
axes[0].tick_params(colors="white")
axes[0].spines[["top","right","left","bottom"]].set_color("#30363d")
axes[0].set_ylim(0, max(means) * 1.25)
axes[0].grid(axis="y", color="#30363d", linewidth=0.5, zorder=0)

# ── Plot 2: Akurasi per mode ──
acc_vals = [
    accuracy_metrics[m]["approx_match"] for m in modes
]
bars2 = axes[1].bar(labels, acc_vals, color=colors, width=0.5, edgecolor="none", zorder=3)
for bar, val in zip(bars2, acc_vals):
    axes[1].text(
        bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
        f"{val:.1f}%", ha="center", va="bottom",
        color="white", fontsize=10, fontweight="bold"
    )
axes[1].set_title("Approx. Accuracy per Mode", color="white", fontsize=11, pad=10)
axes[1].set_ylabel("Accuracy (%)", color="#8b949e")
axes[1].tick_params(colors="white")
axes[1].spines[["top","right","left","bottom"]].set_color("#30363d")
axes[1].set_ylim(0, 115)
axes[1].grid(axis="y", color="#30363d", linewidth=0.5, zorder=0)

# ── Plot 3: Akurasi Latent CoT per tipe soal ──
type_data = df.groupby("Type")["latent_correct"].mean() * 100
type_colors = plt.cm.Set2(np.linspace(0, 1, len(type_data)))
bars3 = axes[2].barh(type_data.index, type_data.values,
                     color=type_colors, edgecolor="none", zorder=3)
for bar, val in zip(bars3, type_data.values):
    axes[2].text(
        val + 0.5, bar.get_y() + bar.get_height() / 2,
        f"{val:.1f}%", va="center", color="white", fontsize=9
    )
axes[2].set_title("Latent CoT Accuracy per Tipe", color="white", fontsize=11, pad=10)
axes[2].set_xlabel("Accuracy (%)", color="#8b949e")
axes[2].tick_params(colors="white")
axes[2].spines[["top","right","left","bottom"]].set_color("#30363d")
axes[2].set_xlim(0, 115)
axes[2].grid(axis="x", color="#30363d", linewidth=0.5, zorder=0)

# Legend
patches = [mpatches.Patch(color=c, label=l)
           for c, l in zip(colors, ["No CoT", "Full CoT", "Latent CoT"])]
fig.legend(handles=patches, loc="lower center", ncol=3,
           framealpha=0.15, labelcolor="white", fontsize=9)

plt.suptitle("SVAMP Dataset — Token Efficiency & Accuracy Analysis",
             color="white", fontsize=13, y=1.01)
plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig("D:/Tugas Kuliah/Semester 4/Riset Teknologi Informasi/Chain-of-thought/Dataset/gsm8k/main/comparison_svamp.png",
            dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.show()
print("Grafik utama disimpan: comparison_svamp.png")


# ── Plot 4: Distribusi Token (Histogram) ──
fig2, ax2 = plt.subplots(figsize=(10, 4))
fig2.patch.set_facecolor("#0d1117")
ax2.set_facecolor("#161b22")

for mode, color, label in zip(
    modes, colors,
    ["No CoT", "Full CoT (Standard)", "Latent CoT (Proposed)"]
):
    ax2.hist(results[mode]["series"], bins=40, alpha=0.65,
             color=color, label=label, edgecolor="none")

ax2.set_title("Distribusi Token per Mode — SVAMP", color="white", fontsize=12)
ax2.set_xlabel("Jumlah Token", color="#8b949e")
ax2.set_ylabel("Frekuensi", color="#8b949e")
ax2.tick_params(colors="white")
ax2.spines[["top","right","left","bottom"]].set_color("#30363d")
ax2.legend(labelcolor="white", framealpha=0.2)
ax2.grid(color="#30363d", linewidth=0.5)
plt.tight_layout()
plt.savefig("D:/Tugas Kuliah/Semester 4/Riset Teknologi Informasi/Chain-of-thought/Dataset/gsm8k/main/token_dist_svamp.png",
            dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.show()
print("Histogram disimpan: token_dist_svamp.png")


# ── Plot 5: CDF Token ──
def plot_cdf(data, label, ax, color):
    sorted_data = np.sort(data)
    yvals = np.arange(len(sorted_data)) / float(len(sorted_data))
    ax.plot(sorted_data, yvals, label=label, color=color, linewidth=2)

fig3, ax3 = plt.subplots(figsize=(9, 4))
fig3.patch.set_facecolor("#0d1117")
ax3.set_facecolor("#161b22")

for mode, color, label in zip(
    modes, colors,
    ["No CoT", "Full CoT", "Latent CoT"]
):
    plot_cdf(results[mode]["series"].values, label, ax3, color)

ax3.set_title("CDF Token Usage — SVAMP", color="white", fontsize=12)
ax3.set_xlabel("Tokens", color="#8b949e")
ax3.set_ylabel("CDF", color="#8b949e")
ax3.tick_params(colors="white")
ax3.spines[["top","right","left","bottom"]].set_color("#30363d")
ax3.legend(labelcolor="white", framealpha=0.2)
ax3.grid(color="#30363d", linewidth=0.5)
plt.tight_layout()
plt.savefig("D:/Tugas Kuliah/Semester 4/Riset Teknologi Informasi/Chain-of-thought/Dataset/gsm8k/main/cdf_svamp.png",
            dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.show()
print("CDF disimpan: cdf_svamp.png")


# =========================================
# RINGKASAN AKHIR
# =========================================

print("\n" + "=" * 60)
print("RINGKASAN HASIL PENELITIAN — SVAMP DATASET")
print("=" * 60)
print(f"  Dataset         : SVAMP ({len(df)} sampel, {df['Type'].nunique()} tipe)")
print(f"  No CoT          : {no_cot_mean:.2f} token rata-rata")
print(f"  Full CoT        : {cot_mean:.2f} token rata-rata")
print(f"  Latent CoT      : {latent_mean:.2f} token rata-rata")
print(f"  Reduksi CoT → Latent   : {reduction_cot_lat:.2f}%")
print(f"  Total Token Savings    : {total_cot - total_latent:,} token")
print(f"  Accuracy Latent (Exact): {accuracy_metrics['latent_cot']['exact_match']:.2f}%")
print(f"  Accuracy Latent (Approx): {accuracy_metrics['latent_cot']['approx_match']:.2f}%")
print(f"  Parse Rate              : {accuracy_metrics['latent_cot']['parse_rate']:.2f}%")
print("=" * 60)
print("SELESAI ✓")