from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

import torch
import pandas as pd

# ----------------------------------------------------------
# 0. Config
# ----------------------------------------------------------

CSV_PATH = "./data/crisisbench/preprocessed_data_test.csv"
OUTPUT_PATH = "/Users/jade/Documents/7643_DL/dl-twitter-crisis/data/crisisbench/preprocessed_data_test_with_llama.csv"

LABELS = ["non_informative", "time_critical", "support_and_relief"]

LIMIT = 1000          # only classify first 1000 rows
BATCH_SIZE = 8        # how many tweets per generate() call
WRITE_CHUNK = 50      # how many predictions to buffer before writing to CSV

MAX_NEW_TOKENS = 4    # just need the label, so keep this small

# ----------------------------------------------------------
# 1. Load model & tokenizer
# ----------------------------------------------------------

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.config.pad_token_id = tokenizer.eos_token_id
model.eval()

# Optional sanity check – comment this out once you trust the setup
# prompt = "You are a helpful assistant. Who are you?"
# inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
# outputs = model.generate(**inputs, max_new_tokens=20)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# print("-" * 80)

# ----------------------------------------------------------
# 2. Prompt + parsing helpers (no chat template)
# ----------------------------------------------------------

def build_prompt(text: str) -> str:
    labels_str = ", ".join(LABELS)
    return (
        "You are a classifier for crisis-related tweets.\n"
        f"Possible labels: {labels_str}.\n\n"
        f"Tweet:\n{text}\n\n"
        "Choose exactly ONE label from the list above.\n"
        "Reply with only the label text, nothing else.\n"
        "Label:"
    )

def parse_label(raw_output: str) -> str:
    s = raw_output.strip().splitlines()[0]
    s = s.split()[0].strip(",.:- ").lower()
    label_map = {l.lower(): l for l in LABELS}
    return label_map.get(s, LABELS[0])  # default to first label if weird


@torch.no_grad()
def classify_batch(texts):
    """
    Classify a batch of texts. Returns a list of labels (same order as texts).
    """
    prompts = [build_prompt(t) for t in texts]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    preds = []
    for prompt, output_ids in zip(prompts, outputs):
        full_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        # Take text after the last "Label:"
        if "Label:" in full_text:
            gen_part = full_text.split("Label:")[-1]
        else:
            # If for some reason the model didn't echo the "Label:", just use entire output
            gen_part = full_text
        preds.append(parse_label(gen_part))
    return preds

# ----------------------------------------------------------
# 3. Load CSV and classify first LIMIT rows (batched + chunked write)
# ----------------------------------------------------------

print(f"Loading test CSV from: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)

if "text" not in df.columns:
    raise ValueError("Expected a 'text' column in the CSV.")

if "class_label_group" in df.columns:
    df = df.dropna(subset=["text", "class_label_group"]).reset_index(drop=True)
else:
    df = df.dropna(subset=["text"]).reset_index(drop=True)

total_rows = len(df)
limit = min(LIMIT, total_rows)
print(f"Total rows after dropna: {total_rows}")
print(f"Will classify first {limit} rows")

# Create output CSV with header once
header_df = pd.DataFrame(columns=["text", "class_label_group", "llama_pred_label"])
header_df.to_csv(OUTPUT_PATH, index=False)

buffer_rows = []   # we’ll buffer some predictions before writing

for start in range(0, limit, BATCH_SIZE):
    end = min(start + BATCH_SIZE, limit)
    batch_df = df.iloc[start:end]
    texts = batch_df["text"].astype(str).tolist()
    true_labels = batch_df.get("class_label_group", [None] * len(texts))

    preds = classify_batch(texts)

    for text, true_label, pred in zip(texts, true_labels, preds):
        buffer_rows.append({
            "text": text,
            "class_label_group": true_label,
            "llama_pred_label": pred,
        })

    # Write to disk every WRITE_CHUNK rows
    if len(buffer_rows) >= WRITE_CHUNK:
        out_df = pd.DataFrame(buffer_rows)
        out_df.to_csv(OUTPUT_PATH, mode="a", index=False, header=False)
        buffer_rows = []

    # Progress update
    print(f"Processed {end}/{limit} rows")

# Flush any remaining rows
if buffer_rows:
    out_df = pd.DataFrame(buffer_rows)
    out_df.to_csv(OUTPUT_PATH, mode="a", index=False, header=False)

print(f"\nDONE — predictions saved to:\n{OUTPUT_PATH}")

# ----------------------------------------------------------
# 4. Evaluation: read OUTPUT csv and compute metrics
# ----------------------------------------------------------

pred_df = pd.read_csv(OUTPUT_PATH)

if "class_label_group" not in pred_df.columns:
    raise ValueError("Expected ground-truth label column 'class_label_group' in the predictions CSV.")

if "llama_pred_label" not in pred_df.columns:
    raise ValueError("Expected prediction column 'llama_pred_label' in the predictions CSV.")

eval_df = pred_df.dropna(subset=["class_label_group", "llama_pred_label"]).reset_index(drop=True)

y_true_str = eval_df["class_label_group"].astype(str).tolist()
y_pred_str = eval_df["llama_pred_label"].astype(str).tolist()

labels = sorted(list(set(y_true_str) | set(y_pred_str)))
label2id = {lab: i for i, lab in enumerate(labels)}
id2label = {i: lab for lab, i in label2id.items()}

y_true = [label2id[l] for l in y_true_str]
y_pred = [label2id.get(l, 0) for l in y_pred_str]

acc = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true,
    y_pred,
    average="macro",
    zero_division=0,
)

print("\n=== Llama 3 8B Zero-Shot Evaluation on First "
      f"{len(eval_df)} CrisisBench Test Samples ===")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 (macro):{f1:.4f}\n")

print("Per-class metrics:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=[id2label[i] for i in range(len(labels))],
        digits=4,
        zero_division=0,
    )
)

