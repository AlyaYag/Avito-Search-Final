import os
import random

import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


SEED = 42
INPUT_FILE = "candidate_data/calibration.f"
OUTPUT_QUERIES = "data/rewritten_queries.parquet"
OUTPUT_EMBEDDINGS = "data/calibration_query_embeddings_rewritten.parquet"
REWRITE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
EMBED_MODEL = "BAAI/bge-m3"
INSTRUCTION = "Represent this sentence for searching relevant passages: "
BATCH_SIZE = 4
MAX_NEW_TOKENS = 64

FEW_SHOT = [
    ("ой, извините, пожалуйста, я тут случайно удалила всё, что делать? помогите",
     "восстановление данных после случайного удаления"),
    ("здравствуйте, у меня не открывается сайт, в чем может быть проблема?",
     "сайт не открывается"),
    ("помогите, как сбросить счетчик просмотров, я нажимал на кнопку но ничего",
     "сброс счетчика просмотров"),
    ("привет! а можно удалить старый аккаунт и привязать новый номер?",
     "удаление аккаунта и смена номера телефона"),
    ("извините за беспокойство, а как привязать карту к аккаунту?",
     "привязка банковской карты к аккаунту"),
]

SYSTEM_PROMPT = (
    "Ты — система очистки поисковых запросов технической поддержки. "
    "Перепиши запрос пользователя, сохранив только ключевую суть проблемы. "
    "Убери извинения, вежливость, лишние детали."
)


def build_messages(query: str) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex_q, ex_r in FEW_SHOT:
        messages.append({"role": "user", "content": f"Запрос: {ex_q}"})
        messages.append({"role": "assistant", "content": ex_r})
    messages.append({"role": "user", "content": f"Запрос: {query}"})
    return messages


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rewrite_queries(query_texts: list[str]) -> list[str]:
    print(f"Loading rewrite model {REWRITE_MODEL} on CPU...")
    tokenizer = AutoTokenizer.from_pretrained(
        REWRITE_MODEL, padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(REWRITE_MODEL)
    model.eval()
    print("Model loaded")

    rewritten = []
    for start in range(0, len(query_texts), BATCH_SIZE):
        batch_queries = query_texts[start : start + BATCH_SIZE]
        messages_list = [build_messages(q) for q in batch_queries]
        texts = [tokenizer.apply_chat_template(msgs, tokenize=False) for msgs in messages_list]

        inputs = tokenizer(texts, padding=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
            )

        for i, out in enumerate(outputs):
            input_len = inputs["input_ids"].shape[1]
            new_tokens = out[input_len:]
            result = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            rewritten.append(result)

        print(f"  Rewritten {min(start + BATCH_SIZE, len(query_texts))}/{len(query_texts)}")

    return rewritten


def main():
    set_seed(SEED)

    df = pd.read_feather(INPUT_FILE)
    print(f"Loaded {len(df)} queries from {INPUT_FILE}")

    query_ids = df["query_id"].tolist()
    query_texts = df["query_text"].tolist()
    ground_truths = df["ground_truth"].tolist()

    rewritten = rewrite_queries(query_texts)

    table = pa.table({
        "query_id": pa.array(query_ids, type=pa.int64()),
        "query_text": pa.array(query_texts, type=pa.string()),
        "rewritten_text": pa.array(rewritten, type=pa.string()),
        "ground_truth": pa.array(ground_truths, type=pa.string()),
    })
    table.to_pandas().to_parquet(OUTPUT_QUERIES, index=False)
    print(f"Saved rewritten queries to {OUTPUT_QUERIES}")

    print(f"\nLoading embed model {EMBED_MODEL} on CPU...")
    embed_model = SentenceTransformer(EMBED_MODEL, device="cpu")
    print("Model loaded")

    prefixed = [INSTRUCTION + r for r in rewritten]
    print("Encoding rewritten queries...")
    embeddings = embed_model.encode(
        prefixed,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64,
    )
    print(f"Encoded {len(embeddings)} rewritten queries, shape: {embeddings.shape}")

    emb_table = pa.table({
        "query_id": pa.array(query_ids, type=pa.int64()),
        "query_text": pa.array(query_texts, type=pa.string()),
        "rewritten_text": pa.array(rewritten, type=pa.string()),
        "ground_truth": pa.array(ground_truths, type=pa.string()),
        "embedding": pa.array([emb.tobytes() for emb in embeddings], type=pa.binary()),
    })
    emb_table.to_pandas().to_parquet(OUTPUT_EMBEDDINGS, index=False)
    print(f"Saved rewritten query embeddings to {OUTPUT_EMBEDDINGS}")


if __name__ == "__main__":
    main()
