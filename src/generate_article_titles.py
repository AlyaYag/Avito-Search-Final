import random

import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


SEED = 42
INPUT_PATH = "data/articles_cleaned.f"
OUTPUT_PATH = "data/generated_titles.parquet"
OUTPUT_EMBEDDINGS = "data/generated_title_embeddings.parquet"
REWRITE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
EMBED_MODEL = "BAAI/bge-m3"
HEAD_SIZE = 1500
BATCH_SIZE_LLM = 8
MAX_NEW_TOKENS = 32

FEW_SHOT = [
    ("У вас будет 2 рабочих дня, чтобы отправить товар (не считая день оплаты заказа). Если не успеваете — продлите отправку на 4 календарных дня по кнопке Продлить срок отправки на странице заказа.",
     "отправка товара продавцом после оплаты заказа"),
    ("Как продавцу подключить скидки. Можно оплатить доставку для покупателя. Тогда на карточке товара и в результатах поиска появится отметка о скидке на доставку.",
     "настройка скидки на доставку для продавца"),
    ("Авито Доставка работает в разных регионах России. Доставку оплачивает покупатель, а продавец платит комиссию за продажу с Авито Доставкой.",
     "условия работы и комиссия сервиса доставки"),
    ("Оплатить товары с Авито Доставкой можно несколькими способами: картой, СБП, бонусами, балансом для покупок или при получении.",
     "способы оплаты заказа при доставке"),
    ("Когда покупатель заберёт заказ, вы сможете получить деньги. Откройте завершённый заказ и нажмите Получить оплату.",
     "получение денег продавцом после доставки"),
]

SYSTEM_PROMPT = (
    "Ты — система генерации кратких описаний статей справки. "
    "По началу статьи напиши короткую фразу (3-10 слов), "
    "которая отражает ключевую суть проблемы или инструкции. "
    "Используй ключевые термины, как если бы пользователь искал эту статью."
)


def build_messages(head: str) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex_head, ex_generated in FEW_SHOT:
        messages.append({"role": "user", "content": f"Начало статьи: {ex_head}"})
        messages.append({"role": "assistant", "content": ex_generated})
    messages.append({"role": "user", "content": f"Начало статьи: {head}"})
    return messages


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def generate(articles: pd.DataFrame) -> list[str]:
    heads = [b[:HEAD_SIZE] for b in articles["body"].tolist()]

    print(f"Loading model {REWRITE_MODEL} on CPU...")
    tokenizer = AutoTokenizer.from_pretrained(REWRITE_MODEL, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(REWRITE_MODEL)
    model.eval()
    print("Model loaded")

    generated = []
    for start in range(0, len(heads), BATCH_SIZE_LLM):
        batch = heads[start : start + BATCH_SIZE_LLM]
        messages_list = [build_messages(h) for h in batch]
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
            result = tokenizer.decode(out[input_len:], skip_special_tokens=True).strip()
            generated.append(result)

        print(f"  Generated {min(start + BATCH_SIZE_LLM, len(heads))}/{len(heads)}")

    return generated


def main():
    set_seed(SEED)

    articles = pd.read_feather(INPUT_PATH)
    print(f"Loaded {len(articles)} articles from {INPUT_PATH}")

    generated_titles = generate(articles)

    table = pa.table({
        "article_id": pa.array(articles["article_id"].tolist(), type=pa.int64()),
        "generated_title": pa.array(generated_titles, type=pa.string()),
    })
    df_out = table.to_pandas()
    df_out.to_feather("data/generated_titles.f")
    print("Saved generated titles to data/generated_titles.f")

    print(f"\nLoading embed model {EMBED_MODEL} on CPU...")
    embed_model = SentenceTransformer(EMBED_MODEL, device="cpu")
    print("Model loaded")

    print("Encoding generated titles...")
    embeddings = embed_model.encode(
        generated_titles,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64,
    )
    print(f"Encoded {len(embeddings)} titles, shape: {embeddings.shape}")

    emb_table = pa.table({
        "article_id": pa.array(articles["article_id"].tolist(), type=pa.int64()),
        "generated_title": pa.array(generated_titles, type=pa.string()),
        "embedding": pa.array([emb.tobytes() for emb in embeddings], type=pa.binary()),
    })
    emb_table.to_pandas().to_parquet(OUTPUT_EMBEDDINGS, index=False)
    print(f"Saved embeddings to {OUTPUT_EMBEDDINGS}")


if __name__ == "__main__":
    main()
