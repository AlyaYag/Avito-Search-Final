import os
from datetime import datetime
from typing import Callable

import numpy as np
import pandas as pd

from src.utils import average_precision_at_k


RUNS_DIR = "results/runs"


def load_embeddings(path: str) -> tuple[np.ndarray, pd.DataFrame]:
    df = pd.read_parquet(path)
    emb = np.stack([np.frombuffer(b) for b in df["embedding"].values])
    return emb, df


def validate(
    article_emb_path: str,
    query_emb_path: str,
    ranker: Callable,
    method_name: str,
    top_k: int = 10,
) -> float:
    os.makedirs(RUNS_DIR, exist_ok=True)

    article_emb, article_df = load_embeddings(article_emb_path)
    query_emb, query_df = load_embeddings(query_emb_path)

    article_ids = article_df["article_id"].values
    query_ids = query_df["query_id"].values
    query_texts = query_df["query_text"].tolist()
    ground_truths = query_df["ground_truth"].tolist()

    rankings = ranker(article_emb, article_ids, query_emb, top_k=top_k)

    per_query_ap = {}
    for qid, gt_str, ranking in zip(query_ids, ground_truths, rankings):
        gt = {int(x) for x in gt_str.split()} if isinstance(gt_str, str) else set()
        ap = average_precision_at_k(gt, ranking, top_k)
        per_query_ap[int(qid)] = ap

    map_score = float(np.mean(list(per_query_ap.values())))

    article_stats: dict[int, dict] = {}
    for aid in article_ids:
        article_stats[int(aid)] = {"retrieved": 0, "relevant": 0, "retrieved_and_relevant": 0}

    for gt_str, ranking in zip(ground_truths, rankings):
        gt = {int(x) for x in gt_str.split()} if isinstance(gt_str, str) else set()
        for aid in ranking:
            article_stats[int(aid)]["retrieved"] += 1
            if aid in gt:
                article_stats[int(aid)]["retrieved_and_relevant"] += 1
        for aid in gt:
            article_stats[int(aid)]["relevant"] += 1

    log_path = os.path.join(RUNS_DIR, f"{method_name}.txt")
    with open(log_path, "w", encoding="utf-8") as f:

        f.write(f"Method: {method_name}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"MAP@{top_k}: {map_score:.4f}\n")
        f.write(f"Queries total: {len(query_ids)}\n\n")

        f.write("=" * 60 + "\n")
        f.write("PER-QUERY AP\n")
        f.write("=" * 60 + "\n")
        for qid in query_ids:
            qid_int = int(qid)
            f.write(f"query_id={qid_int}  AP={per_query_ap[qid_int]:.4f}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("WORST QUERIES (lowest AP)\n")
        f.write("=" * 60 + "\n")

        sorted_queries = sorted(per_query_ap.items(), key=lambda x: x[1])
        query_id_to_text = {int(qid): qt for qid, qt in zip(query_ids, query_texts)}
        query_id_to_gt = {int(qid): gt for qid, gt in zip(query_ids, ground_truths)}
        query_id_to_ranking = {int(qid): r for qid, r in zip(query_ids, rankings)}

        for qid, ap in sorted_queries[:10]:
            f.write(f"\nquery_id={qid}  AP={ap:.4f}\n")
            f.write(f"  text:     {query_id_to_text[qid]}\n")
            f.write(f"  expected: {query_id_to_gt[qid]}\n")
            f.write(f"  got:      {' '.join(str(a) for a in query_id_to_ranking[qid])}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("BEST QUERIES (highest AP)\n")
        f.write("=" * 60 + "\n")
        for qid, ap in sorted_queries[-10:]:
            f.write(f"\nquery_id={qid}  AP={ap:.4f}\n")
            f.write(f"  text:     {query_id_to_text[qid]}\n")
            f.write(f"  expected: {query_id_to_gt[qid]}\n")
            f.write(f"  got:      {' '.join(str(a) for a in query_id_to_ranking[qid])}\n")

        title_map = dict(zip(article_df["article_id"], article_df["title"]))

        f.write("\n" + "=" * 60 + "\n")
        f.write("WORST ARTICLES (most false positive retrievals)\n")
        f.write("=" * 60 + "\n")
        articles_fp = sorted(
            article_stats.items(),
            key=lambda x: x[1]["retrieved"] - x[1]["retrieved_and_relevant"],
            reverse=True,
        )
        for aid, stats in articles_fp[:10]:
            fp = stats["retrieved"] - stats["retrieved_and_relevant"]
            if fp == 0:
                break
            title = title_map.get(aid, "?")
            f.write(f"\narticle_id={aid}  «{title}»\n")
            f.write(f"  retrieved: {stats['retrieved']}, "
                    f"relevant: {stats['relevant']}, "
                    f"FP: {fp}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("WORST ARTICLES (relevant but rarely found)\n")
        f.write("=" * 60 + "\n")
        articles_missed = sorted(
            article_stats.items(),
            key=lambda x: x[1]["relevant"] - x[1]["retrieved_and_relevant"],
            reverse=True,
        )
        for aid, stats in articles_missed[:10]:
            missed = stats["relevant"] - stats["retrieved_and_relevant"]
            if missed == 0:
                break
            title = title_map.get(aid, "?")
            recall = stats["retrieved_and_relevant"] / stats["relevant"] if stats["relevant"] > 0 else 0
            f.write(f"\narticle_id={aid}  «{title}»\n")
            f.write(f"  relevant for {stats['relevant']} queries, "
                    f"found {stats['retrieved_and_relevant']} times, "
                    f"missed: {missed}, recall: {recall:.2f}\n")

        dist = {}
        for ap in per_query_ap.values():
            bucket = int(ap * 10) / 10
            dist[bucket] = dist.get(bucket, 0) + 1

        f.write("\n" + "=" * 60 + "\n")
        f.write("AP DISTRIBUTION\n")
        f.write("=" * 60 + "\n")
        for bucket in sorted(dist.keys(), reverse=True):
            bar = "#" * dist[bucket]
            f.write(f"  AP={bucket:.1f}: {dist[bucket]:4d}  {bar}\n")

    print(f"\nMAP@{top_k}: {map_score:.4f}")
    print(f"Log saved to {log_path}")

    return map_score
