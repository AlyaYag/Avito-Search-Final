import numpy as np
import pandas as pd


METHODS = [
    ("dense_title", "results/runs/dense_title_per_query_ap.npy"),
    ("dense_body", "results/runs/dense_body_per_query_ap.npy"),
]
CALIB_PATH = "candidate_data/calibration.f"
ARTICLES_PATH = "data/articles_cleaned.f"
OUTPUT_PATH = "results/analysis_ap_zero.txt"
N_EXAMPLES = 15


def load_ap(path: str) -> dict[int, float]:
    arr = np.load(path)
    return {int(row["query_id"]): float(row["ap"]) for row in arr}


def main():
    calib = pd.read_feather(CALIB_PATH)
    articles = pd.read_feather(ARTICLES_PATH)

    qid_to_text = dict(zip(calib["query_id"], calib["query_text"]))
    qid_to_gt = dict(zip(calib["query_id"], calib["ground_truth"]))
    aid_to_title = dict(zip(articles["article_id"], articles["title"]))
    aid_to_body = dict(zip(articles["article_id"], articles["body"]))
    aid_to_len = {aid: len(body) for aid, body in aid_to_body.items()}

    lines = []

    for method_name, ap_path in METHODS:
        ap_dict = load_ap(ap_path)

        zero_qids = sorted(qid for qid, ap in ap_dict.items() if ap == 0.0)
        nonzero_qids = [qid for qid, ap in ap_dict.items() if ap > 0.0]

        ground_truth_aids = set()
        gt_len_vals = []
        for qid in zero_qids:
            gt_str = qid_to_gt.get(qid, "")
            aids = [int(x) for x in gt_str.split() if x.strip().isdigit()]
            for aid in aids:
                ground_truth_aids.add(aid)
                gt_len_vals.append(aid_to_len.get(aid, 0))

        lines.append("=" * 70)
        lines.append(f"METHOD: {method_name}")
        lines.append("=" * 70)
        lines.append(f"Total queries: {len(ap_dict)}")
        lines.append(f"  AP = 0: {len(zero_qids)} ({100 * len(zero_qids) / len(ap_dict):.1f}%)")
        lines.append(f"  AP > 0: {len(nonzero_qids)}")
        lines.append("")

        lines.append("--- Body length stats of ground-truth articles (AP=0 queries only) ---")
        if gt_len_vals:
            lines.append(f"  Count: {len(gt_len_vals)}")
            lines.append(f"  Mean:  {np.mean(gt_len_vals):.0f} chars")
            lines.append(f"  Median:{np.median(gt_len_vals):.0f} chars")
            lines.append(f"  Min:   {min(gt_len_vals)} chars")
            lines.append(f"  Max:   {max(gt_len_vals)} chars")
            bins = [0, 500, 1000, 2000, 5000, 10000, 20000, 50000]
            lines.append("  Distribution:")
            for lo, hi in zip(bins, bins[1:] + [float("inf")]):
                cnt = sum(1 for v in gt_len_vals if lo <= v < hi)
                if cnt:
                    hi_str = f"{hi}" if hi != float("inf") else "inf"
                    lines.append(f"    [{lo}, {hi_str}): {cnt}")
        else:
            lines.append("  (no data)")
        lines.append("")

        lines.append(f"--- First {N_EXAMPLES} queries with AP=0 ---")
        examples_shown = 0
        for qid in zero_qids:
            if examples_shown >= N_EXAMPLES:
                break
            examples_shown += 1
            query_text = qid_to_text.get(qid, "?")
            gt_str = qid_to_gt.get(qid, "")
            lines.append(f"query_id={qid}")
            lines.append(f"  query_text: {query_text}")
            lines.append(f"  ground_truth: {gt_str}")

            aids = [int(x) for x in gt_str.split() if x.strip().isdigit()]
            for aid in aids:
                title = aid_to_title.get(aid, "?")
                body = aid_to_body.get(aid, "")
                blen = len(body)
                lines.append(f"  article_id={aid}  title=«{title}»  body_len={blen}")
                lines.append(f"  --- body start ---")
                lines.append(body)
                lines.append(f"  --- body end ---")
            lines.append("")

    lines.append("=" * 70)
    lines.append("END")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Analysis saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
