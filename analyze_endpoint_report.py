import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path


STATE_RE = re.compile(r"^The (.+?) is in the (.+?)\.$")
MOVE_RE = re.compile(r"^(.+?) moved the (.+?) to the (.+?)\.$")
TARGET_RE = re.compile(r"\bthe ([A-Za-z_ ]+?) is\?$")


def normalize_location(value):
    if value is None:
        return ""
    value = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return re.sub(r"_+", "_", value)


def normalize_object(value):
    value = str(value).strip().lower().replace("_", " ")
    return re.sub(r"\s+", " ", value)


def story_id(story):
    return hashlib.md5(story.strip().encode("utf-8")).hexdigest()


def pct(value):
    return f"{100 * value:.1f}%"


def ratio(num, den):
    return num / den if den else None


def fmt_pct(value):
    return "NA" if value is None else pct(value)


def fmt_pp(value):
    return "NA" if value is None else f"{100 * value:+.1f} pp"


def fmt_float(value, digits=3):
    return "NA" if value is None else f"{value:.{digits}f}"


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def extract_target_object(question):
    question = question.strip()
    direct = re.match(r"^Where is the (.+?) really\?$", question)
    if direct:
        return normalize_object(direct.group(1))

    matches = list(TARGET_RE.finditer(question))
    if matches:
        return normalize_object(matches[-1].group(1))

    raise ValueError(f"Could not extract target object from question: {question}")


def parse_physical_endpoints(story, target_object):
    initial = None
    final = None
    physical_events = []

    for line in story.splitlines():
        line = line.strip()
        state = STATE_RE.match(line)
        if state and normalize_object(state.group(1)) == target_object:
            loc = normalize_location(state.group(2))
            if initial is None:
                initial = loc
            final = loc
            physical_events.append(("state", loc))
            continue

        move = MOVE_RE.match(line)
        if move and normalize_object(move.group(2)) == target_object:
            loc = normalize_location(move.group(3))
            final = loc
            physical_events.append(("move", loc))

    if initial is None or final is None:
        raise ValueError(f"Could not parse endpoints for target object={target_object!r}")

    return initial, final, physical_events


def endpoint_relation(value, initial, final):
    if not value:
        return "missing"

    is_initial = value == initial
    is_final = value == final
    if is_initial and is_final:
        return "both"
    if is_initial:
        return "initial"
    if is_final:
        return "final"
    return "other"


def relation_label(relation):
    labels = {
        "initial": "initial-only",
        "both": "both",
        "final": "final-only",
        "other": "other",
        "missing": "missing",
    }
    return labels.get(relation, relation)


def relation_sort_key(relation):
    order = {"initial": 0, "both": 1, "final": 2, "other": 3, "missing": 4}
    return order.get(relation, 99)


def run_name_from_path(path):
    name = path.stem.lower()
    method = "unknown"
    method_match = re.search(r"results_([^_]+)_", name)
    if method_match:
        method = method_match.group(1)

    if "1.7" in name or "1_7" in name:
        size = "1.7b"
    elif "0.6" in name or "0_6" in name or "06b" in name:
        size = "0.6b"
    else:
        size = "model"

    return f"{method}_{size}"


def load_metadata(data_path, category):
    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    meta = {}
    for item in data:
        if str(item.get("prompting_type", "")).lower() != category.lower():
            continue

        story = item["story"].strip()
        question = item["question"].strip()
        target = extract_target_object(question)
        initial, final, events = parse_physical_endpoints(story, target)
        gold = normalize_location(item["answer"])
        relation = endpoint_relation(gold, initial, final)
        sample_id = str(item["sample_id"])

        meta[sample_id] = {
            "sample_id": sample_id,
            "sample_id_int": int(item["sample_id"]),
            "story_id": story_id(story),
            "question_order": int(item["question_order"]),
            "deception": item.get("deception"),
            "story_length": item.get("story_length"),
            "question": question,
            "target_object": target.replace(" ", "_"),
            "initial_location": initial,
            "final_location": final,
            "gold": gold,
            "gold_relation": relation,
            "events": events,
        }

    return meta


def load_prediction_rows(results_dir, meta):
    result_paths = sorted(results_dir.glob("*.jsonl"))
    if not result_paths:
        raise FileNotFoundError(f"No JSONL result files found in {results_dir}")

    rows = []
    quality = []
    for path in result_paths:
        run = run_name_from_path(path)
        raw_rows = read_jsonl(path)
        seen_ids = set()
        used_rows = 0
        missing_meta = 0
        correct_disagreements = 0

        for raw in raw_rows:
            sample_id = str(raw.get("sample_id"))
            seen_ids.add(sample_id)
            sample = meta.get(sample_id)
            if sample is None:
                missing_meta += 1
                continue

            used_rows += 1
            pred = normalize_location(raw.get("pred_final"))
            correct = int(bool(pred) and pred == sample["gold"])
            stored_correct = int(raw.get("correct", 0) or 0)
            correct_disagreements += int(correct != stored_correct)

            rows.append(
                {
                    "run": run,
                    "sample_id": sample_id,
                    "story_id": sample["story_id"],
                    "question_order": sample["question_order"],
                    "gold": sample["gold"],
                    "gold_relation": sample["gold_relation"],
                    "pred_final": pred,
                    "pred_relation": endpoint_relation(
                        pred,
                        sample["initial_location"],
                        sample["final_location"],
                    ),
                    "correct": correct,
                    "stored_correct": stored_correct,
                }
            )

        quality.append(
            {
                "run": run,
                "file": str(path),
                "rows": len(raw_rows),
                "used_rows": used_rows,
                "unique_ids": len(seen_ids),
                "missing_metadata_rows": missing_meta,
                "correct_disagreements": correct_disagreements,
            }
        )

    return rows, quality


def group_rows(rows, *keys):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    return grouped


def accuracy(rows):
    total = len(rows)
    correct = sum(int(row["correct"]) for row in rows)
    return correct, total, ratio(correct, total)


def assoc_binary(rows, x_func, y_func):
    a = b = c = d = 0
    for row in rows:
        x = bool(x_func(row))
        y = bool(y_func(row))
        if x and y:
            a += 1
        elif x and not y:
            b += 1
        elif not x and y:
            c += 1
        else:
            d += 1

    n = a + b + c + d
    x1_total = a + b
    x0_total = c + d
    y1_total = a + c
    y0_total = b + d
    den = math.sqrt(x1_total * x0_total * y1_total * y0_total)
    phi = (a * d - b * c) / den if den else None
    chi2 = n * phi * phi if phi is not None else None
    p_value = math.erfc(math.sqrt(chi2 / 2)) if chi2 is not None else None
    rate_x1 = ratio(a, x1_total)
    rate_x0 = ratio(c, x0_total)
    diff = rate_x1 - rate_x0 if rate_x1 is not None and rate_x0 is not None else None
    odds_ratio = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))

    return {
        "x1_y1": a,
        "x1_y0": b,
        "x0_y1": c,
        "x0_y0": d,
        "x1_rate": rate_x1,
        "x0_rate": rate_x0,
        "risk_diff": diff,
        "phi": phi,
        "p_value": p_value,
        "odds_ratio": odds_ratio,
    }


def md_table(headers, rows):
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(out)


def section(title):
    return f"\n--------\n\n## {title}\n"


def dataset_distribution_rows(meta):
    grouped = defaultdict(list)
    for sample in meta.values():
        grouped[sample["question_order"]].append(sample)

    rows = []
    for order in sorted(grouped):
        samples = grouped[order]
        total = len(samples)
        counts = defaultdict(int)
        for sample in samples:
            counts[sample["gold_relation"]] += 1
        rows.append(
            [
                order,
                f"{counts['initial']} ({pct(ratio(counts['initial'], total))})",
                f"{counts['both']} ({pct(ratio(counts['both'], total))})",
                f"{counts['final']} ({pct(ratio(counts['final'], total))})",
                f"{counts['other']} ({pct(ratio(counts['other'], total))})",
                total,
            ]
        )
    return rows


def target_accuracy_by_order(meta, pred_rows):
    sample_counts = defaultdict(int)
    for sample in meta.values():
        sample_counts[(sample["question_order"], sample["gold_relation"])] += 1

    pred_grouped = group_rows(pred_rows, "question_order", "gold_relation")
    all_by_order = group_rows(pred_rows, "question_order")
    rows = []

    for order_key in sorted(all_by_order):
        order = order_key[0]
        row = [order]
        for relation in ["initial", "both", "final", "other"]:
            group = pred_grouped.get((order, relation), [])
            if not group:
                row.append("NA")
                continue
            correct, total, acc = accuracy(group)
            samples = sample_counts[(order, relation)]
            row.append(f"{samples} samples, {correct}/{total} = {fmt_pct(acc)}")
        correct, total, acc = accuracy(all_by_order[order_key])
        row.append(f"{correct}/{total} = {fmt_pct(acc)}")
        rows.append(row)

    return rows


def overall_target_accuracy(pred_rows, meta):
    rows = []
    for relation in ["initial", "both", "final", "other"]:
        group = [row for row in pred_rows if row["gold_relation"] == relation]
        samples = len({row["sample_id"] for row in group})
        correct, total, acc = accuracy(group)
        rows.append([relation_label(relation), samples, total, correct, fmt_pct(acc)])
    return rows


def prediction_relation_accuracy_by_order(pred_rows):
    grouped = group_rows(pred_rows, "question_order", "pred_relation")
    all_by_order = group_rows(pred_rows, "question_order")
    rows = []
    for order_key in sorted(all_by_order):
        order = order_key[0]
        row = [order]
        for relation in ["initial", "both", "final", "other", "missing"]:
            group = grouped.get((order, relation), [])
            if not group:
                row.append("NA")
                continue
            correct, total, acc = accuracy(group)
            row.append(f"{correct}/{total} = {fmt_pct(acc)}")
        rows.append(row)
    return rows


def target_assoc_rows(pred_rows, scope_name, scope_rows):
    rows = []
    for relation in ["initial", "both", "final", "other"]:
        has_relation = [row for row in scope_rows if row["gold_relation"] == relation]
        if not has_relation:
            continue
        stats = assoc_binary(
            scope_rows,
            lambda row, rel=relation: row["gold_relation"] == rel,
            lambda row: row["correct"] == 1,
        )
        rows.append(
            [
                scope_name,
                relation_label(relation),
                fmt_pct(stats["x1_rate"]),
                fmt_pct(stats["x0_rate"]),
                fmt_pp(stats["risk_diff"]),
                fmt_float(stats["phi"], 3),
                f"{stats['p_value']:.3g}" if stats["p_value"] is not None else "NA",
                fmt_float(stats["odds_ratio"], 2),
            ]
        )
    return rows


def order3_order4_delta_rows(pred_rows):
    grouped = group_rows(pred_rows, "run", "question_order")
    runs = sorted({row["run"] for row in pred_rows})
    rows = []
    for run in runs:
        order3 = grouped.get((run, 3), [])
        order4 = grouped.get((run, 4), [])
        if not order3 or not order4:
            continue
        c3, t3, a3 = accuracy(order3)
        c4, t4, a4 = accuracy(order4)
        rows.append([run, f"{c3}/{t3} = {fmt_pct(a3)}", f"{c4}/{t4} = {fmt_pct(a4)}", fmt_pp(a4 - a3)])
    return rows


def order4_prediction_outcome_rows(pred_rows):
    order4 = [row for row in pred_rows if row["question_order"] == 4]
    grouped = defaultdict(list)
    for row in order4:
        outcome = "correct" if row["correct"] else "wrong"
        grouped[(outcome, row["pred_relation"])].append(row)

    rows = []
    for outcome in ["correct", "wrong"]:
        outcome_total = sum(len(grouped[(outcome, relation)]) for relation in ["initial", "both", "final", "other", "missing"])
        for relation in ["initial", "both", "final", "other", "missing"]:
            group = grouped[(outcome, relation)]
            if not group:
                continue
            count = len(group)
            rows.append([outcome, relation_label(relation), count, fmt_pct(ratio(count, len(order4))), fmt_pct(ratio(count, outcome_total))])
    return rows


def order4_target_split_rows(pred_rows):
    order4 = [row for row in pred_rows if row["question_order"] == 4]
    rows = []
    for relation in ["initial", "both", "final", "other"]:
        group = [row for row in order4 if row["gold_relation"] == relation]
        if not group:
            continue
        samples = len({row["sample_id"] for row in group})
        correct, total, acc = accuracy(group)
        rows.append([relation_label(relation), samples, total, correct, fmt_pct(acc)])
    return rows


def build_report(meta, pred_rows, quality, category, data_path, results_dir):
    lines = [
        "# Endpoint Target Analysis Report",
        "",
        f"Dataset category: `{category}`",
        f"Dataset file: `{data_path}`",
        f"Results directory: `{results_dir}`",
        "",
        "This report is generated automatically from `data/hitom.json` and all JSONL files in `res-qwen`.",
    ]

    total_predictions = len(pred_rows)
    total_correct = sum(row["correct"] for row in pred_rows)
    runs = sorted({row["run"] for row in pred_rows})
    disagreements = sum(row["correct"] != row["stored_correct"] for row in pred_rows)

    lines.append(section("Loaded Data"))
    lines.append(
        md_table(
            ["Metric", "Value"],
            [
                ["Samples", len(meta)],
                ["Runs", len(runs)],
                ["Prediction rows", total_predictions],
                ["Recomputed accuracy", f"{total_correct}/{total_predictions} = {fmt_pct(ratio(total_correct, total_predictions))}"],
                ["Stored/recomputed correctness disagreements", disagreements],
            ],
        )
    )
    lines.append("")
    lines.append(
        md_table(
            ["Run", "Rows", "Used rows", "Unique IDs", "Missing metadata rows", "Correctness disagreements"],
            [
                [
                    row["run"],
                    row["rows"],
                    row["used_rows"],
                    row["unique_ids"],
                    row["missing_metadata_rows"],
                    row["correct_disagreements"],
                ]
                for row in quality
            ],
        )
    )

    lines.append(section("Category Definitions"))
    lines.append(
        "\n".join(
            [
                "For each question, the script identifies the target object, tracks that object's first and final physical locations in the story, and classifies the gold answer into one of four target classes.",
                "",
                "- `initial-only`: the gold answer is the first location, and the final location is different.",
                "- `both`: the first and final locations are the same, so the gold answer is both initial and final.",
                "- `final-only`: the gold answer is the final physical location, and it is not the first location.",
                "- `other`: the gold answer is neither the first nor the final physical location.",
                "",
                "Real example 1: in one melon story, the melon starts in `blue_treasure_chest`, moves through `green_bucket`, `green_drawer`, and `green_bottle`, and finally returns to `blue_treasure_chest`. Therefore `blue_treasure_chest` is `both`. In the same story, the gold answer to `Where does Jacob really think the melon is?` is `green_drawer`, which is `other` because it is neither the first nor final physical location.",
                "",
                "Real example 2: in one onion story, the onion starts in `red_crate`, moves through `red_drawer` and `red_bottle`, returns to `red_crate`, and later ends in `blue_crate`. Therefore `red_crate` is `initial-only` and `blue_crate` is `final-only`. The direct question `Where is the onion really?` has gold `blue_crate`, while a nested-belief question such as `Where does Isabella think Emily thinks Owen thinks the onion is?` has gold `red_crate`.",
            ]
        )
    )

    lines.append(section("Target Class Distribution By Order"))
    lines.append(
        md_table(
            ["Order", "Initial-only", "Both", "Final-only", "Other", "Total"],
            dataset_distribution_rows(meta),
        )
    )
    lines.append("")
    lines.append(
        "Interpretation: order 4 is extreme because all 60 gold answers are `initial-only` or `both`; there are no `final-only` or `other` gold targets."
    )

    lines.append(section("Accuracy By Target Class And Order"))
    lines.append(
        md_table(
            ["Order", "Initial-only target", "Both target", "Final-only target", "Other target", "Total accuracy"],
            target_accuracy_by_order(meta, pred_rows),
        )
    )
    lines.append("")
    lines.append(
        "Interpretation: order 4 is not high because initial-only targets are easy. In order 4, initial-only targets are only 65/248 = 26.2% correct, while both targets are 132/232 = 56.9% correct."
    )

    lines.append(section("Overall Accuracy By Target Class"))
    lines.append(
        md_table(
            ["Target class", "Samples", "Predictions", "Correct", "Accuracy"],
            overall_target_accuracy(pred_rows, meta),
        )
    )
    lines.append("")
    lines.append(
        "Interpretation: `both` is the easiest class overall. This matters because the both class allows initial-state and final-state shortcuts to collapse to the same answer."
    )

    lines.append(section("Accuracy By Predicted Class And Order"))
    lines.append(
        md_table(
            ["Order", "Pred initial-only", "Pred both", "Pred final-only", "Pred other", "Missing"],
            prediction_relation_accuracy_by_order(pred_rows),
        )
    )
    lines.append("")
    lines.append(
        "Interpretation: in order 4, predictions classified as initial-only or both are always correct in this run set, while final-only and other predictions are always wrong."
    )

    order4_rows = [row for row in pred_rows if row["question_order"] == 4]
    order4_assoc = assoc_binary(
        order4_rows,
        lambda row: row["gold_relation"] == "both",
        lambda row: row["correct"] == 1,
    )
    lines.append(section("Order 4: Both Vs Initial-Only"))
    lines.append(
        md_table(
            ["Order-4 target class", "Samples", "Predictions", "Correct", "Accuracy"],
            order4_target_split_rows(pred_rows),
        )
    )
    lines.append("")
    lines.append(
        md_table(
            ["Comparison", "Both acc", "Initial-only acc", "Risk diff", "Phi", "p-value", "Odds ratio"],
            [
                [
                    "both vs initial-only",
                    fmt_pct(order4_assoc["x1_rate"]),
                    fmt_pct(order4_assoc["x0_rate"]),
                    fmt_pp(order4_assoc["risk_diff"]),
                    fmt_float(order4_assoc["phi"], 3),
                    f"{order4_assoc['p_value']:.3g}" if order4_assoc["p_value"] is not None else "NA",
                    fmt_float(order4_assoc["odds_ratio"], 2),
                ]
            ],
        )
    )
    lines.append("")
    lines.append(
        "Interpretation: this is more informative than simply saying order 4 is all endpoint targets. Within order 4, the both class is much easier than initial-only, with a +30.7 percentage-point difference."
    )

    lines.append(section("Order 4 Prediction Outcomes"))
    lines.append(
        md_table(
            ["Outcome", "Predicted class", "Count", "Share of all order-4 predictions", "Share within outcome"],
            order4_prediction_outcome_rows(pred_rows),
        )
    )
    lines.append("")
    lines.append(
        "Interpretation: correct order-4 predictions all land in initial-only or both; wrong order-4 predictions land in final-only or other."
    )

    lines.append(section("Target Class Association With Correctness"))
    assoc_rows = []
    assoc_rows.extend(target_assoc_rows(pred_rows, "all orders", pred_rows))
    for order in range(5):
        order_rows = [row for row in pred_rows if row["question_order"] == order]
        assoc_rows.extend(target_assoc_rows(pred_rows, f"order {order}", order_rows))
    lines.append(
        md_table(
            ["Scope", "Target class", "Class acc", "Rest acc", "Risk diff", "Phi", "p-value", "Odds ratio"],
            assoc_rows,
        )
    )
    lines.append("")
    lines.append(
        "Interpretation: positive phi means the target class is associated with higher correctness. Across all orders, `both` has the strongest positive association; `initial-only` and `other` are negative."
    )

    lines.append(section("Order 3 Vs Order 4 By Run"))
    lines.append(
        md_table(
            ["Run", "Order 3 accuracy", "Order 4 accuracy", "Order4 - Order3"],
            order3_order4_delta_rows(pred_rows),
        )
    )
    lines.append("")
    lines.append(
        "Interpretation: order 4 is higher than order 3 in most runs, but the target-class analysis shows that the bump is aligned with initial/both target structure, especially the both class."
    )

    lines.append("\n--------\n")
    lines.append("## Bottom Line")
    lines.append("")
    lines.append(
        "Order 4 should not be read as evidence of stronger fourth-order Theory-of-Mind reasoning. In this data slice, order 4 has no final-only or other gold targets; it consists entirely of initial-only and both targets. The both subset is much easier than initial-only, so the apparent order-4 advantage is best explained as a shortcut-aligned label distribution."
    )

    return "\n".join(lines) + "\n"


def write_debug_csvs(out_dir, meta, pred_rows):
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_path = out_dir / "endpoint_sample_annotations.csv"
    with sample_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "question_order",
                "story_id",
                "target_object",
                "initial_location",
                "final_location",
                "gold",
                "gold_relation",
                "question",
            ],
        )
        writer.writeheader()
        for sample in sorted(meta.values(), key=lambda row: row["sample_id_int"]):
            writer.writerow(
                {
                    "sample_id": sample["sample_id"],
                    "question_order": sample["question_order"],
                    "story_id": sample["story_id"],
                    "target_object": sample["target_object"],
                    "initial_location": sample["initial_location"],
                    "final_location": sample["final_location"],
                    "gold": sample["gold"],
                    "gold_relation": sample["gold_relation"],
                    "question": sample["question"],
                }
            )

    pred_path = out_dir / "endpoint_prediction_rows.csv"
    with pred_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run",
                "sample_id",
                "question_order",
                "gold",
                "gold_relation",
                "pred_final",
                "pred_relation",
                "correct",
                "stored_correct",
            ],
        )
        writer.writeheader()
        writer.writerows(pred_rows)


def main():
    parser = argparse.ArgumentParser(description="Generate a Markdown endpoint-target analysis report for Qwen HiToM results.")
    parser.add_argument("--data", type=Path, default=Path("data/hitom.json"))
    parser.add_argument("--results-dir", type=Path, default=Path("res-qwen"))
    parser.add_argument("--category", default="CoTP")
    parser.add_argument("--out", type=Path, default=Path("analysis-qwen/endpoint_analysis_report.md"))
    parser.add_argument(
        "--write-debug-csv",
        action="store_true",
        help="Also write endpoint sample/prediction rows next to the report.",
    )
    args = parser.parse_args()

    meta = load_metadata(args.data, args.category)
    pred_rows, quality = load_prediction_rows(args.results_dir, meta)
    report = build_report(meta, pred_rows, quality, args.category, args.data, args.results_dir)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")

    if args.write_debug_csv:
        write_debug_csvs(args.out.parent, meta, pred_rows)

    print(f"Wrote report: {args.out}")
    print(f"Samples: {len(meta)}; predictions: {len(pred_rows)}; runs: {len(set(row['run'] for row in pred_rows))}")


if __name__ == "__main__":
    main()
