"""
Comprehensive Analysis Script for Theory-of-Mind Experiment Results

This script analyzes JSONL experiment result files to provide:
1. Standard metrics (accuracy by method, model, question_order, etc.)
2. Advanced insights for designing improved methods
   - "Impossible" questions (never correctly answered by any model/method)
   - "Easy" questions (always correctly answered by all models)
   - Method complementarity (which methods succeed where others fail)
   - Error pattern analysis
   - Model agreement/disagreement analysis
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict, Counter
import csv


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dictionaries."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_filename(filename: str) -> Dict[str, str]:
    """
    Parse experiment result filename to extract metadata.
    Expected format: hitom_cotp_results_<method>_<model>.jsonl
    """
    parts = filename.replace(".jsonl", "").split("_")
    info = {
        "category": "unknown",
        "method": "unknown",
        "model": "unknown",
        "full_name": filename
    }

    # Try to extract category (CoTP or VR)
    if "cotp" in filename.lower():
        info["category"] = "CoTP"
    elif "vr" in filename.lower():
        info["category"] = "VR"

    # Extract method
    method_keywords = ["vp", "soo", "simtom", "perceptom", "dtom"]
    for i, part in enumerate(parts):
        part_lower = part.lower()
        for method in method_keywords:
            if method in part_lower:
                info["method"] = method.upper()
                break

    # Extract model size/info
    if "qwen" in filename.lower():
        if "1.7" in filename or "1_7" in filename:
            info["model"] = "Qwen3-1.7B"
        elif "0.6" in filename or "06" in filename or "0_6" in filename:
            info["model"] = "Qwen3-0.6B"
        elif "4" in filename:
            info["model"] = "Qwen3-4B"
        else:
            info["model"] = "Qwen3-unknown"

    return info


class TableFormatter:
    """Simple table formatting for display."""

    @staticmethod
    def format_table(headers: List[str], rows: List[List[Any]]) -> str:
        """Format data as a simple text table."""
        if not rows:
            return "No data"

        # Convert all to strings and find column widths
        str_rows = [[str(cell) for cell in row] for row in rows]
        all_data = [headers] + str_rows
        col_widths = [max(len(row[i]) for row in all_data) for i in range(len(headers))]

        # Build output
        lines = []

        # Header row
        header_cells = [headers[i].ljust(col_widths[i]) for i in range(len(headers))]
        lines.append(" | ".join(header_cells))
        lines.append("-" * len(lines[0]))

        # Data rows
        for row in str_rows:
            cells = [row[i].ljust(col_widths[i]) for i in range(len(headers))]
            lines.append(" | ".join(cells))

        return "\n".join(lines)


class ExperimentAnalyzer:
    """Analyzer for ToM experiment results."""

    def __init__(self, experiment_dir: str):
        self.experiment_dir = Path(experiment_dir)
        self.all_results: List[Dict[str, Any]] = []
        self.experiments: Dict[str, List[Dict]] = {}
        self.sample_registry: Dict[str, Dict] = {}  # sample_id -> sample info

    def load_all(self):
        """Load all experiment result files."""
        jsonl_files = list(self.experiment_dir.glob("*.jsonl"))

        print(f"Found {len(jsonl_files)} experiment result files:")
        for f in sorted(jsonl_files):
            meta = parse_filename(f.name)
            exp_key = f"{meta['method']}_{meta['model']}"
            print(f"  - {f.name} -> {exp_key}")

            results = load_jsonl(str(f))
            self.experiments[exp_key] = results

            for r in results:
                r["_exp_key"] = exp_key
                r["_filename"] = f.name
                self.all_results.append(r)

                # Register sample info
                sid = str(r.get("sample_id", ""))
                if sid and sid not in self.sample_registry:
                    self.sample_registry[sid] = {
                        "sample_id": sid,
                        "story_id": r.get("story_id"),
                        "question": r.get("question"),
                        "answer": r.get("answer"),
                        "question_order": r.get("question_order"),
                        "deception": r.get("deception"),
                        "story_length": r.get("story_length"),
                    }

        print(f"\nTotal results loaded: {len(self.all_results)}")
        print(f"Unique samples: {len(self.sample_registry)}")

    # ==================== STANDARD METRICS ====================

    def compute_basic_stats(self) -> Tuple[List[str], List[List[Any]]]:
        """Compute basic accuracy statistics by experiment."""
        headers = ["experiment", "total", "correct", "accuracy", "order_0", "order_1", "order_2", "order_3", "order_4", "deception_true", "deception_false"]
        rows = []

        for exp_key, results in sorted(self.experiments.items()):
            correct = sum(int(r.get("correct", 0)) for r in results)
            total = len(results)
            accuracy = correct / total if total > 0 else 0

            # Breakdown by question order
            order_stats = defaultdict(lambda: {"correct": 0, "total": 0})
            for r in results:
                order = r.get("question_order", -1)
                order_stats[order]["total"] += 1
                order_stats[order]["correct"] += int(r.get("correct", 0))

            # Breakdown by deception
            deception_stats = {"true": {"correct": 0, "total": 0},
                              "false": {"correct": 0, "total": 0}}
            for r in results:
                deception = str(r.get("deception", "")).lower() == "true"
                key = "true" if deception else "false"
                deception_stats[key]["total"] += 1
                deception_stats[key]["correct"] += int(r.get("correct", 0))

            row = [
                exp_key,
                total,
                correct,
                f"{accuracy:.2%}",
                f"{order_stats[0]['correct']}/{order_stats[0]['total']}" if 0 in order_stats else "N/A",
                f"{order_stats[1]['correct']}/{order_stats[1]['total']}" if 1 in order_stats else "N/A",
                f"{order_stats[2]['correct']}/{order_stats[2]['total']}" if 2 in order_stats else "N/A",
                f"{order_stats[3]['correct']}/{order_stats[3]['total']}" if 3 in order_stats else "N/A",
                f"{order_stats[4]['correct']}/{order_stats[4]['total']}" if 4 in order_stats else "N/A",
                f"{deception_stats['true']['correct']}/{deception_stats['true']['total']}",
                f"{deception_stats['false']['correct']}/{deception_stats['false']['total']}",
            ]
            rows.append(row)

        return headers, rows

    def compute_accuracy_by_method(self) -> Tuple[List[str], List[List[Any]]]:
        """Compare methods across all models."""
        method_stats = defaultdict(lambda: {"correct": 0, "total": 0})

        for r in self.all_results:
            method = r.get("method", "UNKNOWN")
            method_stats[method]["total"] += 1
            method_stats[method]["correct"] += int(r.get("correct", 0))

        headers = ["method", "total", "correct", "accuracy"]
        rows = []
        for method, stats in sorted(method_stats.items()):
            acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            rows.append([method, stats["total"], stats["correct"], f"{acc:.2%}"])

        return headers, rows

    def compute_accuracy_by_question_order(self) -> Tuple[List[str], List[List[Any]]]:
        """Analyze accuracy by question order (ToM complexity)."""
        order_stats = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))

        for r in self.all_results:
            order = r.get("question_order", -1)
            method = r.get("method", "UNKNOWN")
            order_stats[order][method]["total"] += 1
            order_stats[order][method]["correct"] += int(r.get("correct", 0))

        headers = ["question_order", "method", "total", "correct", "accuracy"]
        rows = []
        for order in sorted(order_stats.keys()):
            for method in sorted(order_stats[order].keys()):
                stats = order_stats[order][method]
                acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
                rows.append([order, method, stats["total"], stats["correct"], f"{acc:.2%}"])

        return headers, rows

    def compute_accuracy_by_deception(self) -> Tuple[List[str], List[List[Any]]]:
        """Analyze accuracy with and without deception."""
        deception_stats = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))

        for r in self.all_results:
            deception = r.get("deception", False)
            deception_label = "with_deception" if deception else "no_deception"
            method = r.get("method", "UNKNOWN")
            deception_stats[deception_label][method]["total"] += 1
            deception_stats[deception_label][method]["correct"] += int(r.get("correct", 0))

        headers = ["deception", "method", "total", "correct", "accuracy"]
        rows = []
        for deception_label in ["no_deception", "with_deception"]:
            for method in sorted(deception_stats[deception_label].keys()):
                stats = deception_stats[deception_label][method]
                acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
                rows.append([deception_label, method, stats["total"], stats["correct"], f"{acc:.2%}"])

        return headers, rows

    # ==================== ADVANCED INSIGHTS ====================

    def find_impossible_questions(self) -> List[Dict]:
        """
        Find questions that NO model/method combination answered correctly.
        These represent the "hardest" questions that could benefit from a new approach.
        """
        sample_results: Dict[str, List[int]] = defaultdict(list)

        for r in self.all_results:
            sid = str(r.get("sample_id", ""))
            if sid:
                sample_results[sid].append(int(r.get("correct", 0)))

        impossible = []
        for sid, results in sample_results.items():
            if sum(results) == 0:  # No correct answers at all
                info = self.sample_registry.get(sid, {})
                impossible.append({
                    "sample_id": sid,
                    "story_id": info.get("story_id"),
                    "question": info.get("question"),
                    "answer": info.get("answer"),
                    "question_order": info.get("question_order"),
                    "deception": info.get("deception"),
                    "story_length": info.get("story_length"),
                    "attempts": len(results),
                })

        return impossible

    def find_easy_questions(self) -> List[Dict]:
        """
        Find questions that ALL model/method combinations answered correctly.
        These represent "solved" problems.
        """
        sample_results: Dict[str, List[int]] = defaultdict(list)

        for r in self.all_results:
            sid = str(r.get("sample_id", ""))
            if sid:
                sample_results[sid].append(int(r.get("correct", 0)))

        # Find max attempts per sample
        max_attempts = max(len(v) for v in sample_results.values()) if sample_results else 0

        easy = []
        for sid, results in sample_results.items():
            if sum(results) == len(results) and len(results) == max_attempts:  # All correct
                info = self.sample_registry.get(sid, {})
                easy.append({
                    "sample_id": sid,
                    "story_id": info.get("story_id"),
                    "question": info.get("question"),
                    "answer": info.get("answer"),
                    "question_order": info.get("question_order"),
                    "deception": info.get("deception"),
                    "story_length": info.get("story_length"),
                })

        return easy

    def find_partial_success_questions(self) -> List[Dict]:
        """
        Find questions where some methods succeeded but others failed.
        These indicate method complementarity opportunities.
        """
        sample_results: Dict[str, Dict[str, int]] = defaultdict(dict)

        for r in self.all_results:
            sid = str(r.get("sample_id", ""))
            method = r.get("method", "UNKNOWN")
            if sid:
                sample_results[sid][method] = int(r.get("correct", 0))

        partial = []
        for sid, method_results in sample_results.items():
            correct_methods = [m for m, c in method_results.items() if c == 1]
            incorrect_methods = [m for m, c in method_results.items() if c == 0]

            if correct_methods and incorrect_methods:  # Some succeeded, some failed
                info = self.sample_registry.get(sid, {})
                partial.append({
                    "sample_id": sid,
                    "story_id": info.get("story_id"),
                    "question": info.get("question"),
                    "answer": info.get("answer"),
                    "question_order": info.get("question_order"),
                    "deception": info.get("deception"),
                    "story_length": info.get("story_length"),
                    "correct_methods": ", ".join(correct_methods),
                    "incorrect_methods": ", ".join(incorrect_methods),
                    "num_correct": len(correct_methods),
                    "num_incorrect": len(incorrect_methods),
                })

        return partial

    def compute_method_complementarity(self) -> Tuple[List[str], List[List[Any]]]:
        """
        Analyze how methods complement each other.
        For each pair of methods, find how many questions one got right and the other wrong.
        """
        methods = set(r.get("method", "UNKNOWN") for r in self.all_results)
        methods = sorted(methods)

        sample_by_method: Dict[str, Dict[str, int]] = defaultdict(dict)

        for r in self.all_results:
            sid = str(r.get("sample_id", ""))
            method = r.get("method", "UNKNOWN")
            sample_by_method[sid][method] = int(r.get("correct", 0))

        # Build complementarity matrix
        headers = ["method"] + [f"wins_vs_{m}" for m in methods if m != methods[0]]
        rows = []
        for method_a in methods:
            a_wins_b = defaultdict(int)
            for sid, results in sample_by_method.items():
                if method_a in results:
                    a_correct = results[method_a] == 1
                    for method_b, b_correct in results.items():
                        if method_b != method_a and b_correct == 0 and a_correct:
                            a_wins_b[method_b] += 1

            row = [method_a]
            for method_b in methods:
                if method_b != method_a:
                    row.append(a_wins_b.get(method_b, 0))
            rows.append(row)

        return headers, rows

    def analyze_error_patterns(self) -> Dict[str, Any]:
        """Analyze common error patterns across all methods."""
        errors_by_order = defaultdict(lambda: {"total": 0, "errors": 0})
        errors_by_deception = defaultdict(lambda: {"total": 0, "errors": 0})
        errors_by_story_length = defaultdict(lambda: {"total": 0, "errors": 0})

        for r in self.all_results:
            correct = int(r.get("correct", 0))
            order = r.get("question_order", -1)
            deception = r.get("deception", False)
            story_length = r.get("story_length", -1)

            # By question order
            errors_by_order[order]["total"] += 1
            errors_by_order[order]["errors"] += 1 - correct

            # By deception
            key = "with_deception" if deception else "no_deception"
            errors_by_deception[key]["total"] += 1
            errors_by_deception[key]["errors"] += 1 - correct

            # By story length
            errors_by_story_length[story_length]["total"] += 1
            errors_by_story_length[story_length]["errors"] += 1 - correct

        return {
            "by_question_order": dict(errors_by_order),
            "by_deception": dict(errors_by_deception),
            "by_story_length": dict(errors_by_story_length),
        }

    def compute_model_agreement(self) -> Dict[str, float]:
        """Compute how often models agree on predictions."""
        sample_predictions: Dict[str, List[str]] = defaultdict(list)

        for r in self.all_results:
            sid = str(r.get("sample_id", ""))
            pred = r.get("pred_final", "")
            sample_predictions[sid].append(pred)

        total_samples = 0
        unanimous_correct = 0
        unanimous_incorrect = 0
        disagreement = 0

        for sid, predictions in sample_predictions.items():
            if len(predictions) < 2:
                continue

            total_samples += 1
            unique_preds = set(predictions)

            if len(unique_preds) == 1:
                # All agree
                gold = self.sample_registry.get(sid, {}).get("answer", "")
                if unique_preds.pop() == gold:
                    unanimous_correct += 1
                else:
                    unanimous_incorrect += 1
            else:
                disagreement += 1

        if total_samples == 0:
            return {}

        return {
            "total_samples_with_multiple_preds": total_samples,
            "unanimous_correct": unanimous_correct,
            "unanimous_incorrect": unanimous_incorrect,
            "disagreement": disagreement,
            "unanimous_correct_pct": unanimous_correct / total_samples * 100,
            "unanimous_incorrect_pct": unanimous_incorrect / total_samples * 100,
            "disagreement_pct": disagreement / total_samples * 100,
        }

    def find_best_method_per_sample(self) -> Tuple[List[str], List[List[Any]]]:
        """
        For each sample, determine which method performed best.
        If multiple methods got it right, note all of them.
        """
        sample_performance: Dict[str, Dict[str, int]] = defaultdict(dict)

        for r in self.all_results:
            sid = str(r.get("sample_id", ""))
            method = r.get("method", "UNKNOWN")
            sample_performance[sid][method] = int(r.get("correct", 0))

        headers = ["sample_id", "question_order", "deception", "story_length", "best_methods", "num_correct_methods", "total_methods"]
        rows = []
        for sid, perf in sorted(sample_performance.items()):
            correct_methods = [m for m, c in perf.items() if c == 1]
            best_methods = ", ".join(correct_methods) if correct_methods else "NONE"

            info = self.sample_registry.get(sid, {})
            rows.append([
                sid,
                info.get("question_order", ""),
                info.get("deception", ""),
                info.get("story_length", ""),
                best_methods,
                len(correct_methods),
                len(perf),
            ])

        return headers, rows

    def generate_insights_report(self) -> str:
        """Generate a comprehensive text report with actionable insights."""
        formatter = TableFormatter()
        lines = []
        lines.append("=" * 80)
        lines.append("THEORY OF MIND EXPERIMENT RESULTS - COMPREHENSIVE ANALYSIS")
        lines.append("=" * 80)
        lines.append("")

        # 1. Basic Statistics
        lines.append("-" * 80)
        lines.append("1. BASIC STATISTICS BY EXPERIMENT")
        lines.append("-" * 80)
        headers, rows = self.compute_basic_stats()
        lines.append(formatter.format_table(headers, rows))
        lines.append("")

        # 2. Method Comparison
        lines.append("-" * 80)
        lines.append("2. METHOD COMPARISON (ALL MODELS AGGREGATED)")
        lines.append("-" * 80)
        headers, rows = self.compute_accuracy_by_method()
        lines.append(formatter.format_table(headers, rows))
        lines.append("")

        # 3. Question Order Analysis
        lines.append("-" * 80)
        lines.append("3. ACCURACY BY QUESTION ORDER (ToM COMPLEXITY)")
        lines.append("-" * 80)
        headers, rows = self.compute_accuracy_by_question_order()
        lines.append(formatter.format_table(headers, rows))
        lines.append("")

        # 4. Deception Analysis
        lines.append("-" * 80)
        lines.append("4. ACCURACY: DECEPTION VS NON-DECEPTION")
        lines.append("-" * 80)
        headers, rows = self.compute_accuracy_by_deception()
        lines.append(formatter.format_table(headers, rows))
        lines.append("")

        # 5. Impossible Questions
        lines.append("-" * 80)
        lines.append("5. 'IMPOSSIBLE' QUESTIONS (No Method Succeeded)")
        lines.append("-" * 80)
        impossible = self.find_impossible_questions()
        lines.append(f"Total impossible questions: {len(impossible)}")
        if impossible:
            lines.append("")
            lines.append("Breakdown by question order:")
            order_dist = Counter(q["question_order"] for q in impossible)
            for order in sorted(order_dist.keys()):
                lines.append(f"  Order {order}: {order_dist[order]} questions")

            lines.append("")
            lines.append("Breakdown by deception:")
            deception_dist = Counter(q["deception"] for q in impossible)
            for deception, count in deception_dist.items():
                label = "With deception" if deception else "No deception"
                lines.append(f"  {label}: {count} questions")

            lines.append("")
            lines.append("Sample impossible questions:")
            for q in impossible[:5]:
                lines.append(f"  - Sample {q['sample_id']}: {q['question'][:80]}...")
                lines.append(f"    Answer: {q['answer']} | Order: {q['question_order']} | Deception: {q['deception']}")
        lines.append("")

        # 6. Easy Questions
        lines.append("-" * 80)
        lines.append("6. 'EASY' QUESTIONS (All Methods Succeeded)")
        lines.append("-" * 80)
        easy = self.find_easy_questions()
        lines.append(f"Total easy questions: {len(easy)}")
        if easy:
            lines.append("")
            lines.append("Breakdown by question order:")
            order_dist = Counter(q["question_order"] for q in easy)
            for order in sorted(order_dist.keys()):
                lines.append(f"  Order {order}: {order_dist[order]} questions")
        lines.append("")

        # 7. Method Complementarity
        lines.append("-" * 80)
        lines.append("7. METHOD COMPLEMENTARITY (When does each method win?)")
        lines.append("-" * 80)
        lines.append("This shows how many questions each method got RIGHT where another got WRONG.")
        lines.append("")
        headers, rows = self.compute_method_complementarity()
        lines.append(formatter.format_table(headers, rows))
        lines.append("")

        # 8. Partial Success
        lines.append("-" * 80)
        lines.append("8. PARTIAL SUCCESS QUESTIONS (Some Methods Right, Others Wrong)")
        lines.append("-" * 80)
        partial = self.find_partial_success_questions()
        lines.append(f"Total questions with partial success: {len(partial)}")
        if partial:
            # Show distribution of success patterns
            pattern_counts = Counter((q["num_correct"], q["num_incorrect"]) for q in partial)
            lines.append("")
            lines.append("Success pattern distribution:")
            for (correct, incorrect), count in sorted(pattern_counts.items()):
                lines.append(f"  {correct} correct, {incorrect} incorrect methods: {count} questions")

            lines.append("")
            lines.append("Sample partial success questions:")
            for q in partial[:5]:
                lines.append(f"  - Sample {q['sample_id']}: Order {q['question_order']}")
                lines.append(f"    Correct: {q['correct_methods']}")
                lines.append(f"    Incorrect: {q['incorrect_methods']}")
        lines.append("")

        # 9. Error Patterns
        lines.append("-" * 80)
        lines.append("9. ERROR PATTERN ANALYSIS")
        lines.append("-" * 80)
        errors = self.analyze_error_patterns()

        lines.append("Errors by Question Order (higher order = more complex ToM):")
        for order, stats in sorted(errors["by_question_order"].items()):
            rate = stats["errors"] / stats["total"] * 100 if stats["total"] > 0 else 0
            lines.append(f"  Order {order}: {stats['errors']}/{stats['total']} errors ({rate:.1f}%)")

        lines.append("")
        lines.append("Errors by Deception:")
        for key, stats in sorted(errors["by_deception"].items()):
            rate = stats["errors"] / stats["total"] * 100 if stats["total"] > 0 else 0
            label = "With deception" if key == "with_deception" else "No deception"
            lines.append(f"  {label}: {stats['errors']}/{stats['total']} errors ({rate:.1f}%)")

        lines.append("")
        lines.append("Errors by Story Length:")
        for length, stats in sorted(errors["by_story_length"].items()):
            rate = stats["errors"] / stats["total"] * 100 if stats["total"] > 0 else 0
            lines.append(f"  Length {length}: {stats['errors']}/{stats['total']} errors ({rate:.1f}%)")
        lines.append("")

        # 10. Model Agreement
        lines.append("-" * 80)
        lines.append("10. MODEL AGREEMENT ANALYSIS")
        lines.append("-" * 80)
        agreement = self.compute_model_agreement()
        if agreement:
            lines.append(f"Samples evaluated by multiple methods: {agreement['total_samples_with_multiple_preds']}")
            lines.append(f"  All correct: {agreement['unanimous_correct']} ({agreement['unanimous_correct_pct']:.1f}%)")
            lines.append(f"  All wrong: {agreement['unanimous_incorrect']} ({agreement['unanimous_incorrect_pct']:.1f}%)")
            lines.append(f"  Mixed/Disagreement: {agreement['disagreement']} ({agreement['disagreement_pct']:.1f}%)")
        lines.append("")

        # 11. Recommendations for New Method
        lines.append("=" * 80)
        lines.append("11. RECOMMENDATIONS FOR NEW METHOD DESIGN")
        lines.append("=" * 80)
        lines.append("")

        if impossible:
            lines.append(f"* TARGET THE IMPOSSIBLE: {len(impossible)} questions stumped ALL methods.")
            lines.append("  Focus new method on addressing these particularly challenging cases.")
            lines.append("")

        if partial:
            lines.append(f"* LEVERAGE COMPLEMENTARITY: {len(partial)} questions had mixed results.")
            lines.append("  A hybrid approach combining methods might succeed where individual methods fail.")
            lines.append("")

        # Analyze which question orders need most help
        order_errors = errors["by_question_order"]
        if order_errors:
            max_order = max(order_errors.keys())
            lines.append(f"* COMPLEX ToM CHALLENGE: Question order {max_order} has highest error rate.")
            lines.append(f"  Error rate at order {max_order}: {order_errors[max_order]['errors']}/{order_errors[max_order]['total']}")
            lines.append("")

        # Deception insight
        if errors["by_deception"]:
            deception_rate = errors["by_deception"]["with_deception"]["errors"] / errors["by_deception"]["with_deception"]["total"] * 100
            no_deception_rate = errors["by_deception"]["no_deception"]["errors"] / errors["by_deception"]["no_deception"]["total"] * 100
            if deception_rate > no_deception_rate:
                lines.append(f"* DECEPTION HANDLING: Error rate {deception_rate:.1f}% with deception vs {no_deception_rate:.1f}% without.")
                lines.append("  New method should improve handling of deceptive scenarios.")
                lines.append("")

        lines.append("* POTENTIAL STRATEGIES:")
        lines.append("  - Ensemble approach: Combine multiple methods' outputs")
        lines.append("  - Cascading: Start with simple method, escalate to complex if uncertain")
        lines.append("  - Meta-reasoning: Train model to select appropriate method per-question")
        lines.append("  - Deception detection: Explicitly identify and handle false statements")
        lines.append("  - Iterative verification: Check consistency of predictions")
        lines.append("")

        lines.append("=" * 80)
        lines.append("END OF ANALYSIS")
        lines.append("=" * 80)

        return "\n".join(lines)


def export_to_csv(filename: str, headers: List[str], rows: List[List[Any]]):
    """Export data to CSV file."""
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def export_dicts_to_csv(filename: str, dicts: List[Dict]):
    """Export list of dicts to CSV file."""
    if not dicts:
        return
    headers = list(dicts[0].keys())
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(dicts)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Theory-of-Mind experiment results"
    )
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default="experiment_results",
        help="Directory containing experiment result JSONL files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="analysis_report.txt",
        help="Output file for the analysis report",
    )
    parser.add_argument(
        "--export_csv",
        type=str,
        default=None,
        help="Base name for CSV exports (optional, will create multiple files)",
    )
    args = parser.parse_args()

    analyzer = ExperimentAnalyzer(args.experiment_dir)
    analyzer.load_all()

    # Generate comprehensive report
    report = analyzer.generate_insights_report()

    # Write report
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Analysis report written to: {args.output}")

    # Optionally export detailed CSVs
    if args.export_csv:
        # Export best method per sample
        headers, rows = analyzer.find_best_method_per_sample()
        export_to_csv(f"{args.export_csv}_samples.csv", headers, rows)
        print(f"Sample analysis exported to: {args.export_csv}_samples.csv")

        # Export impossible questions
        impossible = analyzer.find_impossible_questions()
        if impossible:
            export_dicts_to_csv(f"{args.export_csv}_impossible.csv", impossible)
            print(f"Impossible questions exported to: {args.export_csv}_impossible.csv")

        # Export partial success
        partial = analyzer.find_partial_success_questions()
        if partial:
            export_dicts_to_csv(f"{args.export_csv}_partial.csv", partial)
            print(f"Partial success questions exported to: {args.export_csv}_partial.csv")

        # Export easy questions
        easy = analyzer.find_easy_questions()
        if easy:
            export_dicts_to_csv(f"{args.export_csv}_easy.csv", easy)
            print(f"Easy questions exported to: {args.export_csv}_easy.csv")


if __name__ == "__main__":
    main()
