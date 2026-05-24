"""
Correlation Analysis Script for Theory-of-Mind Experiment Results

This script analyzes correlations between:
1. Methods (which methods succeed/fail together)
2. Entry fields (question_order, story_length, deception, etc.)
3. Method x Feature interactions (how methods perform across different feature values)
4. Cross-feature correlations (e.g., question_order vs story_length effects on accuracy)
"""

import json
import argparse
import math
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
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

    if "cotp" in filename.lower():
        info["category"] = "CoTP"
    elif "vr" in filename.lower():
        info["category"] = "VR"

    method_keywords = ["vp", "soo", "simtom", "perceptom", "dtom"]
    for i, part in enumerate(parts):
        part_lower = part.lower()
        for method in method_keywords:
            if method in part_lower:
                info["method"] = method.upper()
                break

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

        str_rows = [[str(cell) for cell in row] for row in rows]
        all_data = [headers] + str_rows
        col_widths = [max(len(row[i]) for row in all_data) for i in range(len(headers))]

        lines = []
        header_cells = [headers[i].ljust(col_widths[i]) for i in range(len(headers))]
        lines.append(" | ".join(header_cells))
        lines.append("-" * len(lines[0]))

        for row in str_rows:
            cells = [row[i].ljust(col_widths[i]) for i in range(len(headers))]
            lines.append(" | ".join(cells))

        return "\n".join(lines)


def compute_pearson_correlation(x: List[float], y: List[float]) -> Tuple[float, float]:
    """
    Compute Pearson correlation coefficient between two lists.
    Returns (correlation, p-value_approx).
    """
    n = len(x)
    if n != len(y) or n < 2:
        return 0.0, 1.0

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    var_x = sum((xi - mean_x) ** 2 for xi in x)
    var_y = sum((yi - mean_y) ** 2 for yi in y)

    if var_x == 0 or var_y == 0:
        return 0.0, 1.0

    covariance = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    corr = covariance / (math.sqrt(var_x) * math.sqrt(var_y))

    # Approximate p-value (simplified)
    if abs(corr) >= 1.0:
        p_value = 0.0
    else:
        t_stat = corr * math.sqrt((n - 2) / (1 - corr ** 2))
        # Simplified approximation: smaller p for larger |t|
        p_value = max(0.001, min(1.0, 2 / (abs(t_stat) + 2)))

    return corr, p_value


def compute_cramers_v(x: List[Any], y: List[Any]) -> float:
    """
    Compute Cramer's V for association between two categorical variables.
    """
    # Build contingency table
    x_vals = list(set(x))
    y_vals = list(set(y))

    if len(x_vals) < 2 or len(y_vals) < 2:
        return 0.0

    contingency = defaultdict(lambda: defaultdict(int))
    for xi, yi in zip(x, y):
        contingency[xi][yi] += 1

    # Compute chi-squared
    n = len(x)
    if n == 0:
        return 0.0

    row_totals = {xv: sum(contingency[xv].values()) for xv in x_vals}
    col_totals = {yv: sum(contingency[xv][yv] for xv in x_vals) for yv in y_vals}

    chi2 = 0.0
    for xv in x_vals:
        for yv in y_vals:
            observed = contingency[xv][yv]
            expected = (row_totals[xv] * col_totals[yv]) / n
            if expected > 0:
                chi2 += (observed - expected) ** 2 / expected

    # Cramer's V
    min_dim = min(len(x_vals) - 1, len(y_vals) - 1)
    if min_dim == 0:
        return 0.0

    cramers_v = math.sqrt(chi2 / (n * min_dim))
    return cramers_v


def cohens_kappa(rater1: List[int], rater2: List[int]) -> float:
    """
    Compute Cohen's Kappa for inter-rater agreement.
    Values: 1 = perfect agreement, 0 = chance agreement, negative = worse than chance
    """
    n = len(rater1)
    if n != len(rater2) or n == 0:
        return 0.0

    # Observed agreement
    agreement = sum(1 for a, b in zip(rater1, rater2) if a == b) / n

    # Expected agreement by chance
    p1_1 = sum(rater1) / n  # P(rater1 = 1)
    p1_0 = 1 - p1_1
    p2_1 = sum(rater2) / n  # P(rater2 = 1)
    p2_0 = 1 - p2_1

    expected = p1_1 * p2_1 + p1_0 * p2_0

    if expected >= 1:
        return 0.0

    kappa = (agreement - expected) / (1 - expected)
    return kappa


class CorrelationAnalyzer:
    """Analyzer for correlations in ToM experiment results."""

    def __init__(self, experiment_dir: str):
        self.experiment_dir = Path(experiment_dir)
        self.all_results: List[Dict[str, Any]] = []
        self.experiments: Dict[str, List[Dict]] = {}
        self.sample_registry: Dict[str, Dict] = {}
        self.methods: Set[str] = set()
        self.models: Set[str] = set()

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

                self.methods.add(r.get("method", "UNKNOWN"))
                self.models.add(meta["model"])

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
        print(f"Methods: {sorted(self.methods)}")
        print(f"Models: {sorted(self.models)}")

    # ==================== METHOD-METHOD CORRELATIONS ====================

    def compute_method_correlation_matrix(self) -> Tuple[List[str], List[List[Any]]]:
        """
        Compute correlation between methods' success/failure patterns.
        Uses Cohen's Kappa to measure agreement.
        """
        methods = sorted(self.methods)

        # Build sample-level correctness matrix
        sample_correctness: Dict[str, Dict[str, int]] = defaultdict(dict)
        for r in self.all_results:
            sid = str(r.get("sample_id", ""))
            method = r.get("method", "UNKNOWN")
            sample_correctness[sid][method] = int(r.get("correct", 0))

        # Keep only samples evaluated by all methods
        complete_samples = {
            sid: results for sid, results in sample_correctness.items()
            if len(results) == len(methods)
        }

        if not complete_samples:
            return ["Error"], [["No samples evaluated by all methods"]]

        # Compute kappa between each pair
        headers = ["Method Pair", "Cohen's Kappa", "Agreement", "Interpretation"]
        rows = []

        for i, m1 in enumerate(methods):
            for m2 in methods[i+1:]:
                rater1 = [complete_samples[sid][m1] for sid in complete_samples]
                rater2 = [complete_samples[sid][m2] for sid in complete_samples]

                kappa = cohens_kappa(rater1, rater2)
                agreement = sum(1 for a, b in zip(rater1, rater2) if a == b) / len(rater1)

                # Interpret kappa
                if kappa < 0:
                    interp = "Poor (disagreement)"
                elif kappa < 0.2:
                    interp = "Slight agreement"
                elif kappa < 0.4:
                    interp = "Fair agreement"
                elif kappa < 0.6:
                    interp = "Moderate agreement"
                elif kappa < 0.8:
                    interp = "Substantial agreement"
                else:
                    interp = "Almost perfect"

                rows.append([f"{m1} vs {m2}", f"{kappa:.3f}", f"{agreement:.1%}", interp])

        return headers, rows

    def find_method_clusters(self) -> Dict[str, List[str]]:
        """
        Group methods into clusters based on their correlation patterns.
        High agreement = same cluster.
        """
        methods = sorted(self.methods)

        sample_correctness: Dict[str, Dict[str, int]] = defaultdict(dict)
        for r in self.all_results:
            sid = str(r.get("sample_id", ""))
            method = r.get("method", "UNKNOWN")
            sample_correctness[sid][method] = int(r.get("correct", 0))

        complete_samples = {
            sid: results for sid, results in sample_correctness.items()
            if len(results) == len(methods)
        }

        if not complete_samples:
            return {}

        # Compute average agreement for each method
        method_agreement = defaultdict(lambda: defaultdict(float))
        for i, m1 in enumerate(methods):
            for m2 in methods[i+1:]:
                rater1 = [complete_samples[sid][m1] for sid in complete_samples]
                rater2 = [complete_samples[sid][m2] for sid in complete_samples]
                kappa = cohens_kappa(rater1, rater2)
                method_agreement[m1][m2] = kappa
                method_agreement[m2][m1] = kappa

        # Simple clustering: methods with kappa > 0.5 are in same cluster
        clusters = []
        assigned = set()

        for method in methods:
            if method in assigned:
                continue

            cluster = [method]
            assigned.add(method)

            for other in methods:
                if other not in assigned and method_agreement[method][other] > 0.5:
                    cluster.append(other)
                    assigned.add(other)

            clusters.append(cluster)

        return {f"Cluster_{i+1}": cluster for i, cluster in enumerate(clusters)}

    # ==================== FEATURE-FEATURE CORRELATIONS ====================

    def compute_feature_correlations(self) -> Tuple[List[str], List[List[Any]]]:
        """
        Analyze correlations between entry fields and accuracy.
        """
        features = ["question_order", "story_length", "deception"]

        # Collect data
        data = {
            "question_order": [],
            "story_length": [],
            "deception": [],
            "correct": []
        }

        for r in self.all_results:
            data["question_order"].append(r.get("question_order", -1))
            data["story_length"].append(r.get("story_length", -1))
            data["deception"].append(1 if r.get("deception", False) else 0)
            data["correct"].append(int(r.get("correct", 0)))

        headers = ["Feature A", "Feature B", "Correlation", "Strength", "Relationship"]
        rows = []

        # Compute correlations
        correlations = [
            ("question_order", "story_length"),
            ("question_order", "deception"),
            ("story_length", "deception"),
            ("question_order", "correct"),
            ("story_length", "correct"),
            ("deception", "correct"),
        ]

        for feat_a, feat_b in correlations:
            if feat_a == "deception" or feat_b == "deception":
                # Use Cramer's V for categorical
                corr = compute_cramers_v(data[feat_a], data[feat_b])
                corr_type = "Cramer's V"
            else:
                # Use Pearson for continuous
                corr, _ = compute_pearson_correlation(data[feat_a], data[feat_b])
                corr_type = "Pearson r"

            # Interpret strength
            abs_corr = abs(corr)
            if abs_corr < 0.1:
                strength = "Negligible"
            elif abs_corr < 0.3:
                strength = "Small"
            elif abs_corr < 0.5:
                strength = "Medium"
            elif abs_corr < 0.7:
                strength = "Large"
            else:
                strength = "Very Large"

            # Direction
            if corr > 0:
                direction = "Positive"
            elif corr < 0:
                direction = "Negative"
            else:
                direction = "None"

            rows.append([
                feat_a,
                feat_b,
                f"{corr:.3f} ({corr_type})",
                strength,
                direction
            ])

        return headers, rows

    def analyze_feature_interactions(self) -> Tuple[List[str], List[List[Any]]]:
        """
        Analyze how combinations of features affect accuracy.
        """
        # Group by feature combinations
        combo_stats = defaultdict(lambda: {"correct": 0, "total": 0})

        for r in self.all_results:
            order = r.get("question_order", -1)
            deception = "deception" if r.get("deception", False) else "no_deception"
            length_bucket = self._bucket_story_length(r.get("story_length", -1))

            key = (order, deception, length_bucket)
            combo_stats[key]["total"] += 1
            combo_stats[key]["correct"] += int(r.get("correct", 0))

        headers = ["question_order", "deception", "story_length", "accuracy", "sample_count"]
        rows = []

        for (order, deception, length), stats in sorted(combo_stats.items()):
            if stats["total"] >= 5:  # Only show combinations with enough samples
                acc = stats["correct"] / stats["total"]
                rows.append([
                    order,
                    deception,
                    length,
                    f"{acc:.2%}",
                    stats["total"]
                ])

        # Sort by accuracy
        rows.sort(key=lambda x: float(x[3].rstrip('%')) / 100, reverse=True)

        return headers, rows

    def _bucket_story_length(self, length: int) -> str:
        """Bucket story length into categories."""
        if length < 0:
            return "unknown"
        elif length <= 100:
            return "short (<=100)"
        elif length <= 200:
            return "medium (101-200)"
        elif length <= 300:
            return "long (201-300)"
        else:
            return "very_long (>300)"

    # ==================== METHOD X FEATURE CORRELATIONS ====================

    def compute_method_feature_correlations(self) -> Tuple[List[str], List[List[Any]]]:
        """
        For each method, compute correlation with each feature.
        This shows which methods are more/less sensitive to which features.
        """
        methods = sorted(self.methods)
        features = ["question_order", "story_length", "deception"]

        headers = ["Method", "Feature", "Correlation", "P-value", "Interpretation"]
        rows = []

        for method in methods:
            # Collect data for this method
            method_data = [r for r in self.all_results if r.get("method") == method]

            for feature in features:
                if feature == "deception":
                    x = [1 if r.get("deception", False) else 0 for r in method_data]
                else:
                    x = [r.get(feature, -1) for r in method_data]

                y = [int(r.get("correct", 0)) for r in method_data]

                if len(set(x)) < 2:
                    continue

                corr, p_val = compute_pearson_correlation(x, y)

                # Interpret
                if abs(corr) < 0.1:
                    interp = "No correlation"
                elif corr > 0.3:
                    interp = f"Strong positive (higher {feature} = better)"
                elif corr > 0.1:
                    interp = f"Weak positive (higher {feature} = slightly better)"
                elif corr < -0.3:
                    interp = f"Strong negative (higher {feature} = worse)"
                else:
                    interp = f"Weak negative (higher {feature} = slightly worse)"

                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""

                rows.append([
                    method,
                    feature,
                    f"{corr:.3f}",
                    f"{p_val:.3f}{sig}",
                    interp
                ])

        return headers, rows

    def compute_method_rankings_by_feature(self) -> Tuple[List[str], List[List[Any]]]:
        """
        For each feature value, rank methods by performance.
        """
        methods = sorted(self.methods)

        # By question order
        order_stats = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))
        for r in self.all_results:
            order = r.get("question_order", -1)
            method = r.get("method", "UNKNOWN")
            order_stats[order][method]["total"] += 1
            order_stats[order][method]["correct"] += int(r.get("correct", 0))

        # By deception
        deception_stats = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))
        for r in self.all_results:
            deception = "with_deception" if r.get("deception", False) else "no_deception"
            method = r.get("method", "UNKNOWN")
            deception_stats[deception][method]["total"] += 1
            deception_stats[deception][method]["correct"] += int(r.get("correct", 0))

        # By story length bucket
        length_stats = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))
        for r in self.all_results:
            length = self._bucket_story_length(r.get("story_length", -1))
            method = r.get("method", "UNKNOWN")
            length_stats[length][method]["total"] += 1
            length_stats[length][method]["correct"] += int(r.get("correct", 0))

        headers = ["Feature_Value", "Method", "Accuracy", "Rank"]
        rows = []

        # Output for question order
        for order in sorted(order_stats.keys()):
            method_accs = []
            for method in methods:
                stats = order_stats[order][method]
                if stats["total"] > 0:
                    acc = stats["correct"] / stats["total"]
                    method_accs.append((method, acc, stats["total"]))

            method_accs.sort(key=lambda x: x[1], reverse=True)
            for rank, (method, acc, total) in enumerate(method_accs, 1):
                rows.append([f"order_{order}", method, f"{acc:.2%} (n={total})", rank])

        # Output for deception
        for deception in ["no_deception", "with_deception"]:
            method_accs = []
            for method in methods:
                stats = deception_stats[deception][method]
                if stats["total"] > 0:
                    acc = stats["correct"] / stats["total"]
                    method_accs.append((method, acc, stats["total"]))

            method_accs.sort(key=lambda x: x[1], reverse=True)
            for rank, (method, acc, total) in enumerate(method_accs, 1):
                rows.append([deception, method, f"{acc:.2%} (n={total})", rank])

        return headers, rows

    # ==================== COMPLEMENTARITY ANALYSIS ====================

    def compute_conditional_success_rates(self) -> Tuple[List[str], List[List[Any]]]:
        """
        Analyze: Given that Method A succeeded, what's the probability Method B succeeds?
        This reveals conditional dependencies between methods.
        """
        methods = sorted(self.methods)

        sample_correctness: Dict[str, Dict[str, int]] = defaultdict(dict)
        for r in self.all_results:
            sid = str(r.get("sample_id", ""))
            method = r.get("method", "UNKNOWN")
            sample_correctness[sid][method] = int(r.get("correct", 0))

        complete_samples = {
            sid: results for sid, results in sample_correctness.items()
            if len(results) == len(methods)
        }

        if not complete_samples:
            return ["Error"], [["No samples with all methods"]]

        headers = ["Given Method A Succeeded"] + [f"P(B={m})" for m in methods]
        rows = []

        for method_a in methods:
            row = [method_a]

            # Find samples where method_a succeeded
            a_success_samples = {
                sid: results for sid, results in complete_samples.items()
                if results[method_a] == 1
            }

            if not a_success_samples:
                row.extend(["N/A"] * len(methods))
                rows.append(row)
                continue

            for method_b in methods:
                # P(B succeeds | A succeeded)
                b_success_given_a = sum(
                    1 for results in a_success_samples.values()
                    if results[method_b] == 1
                )
                prob = b_success_given_a / len(a_success_samples)
                row.append(f"{prob:.2%}")

            rows.append(row)

        return headers, rows

    def find_synergistic_pairs(self) -> Tuple[List[str], List[List[Any]]]:
        """
        Find pairs of methods that are synergistic:
        - High combined accuracy (union of successes)
        - Low individual agreement (not redundant)
        """
        methods = sorted(self.methods)

        sample_correctness: Dict[str, Dict[str, int]] = defaultdict(dict)
        for r in self.all_results:
            sid = str(r.get("sample_id", ""))
            method = r.get("method", "UNKNOWN")
            sample_correctness[sid][method] = int(r.get("correct", 0))

        complete_samples = {
            sid: results for sid, results in sample_correctness.items()
            if len(results) == len(methods)
        }

        if not complete_samples:
            return ["Error"], [["No samples with all methods"]]

        headers = ["Method Pair", "Individual A", "Individual B", "Combined (Union)",
                   "Overlap (Both Right)", "Complementarity Score"]
        rows = []

        for i, m1 in enumerate(methods):
            for m2 in methods[i+1:]:
                m1_right = sum(1 for r in complete_samples.values() if r[m1] == 1)
                m2_right = sum(1 for r in complete_samples.values() if r[m2] == 1)
                both_right = sum(1 for r in complete_samples.values() if r[m1] == 1 and r[m2] == 1)
                either_right = sum(1 for r in complete_samples.values() if r[m1] == 1 or r[m2] == 1)

                n = len(complete_samples)
                indiv_a = m1_right / n
                indiv_b = m2_right / n
                combined = either_right / n
                overlap = both_right / n

                # Complementarity: combined - max(individual)
                # Higher = more benefit from combining
                complementarity = combined - max(indiv_a, indiv_b)

                rows.append([
                    f"{m1} + {m2}",
                    f"{indiv_a:.2%}",
                    f"{indiv_b:.2%}",
                    f"{combined:.2%}",
                    f"{overlap:.2%}",
                    f"{complementarity:.3f}"
                ])

        # Sort by complementarity score
        rows.sort(key=lambda x: float(x[-1]), reverse=True)

        return headers, rows

    # ==================== TEMPORAL/SEQUENTIAL PATTERNS ====================

    def analyze_order_effects(self) -> Tuple[List[str], List[List[Any]]]:
        """
        Analyze if accuracy on previous questions predicts accuracy on current question.
        """
        # Group by story_id to see sequences within a story
        story_results = defaultdict(list)
        for r in self.all_results:
            story_id = r.get("story_id", "")
            if story_id:
                story_results[story_id].append(r)

        # For each method, check correlation between consecutive question accuracy
        methods = sorted(self.methods)

        headers = ["Method", "Consecutive Agreement", "Runs Test Z", "Pattern"]
        rows = []

        for method in methods:
            method_results = [r for r in self.all_results if r.get("method") == method]

            # Sort by sample_id (proxy for sequence)
            sorted_results = sorted(method_results, key=lambda x: (x.get("story_id", ""), x.get("sample_id", "")))

            # Get correctness sequence
            correctness = [int(r.get("correct", 0)) for r in sorted_results]

            if len(correctness) < 10:
                continue

            # Consecutive agreement: how often same result repeats
            consecutive_same = sum(
                1 for i in range(len(correctness) - 1)
                if correctness[i] == correctness[i + 1]
            )
            consec_agreement = consecutive_same / (len(correctness) - 1)

            # Simple runs test (simplified)
            runs = 1
            for i in range(1, len(correctness)):
                if correctness[i] != correctness[i - 1]:
                    runs += 1

            # Pattern description
            if consec_agreement > 0.6:
                pattern = "Clustered (streaky)"
            elif consec_agreement < 0.4:
                pattern = "Alternating"
            else:
                pattern = "Random"

            # Approximate z-score for runs test
            n1 = sum(correctness)
            n0 = len(correctness) - n1
            expected_runs = (2 * n1 * n0) / (n1 + n0) + 1 if (n1 + n0) > 0 else 0
            var_runs = (2 * n1 * n0 * (2 * n1 * n0 - n1 - n0)) / ((n1 + n0) ** 2 * (n1 + n0 - 1)) if (n1 + n0) > 1 else 1
            z_score = (runs - expected_runs) / math.sqrt(var_runs) if var_runs > 0 else 0

            rows.append([
                method,
                f"{consec_agreement:.2%}",
                f"{z_score:.2f}",
                pattern
            ])

        return headers, rows

    # ==================== REPORT GENERATION ====================

    def generate_correlation_report(self) -> str:
        """Generate a comprehensive correlation analysis report."""
        formatter = TableFormatter()
        lines = []

        lines.append("=" * 100)
        lines.append("THEORY OF MIND EXPERIMENT RESULTS - CORRELATION ANALYSIS")
        lines.append("=" * 100)
        lines.append("")
        lines.append("This report analyzes correlations between methods, entry fields, and their interactions.")
        lines.append("")

        # 1. Method-Method Correlations
        lines.append("-" * 100)
        lines.append("1. METHOD-METHOD AGREEMENT (Cohen's Kappa)")
        lines.append("-" * 100)
        lines.append("Measures how often methods succeed/fail together. Higher = more similar patterns.")
        lines.append("")
        headers, rows = self.compute_method_correlation_matrix()
        lines.append(formatter.format_table(headers, rows))
        lines.append("")
        lines.append("Interpretation: Negative/low kappa suggests methods are complementary.")
        lines.append("")

        # Method Clusters
        lines.append("-" * 100)
        lines.append("2. METHOD CLUSTERS (Based on Agreement Patterns)")
        lines.append("-" * 100)
        clusters = self.find_method_clusters()
        for cluster_name, methods in clusters.items():
            lines.append(f"  {cluster_name}: {', '.join(methods)}")
        lines.append("")
        lines.append("Methods in same cluster tend to succeed/fail together.")
        lines.append("")

        # 3. Feature-Feature Correlations
        lines.append("-" * 100)
        lines.append("3. FEATURE CORRELATIONS")
        lines.append("-" * 100)
        lines.append("Correlations between entry fields (question_order, story_length, deception)")
        lines.append("and with accuracy (correct).")
        lines.append("")
        headers, rows = self.compute_feature_correlations()
        lines.append(formatter.format_table(headers, rows))
        lines.append("")

        # 4. Feature Interactions
        lines.append("-" * 100)
        lines.append("4. FEATURE INTERACTIONS (Combined Effects on Accuracy)")
        lines.append("-" * 100)
        lines.append("How combinations of features affect accuracy. Only showing combinations with 5+ samples.")
        lines.append("")
        headers, rows = self.analyze_feature_interactions()
        lines.append(formatter.format_table(headers, rows[:20]))  # Top 20
        if len(rows) > 20:
            lines.append(f"... ({len(rows) - 20} more combinations)")
        lines.append("")

        # 5. Method-Feature Correlations
        lines.append("-" * 100)
        lines.append("5. METHOD X FEATURE CORRELATIONS")
        lines.append("-" * 100)
        lines.append("How each method's performance correlates with entry features.")
        lines.append("Significance: *** p<0.001, ** p<0.01, * p<0.05")
        lines.append("")
        headers, rows = self.compute_method_feature_correlations()
        lines.append(formatter.format_table(headers, rows))
        lines.append("")

        # 6. Method Rankings by Feature
        lines.append("-" * 100)
        lines.append("6. METHOD RANKINGS BY FEATURE VALUE")
        lines.append("-" * 100)
        lines.append("Performance ranking of methods for different feature values.")
        lines.append("")
        headers, rows = self.compute_method_rankings_by_feature()
        lines.append(formatter.format_table(headers, rows))
        lines.append("")

        # 7. Conditional Success Rates
        lines.append("-" * 100)
        lines.append("7. CONDITIONAL SUCCESS PROBABILITIES")
        lines.append("-" * 100)
        lines.append("P(Method B succeeds | Method A succeeded)")
        lines.append("High values indicate methods succeed together (redundant).")
        lines.append("Low values given high individual success indicate complementarity.")
        lines.append("")
        headers, rows = self.compute_conditional_success_rates()
        lines.append(formatter.format_table(headers, rows))
        lines.append("")

        # 8. Synergistic Pairs
        lines.append("-" * 100)
        lines.append("8. SYNERGISTIC METHOD PAIRS")
        lines.append("-" * 100)
        lines.append("Pairs ranked by complementarity score (combined - max(individual)).")
        lines.append("Higher scores indicate better potential for ensemble methods.")
        lines.append("")
        headers, rows = self.find_synergistic_pairs()
        lines.append(formatter.format_table(headers, rows))
        lines.append("")

        # 9. Order Effects
        lines.append("-" * 100)
        lines.append("9. SEQUENTIAL PATTERNS (Order Effects)")
        lines.append("-" * 100)
        lines.append("Analysis of whether correct/incorrect answers cluster in sequences.")
        lines.append("")
        headers, rows = self.analyze_order_effects()
        if rows:
            lines.append(formatter.format_table(headers, rows))
        else:
            lines.append("Insufficient data for order effects analysis.")
        lines.append("")

        # 10. Key Insights Summary
        lines.append("=" * 100)
        lines.append("10. KEY CORRELATION INSIGHTS")
        lines.append("=" * 100)
        lines.append("")

        # Find best/worst feature correlations
        headers, corr_rows = self.compute_feature_correlations()
        accuracy_corrs = [r for r in corr_rows if "correct" in r[1]]
        if accuracy_corrs:
            strongest = max(accuracy_corrs, key=lambda x: abs(float(x[2].split()[0])))
            lines.append(f"* STRONGEST PREDICTOR: {strongest[0]} has {strongest[3].lower()} correlation with accuracy")
            lines.append(f"  ({strongest[2]}, {strongest[4]} relationship)")
            lines.append("")

        # Find most synergistic pair
        headers, synergy_rows = self.find_synergistic_pairs()
        if synergy_rows:
            best_pair = synergy_rows[0]
            lines.append(f"* BEST ENSEMBLE PAIR: {best_pair[0]}")
            lines.append(f"  Combined accuracy: {best_pair[3]} vs individual max {max(best_pair[1], best_pair[2])}")
            lines.append(f"  Complementarity score: {best_pair[-1]}")
            lines.append("")

        # Feature interaction insight
        headers, interaction_rows = self.analyze_feature_interactions()
        if interaction_rows:
            best_combo = interaction_rows[0]
            worst_combo = interaction_rows[-1]
            lines.append(f"* EASIEST CONDITIONS: Order={best_combo[0]}, {best_combo[1]}, length={best_combo[2]} ({best_combo[3]})")
            lines.append(f"* HARDEST CONDITIONS: Order={worst_combo[0]}, {worst_combo[1]}, length={worst_combo[2]} ({worst_combo[3]})")
            lines.append("")

        lines.append("* RECOMMENDATIONS:")
        lines.append("  - Consider ensemble methods for pairs with high complementarity scores")
        lines.append("  - Methods from different clusters may provide diverse perspectives")
        lines.append("  - Target feature combinations with lowest accuracy for method improvement")
        lines.append("  - Feature correlations suggest which question characteristics are linked")
        lines.append("")

        lines.append("=" * 100)
        lines.append("END OF CORRELATION ANALYSIS")
        lines.append("=" * 100)

        return "\n".join(lines)


def export_to_csv(filename: str, headers: List[str], rows: List[List[Any]]):
    """Export data to CSV file."""
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze correlations in Theory-of-Mind experiment results"
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
        default="correlation_report.txt",
        help="Output file for the correlation analysis report",
    )
    parser.add_argument(
        "--export_csv",
        type=str,
        default=None,
        help="Base name for CSV exports (optional, will create multiple files)",
    )
    args = parser.parse_args()

    analyzer = CorrelationAnalyzer(args.experiment_dir)
    analyzer.load_all()

    # Generate correlation report
    report = analyzer.generate_correlation_report()

    # Write report
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Correlation analysis report written to: {args.output}")

    # Optionally export detailed CSVs
    if args.export_csv:
        # Export method correlations
        headers, rows = analyzer.compute_method_correlation_matrix()
        export_to_csv(f"{args.export_csv}_method_correlations.csv", headers, rows)
        print(f"Method correlations exported to: {args.export_csv}_method_correlations.csv")

        # Export feature correlations
        headers, rows = analyzer.compute_feature_correlations()
        export_to_csv(f"{args.export_csv}_feature_correlations.csv", headers, rows)
        print(f"Feature correlations exported to: {args.export_csv}_feature_correlations.csv")

        # Export method-feature correlations
        headers, rows = analyzer.compute_method_feature_correlations()
        export_to_csv(f"{args.export_csv}_method_feature_correlations.csv", headers, rows)
        print(f"Method-feature correlations exported to: {args.export_csv}_method_feature_correlations.csv")

        # Export synergistic pairs
        headers, rows = analyzer.find_synergistic_pairs()
        export_to_csv(f"{args.export_csv}_synergistic_pairs.csv", headers, rows)
        print(f"Synergistic pairs exported to: {args.export_csv}_synergistic_pairs.csv")

        # Export feature interactions
        headers, rows = analyzer.analyze_feature_interactions()
        export_to_csv(f"{args.export_csv}_feature_interactions.csv", headers, rows)
        print(f"Feature interactions exported to: {args.export_csv}_feature_interactions.csv")


if __name__ == "__main__":
    main()
