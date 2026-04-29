import json
import random
from collections import defaultdict
from typing import List, Dict, Any

def create_balanced_subset(input_path: str, output_path: str) -> None:
    """
    Creates a subset of 250 "VP" and 250 "COTP" random subsamples. Each of them having 50 samples of each question_order (0-4).
    """
    # 1. Load the original dataset
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input file must be a JSON list.")

    # 2. Categorize the data by prompting_type and question_order
    # Structure: categorized_data[prompting_type][question_order] = [sample1, sample2, ...]
    categorized_data = defaultdict(lambda: defaultdict(list))

    for sample in data:
        p_type = sample.get("prompting_type")
        q_order = sample.get("question_order")
        
        # Only categorize the types and orders we explicitly care about
        if p_type in ["CoTP", "VP"] and q_order in [0, 1, 2, 3, 4]:
            categorized_data[p_type][q_order].append(sample)

    # 3. Randomly sample the required amounts
    subset: List[Dict[str, Any]] = []
    samples_per_category = 50  # 250 total per prompting_type / 5 question_orders

    for p_type in ["CoTP", "VP"]:
        for q_order in [0, 1, 2, 3, 4]:
            available_samples = categorized_data[p_type][q_order]
            
            # Safety check: ensure we actually have enough samples to draw from
            if len(available_samples) < samples_per_category:
                print(f"Warning: Only found {len(available_samples)} samples for {p_type} / order {q_order}. Expected {samples_per_category}.")
                # Take all available if we fall short
                sampled = available_samples
            else:
                # Randomly select 50 samples without replacement
                sampled = random.sample(available_samples, samples_per_category)
            
            subset.extend(sampled)

    # 4. Shuffle the final subset so the dataset isn't ordered by type/order
    random.shuffle(subset)

    # 5. Save back to a JSON list with formatting
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(subset, f, indent=4, ensure_ascii=False)

    # Output stats
    print(f"Successfully created subset with {len(subset)} samples.")
    print(f"Saved to: {output_path}")
    
    # Quick verification print
    verify_counts = defaultdict(int)
    for s in subset:
        verify_counts[f"{s['prompting_type']}_order_{s['question_order']}"] += 1
    
    print("\nSubset Breakdown:")
    for key, count in sorted(verify_counts.items()):
        print(f"  {key}: {count} samples")

if __name__ == "__main__":
    # Define your paths here
    INPUT_FILE = "data/hitom.json"
    OUTPUT_FILE = "data/hitom_subset.json"
    
    create_balanced_subset(INPUT_FILE, OUTPUT_FILE)
