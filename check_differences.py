import json
import re

def find_discrepancies(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Dictionary to store: {(normalized_story, question_order): [list_of_samples]}
    story_map = {}
    
    # Prefix to ignore in VP prompting types
    vp_prefix = "Read the following story and answer the multiple-choice question. Please provide answer without explanations.\n"

    for entry in data:
        # 1. Normalize the story
        raw_story = entry.get("story", "")
        normalized_story = raw_story.replace(vp_prefix, "").strip()
        
        q_order = entry.get("question_order")
        key = (normalized_story, q_order)
        
        if key not in story_map:
            story_map[key] = []
        story_map[key].append(entry)

    # 2. Identify cases where answers differ for the same story + order
    discrepancies = []
    for (story_text, q_order), samples in story_map.items():
        # Get unique answers within this group
        unique_answers = set(s["answer"] for s in samples)
        
        if len(unique_answers) > 1:
            discrepancies.append({
                "question_order": q_order,
                "answers_found": list(unique_answers),
                "sample_ids": [s["sample_id"] for s in samples],
                "story_snippet": story_text[:100] + "..."
            })

    return discrepancies

# Example Usage:
results = find_discrepancies('data/hitom.json')
for d in results:
   print(f"Conflict found in IDs {d['sample_ids']}: Answers vary {d['answers_found']}")