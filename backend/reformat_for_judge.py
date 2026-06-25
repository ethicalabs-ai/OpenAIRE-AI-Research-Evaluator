import argparse
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Reformat chat JSONL datasets into flat JSONL for judge_cli.py"
    )
    parser.add_argument(
        "input_path", type=str, help="Path to input train.jsonl or val.jsonl"
    )
    parser.add_argument("output_path", type=str, help="Path to output flat JSONL")
    args = parser.parse_args()

    in_path = Path(args.input_path)
    out_path = Path(args.output_path)

    if not in_path.exists():
        print(f"Error: {in_path} does not exist.")
        return

    records_written = 0
    with (
        open(in_path, "r", encoding="utf-8") as f_in,
        open(out_path, "w", encoding="utf-8") as f_out,
    ):
        for i, line in enumerate(f_in):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"Line {i} parsing error: {e}")
                continue

            # Extract user message
            user_content = ""
            for msg in obj.get("messages", []):
                if msg.get("role") == "user":
                    user_content = msg.get("content", "")
                    break

            # Extract title and abstract using regex
            title_match = re.search(
                r"Title:\s*(.*?)\nAbstract:", user_content, re.DOTALL
            )
            abstract_match = re.search(r"Abstract:\s*(.*)$", user_content, re.DOTALL)

            if title_match and abstract_match:
                title = title_match.group(1).strip()
                abstract = abstract_match.group(1).strip()
            else:
                title = (obj.get("title") or "").strip()
                abstract = (obj.get("abstract") or "").strip()

            label = (obj.get("label") or "").strip()
            import hashlib

            title_hash = hashlib.sha256(title.encode("utf-8")).hexdigest()[:16]
            doi = (obj.get("doi") or "").strip()
            if not doi:
                prefix = "val" if "val" in in_path.name else "train"
                doi = f"10.5555/{prefix}-{title_hash}"

            if not title or not abstract or not label:
                # If we couldn't match, let's look for fallback labels
                continue

            f_out.write(
                json.dumps(
                    {"title": title, "abstract": abstract, "label": label, "doi": doi}
                )
                + "\n"
            )
            records_written += 1

    print(f"Successfully wrote {records_written} flat records to {out_path}")


if __name__ == "__main__":
    main()
