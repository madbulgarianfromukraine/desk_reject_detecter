import pandas as pd
import os
import sys
from typing import Dict, List
from google.genai import types, errors

os.chdir("/home/nazarii-yukhnovskyi/Documents/uni/5_Semester/Seminar_CT_AI/desk_reject_detecter/")
from core.config import create_engine
from core.schemas import FormattingCheck
from core.files import process_supplemental_files

CSV_DIR = "data/iclr/data"
CSV_FILES = [
    "evaluation_results_ddr.csv",
    "evaluation_results_ddr_1_iteration.csv",
    "evaluation_results_sasp.csv",
    "evaluation_results_sacp.csv",
]

MODEL_ID = "gemini-2.5-flash"


def load_csvs() -> Dict[str, pd.DataFrame]:
    dfs = {}
    for csv_file in CSV_FILES:
        path = os.path.join(CSV_DIR, csv_file)
        dfs[csv_file] = pd.read_csv(path)
    return dfs


def validate_dirs(dfs: Dict[str, pd.DataFrame]) -> List[str]:
    dir_names = [set(df["directory_name"].unique()) for df in dfs.values()]

    assert dir_names[0] == dir_names[1]
    assert dir_names[1] == dir_names[2]
    assert dir_names[2] == dir_names[3]
    assert dir_names[0] == dir_names[3]

    return sorted(list(dir_names[0]))


def build_prompt_parts(submission_dir: str) -> List[types.Part]:
    prompt_parts: List[types.Part] = []

    prompt_parts.append(types.Part.from_text(text="Here is the main_paper.pdf for the paper"))
    with open(f"{submission_dir}/main_paper.pdf", "rb") as f:
        prompt_parts.append(types.Part.from_bytes(
            data=f.read(),
            mime_type="application/pdf"
        ))

    supp_path = os.path.join(submission_dir, "supplemental_files")
    if os.path.exists(supp_path):
        process_supplemental_files(supp_path, prompt_parts)

    return prompt_parts


def check_token_error(submission_dir: str, engine) -> bool:
    """Returns True if main_paper_only was used, False if full content sent."""
    try:
        prompt_parts = build_prompt_parts(submission_dir)
        limit = engine.get_model_limit()

        valid_parts = [
            p for p in prompt_parts
            if (getattr(p, 'text', None) and p.text.strip()) or
               (getattr(p, 'inline_data', None) and len(p.inline_data.data) > 0)
        ]

        try:
            total_tokens = engine.count_tokens(valid_parts)
        except errors.ClientError as e:
            return True

        return total_tokens > limit
    except Exception:
        return None


def process_submissions(submission_dirs: List[str]) -> List[bool]:
    engine = create_engine(
        model_id=MODEL_ID,
        pydantic_model=FormattingCheck,
        system_instruction="",
        thinking_included=False,
        search_included=False,
    )

    results = []
    for idx, sub_dir in enumerate(submission_dirs):
        if idx % 20 == 0:
            print(f"Progress: {idx}/{len(submission_dirs)}")

        result = check_token_error(sub_dir, engine)
        results.append(result)

    return results


def display_summary(dfs: Dict[str, pd.DataFrame]) -> None:
    print("\n" + "="*80)
    print("TOKEN ERROR SUMMARY")
    print("="*80)

    for csv_file, df in dfs.items():
        if "token_error_occurred" in df.columns:
            errors = df["token_error_occurred"].sum()
            total = len(df)
            pct = (errors / total * 100) if total > 0 else 0
            print(f"\n{csv_file}:")
            print(f"  Token errors: {errors}/{total} ({pct:.1f}%)")
            print(f"  Head (first 5):")
            head_df = df[["directory_name", "token_error_occurred"]].head()
            print(head_df.to_string(index=False))


def save_csvs(dfs: Dict[str, pd.DataFrame]) -> None:
    for csv_file, df in dfs.items():
        path = os.path.join(CSV_DIR, csv_file)
        df.to_csv(path, index=False)
        print(f"Saved {csv_file}")


def main():
    print("Loading CSVs...")
    dfs = load_csvs()

    print("Validating directories...")
    submission_dirs = validate_dirs(dfs)
    print(f"✓ All datasets have {len(submission_dirs)} unique submissions")

    print("Checking token errors...")
    results = process_submissions(submission_dirs)

    for csv_file in CSV_FILES:
        dfs[csv_file]["token_error_occurred"] = results

    display_summary(dfs)

    response = input("\nSave results? (yes/no): ").strip().lower()
    if response == "yes":
        save_csvs(dfs)
        print("✓ Saved successfully")
    else:
        print("Cancelled")


if __name__ == "__main__":
    print(os.getcwd())
    main()
