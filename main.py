import fire
import sys
import os

from ddr import desk_rejection_system



class DeskRejectionCLI:
    """
    A CLI tool for analyzing ICLR paper submissions for desk rejection criteria.
    """

    def determine_desk_rejection(self, directory: str):
        """
        Runs the full protocol and outputs a binding YES/NO decision.

        Usage: python cli.py determine_desk_rejection ./my_paper_folder
        """
        print(f"--- DETERMINING DESK REJECTION FOR: {directory} ---")

        pdf_path, style_paths = self._find_files(directory)
        print(f"Found PDF: {os.path.basename(pdf_path)}")

        try:
            # Call the pipeline from main.py
            json_result = desk_rejection_pipeline(pdf_path, style_paths)
            print("\n=== FINAL DECISION ===")
            print(json_result)
        except Exception as e:
            print(f"Pipeline failed: {e}")
            sys.exit(1)

    def evaluate_desk_rejection(self, directory: str):
        """
        Runs an evaluation and produces a report without a binding decision.

        Usage: python cli.py evaluate_desk_rejection ./my_paper_folder
        """
        print(f"--- EVALUATING SUBMISSION: {directory} ---")

        pdf_path, style_paths = self._find_files(directory)

        try:
            # We reuse the same pipeline but wrapping it differently for the user
            json_result = desk_rejection_system(directory)
            print("\n=== EVALUATION REPORT ===")
            print(json_result)
        except Exception as e:
            print(f"Evaluation failed: {e}")
            sys.exit(1)


if __name__ == "__main__":
    fire.Fire(DeskRejectionCLI)