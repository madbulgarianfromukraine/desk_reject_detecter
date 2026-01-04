import fire
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from ddr import desk_rejection_system
from core.log import LOG
from core.schemas import FinalDecision
from core.metrics import evaluate_submission_answers_only, evaluate_submission_full



class DeskRejectionCLI:
    """
    A CLI tool for analyzing ICLR paper submissions for desk rejection criteria.
    """

    def determine_desk_rejection(self, directory: str) -> FinalDecision:
        """
        Runs the full protocol and outputs a binding YES/NO decision. Usage: python cli.py determine_desk_rejection ./my_paper_folder
        :param directory: Directory of the paper submission.
        """
        LOG.debug(f"--- DETERMINING DESK REJECTION FOR: {directory} ---")

        try:
            # Call the pipeline from main.py
            final_decision = desk_rejection_system(directory)
            LOG.info(final_decision)
            return final_decision
        except Exception as e:
            LOG.error(f"Pipeline failed: {e}")
            sys.exit(1)

    def evaluate_desk_rejection(self, directory: str, parallel: bool = False,
                                answers_only: bool = False) -> None:
        """
        Runs an evaluation of all submissions in the directory and produces a report without a binding decision.
        Usage: python cli.py evaluate_desk_rejection ./my_paper_folder
        :param parallel: Whether to run in parallel mode.
        :param answers_only: Evaluate only the precision of the answer or also consider the precision of the reason for desk rejection.
        """
        eval_results = {}
        LOG.debug(f"--- EVALUATING SUBMISSION: {directory} ---")
        if parallel:
            with ThreadPoolExecutor() as executor:
                future_to_eval_result = {executor.submit(desk_rejection_system, diry): diry for diry in os.listdir(directory) if os.path.isdir(diry)}

                for future in as_completed(future_to_eval_result):
                    evaluation_paper_dir = future_to_eval_result[future]
                    try:
                        eval_results[evaluation_paper_dir] = future.result()
                        LOG.debug(f"{evaluation_paper_dir} determination of desk rejection completed.")
                    except Exception as exc:
                        LOG.error(f"{evaluation_paper_dir} generated an exception: {exc}")

        else:
            for diry in os.listdir(directory):

                    evaluation_paper_result = desk_rejection_system(diry)
                    eval_results[diry] = evaluation_paper_result
            try:
                # We reuse the same pipeline but wrapping it differently for the user
                final_decision = desk_rejection_system(directory)
                LOG.info(final_decision)

            except Exception as e:
                print(f"Evaluation failed: {e}")
                sys.exit(1)


        if answers_only:
            return evaluate_submission_answers_only(evaluation_results=eval_results)
        return evaluation_submission_full(evaluation_results=eval_results)



if __name__ == "__main__":
    fire.Fire(DeskRejectionCLI)