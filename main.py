import fire
import sys
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from ddr import desk_rejection_system
from core.log import LOG, configure_logging
from core.metrics import evaluate_submission_answers_only, evaluate_submission_full

AVAILABLE_SYSTEMS = [
    'ddr', # desk reject detecter
    'ddr-1-iteration' # desk reject_detecter with only 1 iteration
    'sasp',
    'sacp',
]

class DeskRejectionCLI:
    """
    A CLI tool for analyzing ICLR paper submissions for desk rejection criteria.
    """

    def __init__(self, log_level: str = "WARNING") -> None:
        """
        Constructs a new TSCloudCTL CLI instance and does initial setup before the subcommand is executed by Fire.

        :param log_level: The log level to use throughout the program
        """
        super().__init__()
        DeskRejectionCLI.__log_level = log_level.strip().upper()

        configure_logging(DeskRejectionCLI.__log_level)

    def determine_desk_rejection(self, directory: str, think: bool = False, search: bool = False) -> str:
        """
        Runs the full protocol and outputs a binding YES/NO decision. Usage: python cli.py determine_desk_rejection ./my_paper_folder
        :param directory: Directory of the paper submission.
        :param think: Whether to use thinking for agents.
        :param search: Whether to use search for agents.
        """
        LOG.debug(f"--- DETERMINING DESK REJECTION FOR: {directory.split(sep='/')[-1]} ---")

        try:
            # Call the pipeline from main.py
            final_decision = desk_rejection_system(directory, think=think, search=search)
            return final_decision.model_dump_json()
        except Exception as e:
            LOG.error(f"Pipeline failed: {e}")
            sys.exit(1)

    def evaluate_desk_rejection(self, directory: str,
                                system_used: str = 'ddr',
                                parallel: bool = False,
                                answers_only: bool = False, think: bool = False, 
                                search: bool = False, limit: int = None) -> None:
        """
        Runs an evaluation of all submissions in the directory and produces a report without a binding decision.
        Usage: python cli.py evaluate_desk_rejection ./my_paper_folder --limit 5
        :param directory: Directory of the paper submission.
        :param parallel: Whether to run in parallel mode.
        :param answers_only: Evaluate only the precision of the answer or also consider the precision of the reason for desk rejection.
        :param think: Whether to use thinking for agents.
        :param search: Whether to use search for agents.
        :param limit: Limits the amount of tested instances.
        """
        eval_results = {}
        LOG.debug(f"--- EVALUATING {directory.split(sep='/')[-1]} with answers_only={answers_only} and think={think} and search={search} ---")

        if system_used is None:
            LOG.warn("No system was specified, using ddr")
            system_used = 'ddr'

        if system_used not in AVAILABLE_SYSTEMS:
            LOG.warn(f"The system {system_used} is not supported. Defaulting to ddr")
            system_used = 'ddr'
        subdirs = [os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        
        # Apply the limit if provided
        if limit <= 0 or not limit:
            limit = len(subdirs)
        else:
            limit = min(limit, len(subdirs))

        if limit < len(subdirs):
            subdirs = random.sample(subdirs, limit)
        
        if parallel:
            with ThreadPoolExecutor() as executor:
                future_to_eval_result = {executor.submit(desk_rejection_system, diry, think=think, search=search): diry for diry in subdirs}

                for future in as_completed(future_to_eval_result):
                    evaluation_paper_dir = future_to_eval_result[future]
                    try:
                        eval_results[evaluation_paper_dir] = future.result()
                        LOG.debug(f"{evaluation_paper_dir} determination of desk rejection completed.")
                    except Exception as exc:
                        LOG.error(f"{evaluation_paper_dir} generated an exception: {exc}")

        else:
            for diry in subdirs:
                try:
                    evaluation_paper_result = desk_rejection_system(diry, think=think, search=search)
                    eval_results[diry] = evaluation_paper_result
                except Exception as exc:
                    LOG.error(f"{diry} generated an exception: {exc}")

        if answers_only:
            return evaluate_submission_answers_only(evaluation_results=eval_results)
        return evaluate_submission_full(evaluation_results=eval_results)



if __name__ == "__main__":
    fire.Fire(DeskRejectionCLI)