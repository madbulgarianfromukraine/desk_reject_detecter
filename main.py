import fire
import sys
import os
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional, Union
import google.auth
from google.auth.exceptions import DefaultCredentialsError

from dotenv import load_dotenv
load_dotenv(dotenv_path='./google.env', verbose=True) # importing all the env variables before our project imports.

from systems.ddr import ddr
from core.schemas import FinalDecision
from core.log import LOG, configure_logging
from core.metrics import evaluate_submission_answers_only, evaluate_submission_full
from core.utils import cleanup_caches

AVAILABLE_SYSTEMS = [
    'ddr', # desk reject detecter
    'ddr-1-iteration' # desk reject_detecter with only 1 iteration
    'ddr-think-search'
    'sasp', #single agent single prompt
    'sacp', # single agent multiple prompt
]

def ensure_authenticated():
    try:
        # Attempt to refresh/load credentials
        credentials, project = google.auth.default()

        # Check if they are valid (or can be refreshed)
        if not credentials.valid:
            from google.auth.transport.requests import Request

            credentials.refresh(Request())

        print(f"✅ Already authenticated for project: {project}")
        return credentials

    except (DefaultCredentialsError, Exception):
        # This block runs ONLY if no credentials are found
        print("❌ No valid credentials found. Launching login...")
        try:
            subprocess.run(["gcloud", "auth", "application-default", "login"], check=True)
            # Re-verify after login
            credentials, project = google.auth.default()
            return credentials
        except subprocess.CalledProcessError:
            print("🚨 Login failed or was cancelled by the user.")
            return None


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

    def determine_desk_rejection(self, directory: str, think: bool = False, search: bool = False, iterations: int = 3) -> str:
        """
        Runs the full protocol and outputs a binding YES/NO decision. Usage: python cli.py determine_desk_rejection ./my_paper_folder
        :param directory: Directory of the paper submission.
        :param think: Whether to use thinking for agents.
        :param search: Whether to use search for agents.
        :param iterations: The maximum number of self-correction iterations for agents.
        """
        LOG.debug(f"--- DETERMINING DESK REJECTION FOR: {directory.split(sep='/')[-1]} ---")

        try:
            # Call the pipeline from main.py
            final_decision = ddr(directory, think=think, search=search, iterations=iterations)
            return final_decision.model_dump_json()
        except Exception as e:
            LOG.error(f"Pipeline failed: {e}")
            sys.exit(1)

    def evaluate_desk_rejection(self, directory: str,
                                system_used: str = 'ddr',
                                parallel: bool = False,
                                answers_only: bool = False, limit: int = None) -> None:
        """
        Runs an evaluation of all submissions in the directory and produces a report without a binding decision.
        Usage: python cli.py evaluate_desk_rejection ./my_paper_folder --limit 5
        :param directory: Directory of the paper submission.
        :param parallel: Whether to run in parallel mode.
        :param answers_only: Evaluate only the precision of the answer or also consider the precision of the reason for desk rejection.
        :param limit: Limits the amount of tested instances.
        """
        eval_results = {}
        LOG.debug(f"--- EVALUATING {directory.split(sep='/')[-1]} with answers_only={answers_only} ---")

        desk_rejection_system : Optional[Callable] = None

        match system_used:
            case 'sasp':
                from systems.sasp import sasp
                desk_rejection_system = sasp
            case 'sacp':
                from systems.sacp import sacp
                desk_rejection_system = sacp
            case 'ddr-1-iteration':
                def __run_ddr_1_iteration(path_sub_dir: Union[os.PathLike, str], think: bool = False, search: bool = False) -> FinalDecision:
                    return ddr(path_sub_dir=path_sub_dir, think=think, search=search, iterations=1, ttl_seconds="10800s")

                desk_rejection_system = __run_ddr_1_iteration
            case 'ddr-think-search':
                def __run_ddr_think_search(path_sub_dir: Union[os.PathLike, str]) -> FinalDecision:
                    return ddr(path_sub_dir=path_sub_dir, think=True, search=True, ttl_seconds="10800s")

                desk_rejection_system = __run_ddr_think_search
            case _ :
                def __run_ddr_default(path_sub_dir: Union[os.PathLike, str], think: bool = False, search: bool = False) -> FinalDecision:
                    return ddr(path_sub_dir=path_sub_dir, think=think, search=search, ttl_seconds="10800s")

                desk_rejection_system = __run_ddr_default

        subdirs = [os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        
        # Apply the limit if provided
        if limit is None or limit <= 0:
            limit = len(subdirs)
        else:
            limit = min(limit, len(subdirs))

        if limit < len(subdirs):
            random.seed(42)
            subdirs = random.sample(subdirs, limit)
        
        if parallel:
            with ThreadPoolExecutor(thread_name_prefix="Directory_Evaluation") as executor:
                future_to_eval_result = {executor.submit(desk_rejection_system, diry): diry for diry in subdirs}

                for future in as_completed(future_to_eval_result):
                    evaluation_paper_dir = future_to_eval_result[future]
                    try:
                        eval_results[evaluation_paper_dir] = future.result()
                        LOG.debug(f"{evaluation_paper_dir} determination of desk rejection completed.")
                    except Exception as exc:
                        eval_results[evaluation_paper_dir] = None
                        LOG.error(f"{evaluation_paper_dir} generated an exception: {exc}")


        else:
            for diry in subdirs:
                try:
                    evaluation_paper_result = desk_rejection_system(diry)
                    eval_results[diry] = evaluation_paper_result
                except Exception as exc:
                    eval_results[diry] = None
                    LOG.error(f"{diry} generated an exception: {exc}")

        if answers_only:
            return evaluate_submission_answers_only(evaluation_results=eval_results)
        return evaluate_submission_full(evaluation_results=eval_results, system_used=system_used)



if __name__ == "__main__":
    try:
        ensure_authenticated()
        fire.Fire(DeskRejectionCLI)
    except KeyboardInterrupt:
        LOG.info("Received KeyboardInterrupt. Cleaning up resources...")
        cleanup_caches()
        sys.exit(0)
    finally:
        cleanup_caches()