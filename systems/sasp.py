from typing import Union
import os
from core.schemas import FinalDecision
from core.metrics import SubmissionMetrics
# Model configuration
MODEL_ID = "gemini-2.5-flash"

def sasp(path_sub_dir: Union[os.PathLike, str], think: bool = False, search: bool = False) -> SubmissionMetrics:
    """
    Placeholder for Single Agent Single Prompt system.
    """
    pass
