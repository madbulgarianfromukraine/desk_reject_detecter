SUBMISSIONS_CSV_PATH = "../iclr/data/submissions.csv"

import pandas as pd
from pydantic import BaseModel, ConfigDict
from typing import List, Literal, Tuple
from google.genai import types

from dotenv import load_dotenv
load_dotenv(dotenv_path="../../google.env", verbose=True)

from core.config import VertexEngine

submissions_df = pd.read_csv(SUBMISSIONS_CSV_PATH)

dr_reason = submissions_df["desk_reject_comments"]

engine = VertexEngine(model_id="gemini-2.5-pro")

engine.set_system_instruction(
    """
    You will be given a column from the .csv file which specifies the reason for desk-rejection or <null> if it was not a desk reject
    Your task is to classify them according to categories("Code_of_Ethics", "Anonymity", "Formatting", "Visual_Integrity", "Policy", "Scope") of desk rejection or None if it wasn't a desk rejection.
    
    Now here is which reasons count to corresponding categories:
    * Code_of_Ethics: "Privacy", "Harm", "Misconduct"
    * Anonymity: "Author_Names", "Visual_Anonymity", "Self-Citation", "Links"
    * Formatting : "Page_Limit", "Statement_Limit", "Margins/Spacing", "Line_Numbers"
    * Visual_Integrity: "Placeholder_Figures", "Unreadable_Content", "Broken_Rendering"
    * Policy : "Placeholder_Text", "Dual_Submission", "Plagiarism"
    * Scope: "Scope", "Language"
    
    Ignore all the NaNs at the input, but return the index for each classified category(The corresponding indices from the input value)
    categories and indices attributes must have the same length.
    """
)

class DRCategorizationResults(BaseModel):
    categories: List[Literal["Code_of_Ethics", "Anonymity", "Formatting", "Visual_Integrity", "Policy", "Scope", "None"]]
    indices: List[int]

engine.set_schema(schema=DRCategorizationResults)

cat_results = engine.generate(contents=[types.Part.from_text(
        text=dr_reason.to_string()
    )])

indices, categories = cat_results.parsed.indices, cat_results.parsed.categories

submissions_df["category"] = None

submissions_df.loc[indices, "category"] = categories

submissions_df.to_csv(SUBMISSIONS_CSV_PATH, index=False)

