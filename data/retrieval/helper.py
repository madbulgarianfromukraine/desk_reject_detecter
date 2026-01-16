import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import openreview.api

from typing import Optional, List, Dict, Any
from data.retrieval.util.threading import num_workers

def get_note_value(note: openreview.api.Note, field: str = "") -> Optional[str]:
    """Safely extracts the string 'value' from a nested note content dictionary."""
    return note.content.get(field, {}).get('value', None)

def filter_proper_desk_rejections(client: openreview.api.OpenReviewClient, initial_desk_rejections: List[openreview.api.Note]) -> List[Dict[str, Any]]:
    submissions_to_process = []
    for i, submission in enumerate(initial_desk_rejections):

        # 1. Check for mandatory PDF path
        pdf_path = get_note_value(submission, 'pdf')

        if pdf_path is None:
            print(f"Desk Rejected Submission: ❌ Skipping Submission ID {submission.id} and {submission.content["title"]}: No main PDF path found.")
            continue

        # 2. Check for mandatory desk reject comment existence (metadata check)
        comment_notes = client.get_all_notes(replyto=submission.id, details='content')
        desk_reject_notes = []

        for note in comment_notes:
            # We check for the presence of the required reason field
            if get_note_value(note=note, field="desk_reject_comments"):
                desk_reject_notes.append(note)

        # 3. CRITICAL FILTER: Check if the amount is exactly 1
        if len(desk_reject_notes) != 1:
            # If 0 (no reason found) or >1 (ambiguous), skip the submission
            print(f"Desk Rejected Submission: ❌ Skipping Submission ID {submission.id}: Found {len(desk_reject_notes)} desk reject notes (must be exactly 1).")
            continue

        # If all checks pass, add the note and its corresponding comment note to the processing list
        submissions_to_process.append({
            'submission': submission,
            'comment_note': desk_reject_notes[0], # If we reached this point, exactly one valid desk_reject_note was found.
        })

    return submissions_to_process

def filter_proper_accepted_papers(
    client: openreview.api.OpenReviewClient,
    initial_accepted_papers: List[openreview.api.Note],
    dr_submissions_count: int,
    desk_rejection_ids: Optional[List[str]] = None,
    withdrawal_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    # Remove any papers that are desk-rejected or withdrawn (moved from iclr_2025_main.py)
    try:
        excluded_ids = set(desk_rejection_ids or []) | set(withdrawal_ids or [])
    except Exception:
        excluded_ids = set()

    if excluded_ids:
        all_unique_forum_ids = {note.forum for note in initial_accepted_papers}
        ndr_unique_forum_ids = all_unique_forum_ids - excluded_ids
        initial_accepted_papers = [
            note for note in initial_accepted_papers
            if (note.forum not in excluded_ids and note.id not in excluded_ids)
        ]
        removed = len(all_unique_forum_ids) - len(ndr_unique_forum_ids)
        if removed:
            print(
                f"Filtered out {removed} submissions due to desk-rejection/withdrawal before processing accepted."
            )

    submissions_to_process: List[Dict[str, Any]] = []

    def __process_accepted_paper(submission: openreview.api.Note) -> Optional[Dict[str, Any]]:
        # 1. Check for mandatory PDF path

        pdf_path = get_note_value(submission, 'pdf')
        if pdf_path is None:
            print(f"Not Desk Rejected Submission:❌ Skipping Submission ID {submission.id} and {submission.content['title']}: No main PDF path found.")
            return None


        # 2. Fetch related notes and check for decision
        try:
            comment_notes = client.get_all_notes(replyto=submission.id, details='content')
        except Exception as e:
            print(f"Not Desk Rejected Submission:❌ Skipping Submission ID {submission.id}: failed to fetch comment notes: {e}")
            return None

        has_decision = False
        for note in comment_notes:
            if get_note_value(note=note, field="decision"):
                has_decision = True
                break

        if has_decision:
            return {
                'submission': submission,
                'comment_note': None
            }
        else:
            print(f"Not Desk Rejected Submission:❌ Skipping Submission ID {submission.id} and {submission.content['title']}: No Decision Note found.")
            return None

    with ThreadPoolExecutor(max_workers=num_workers(), thread_name_prefix=f"NDR-filtering-") as executor:
        future_map = {executor.submit(__process_accepted_paper, sub): sub for sub in initial_accepted_papers}
        for future in as_completed(future_map):
            try:
                result = future.result()
            except Exception as e:
                # Log and skip this submission on unexpected worker error
                sub = future_map[future]
                print(f"Not Desk Rejected Submission:❌ Skipping Submission ID {sub.id}: worker error: {e}")
                continue

            if result is not None:
                submissions_to_process.append(result)

    MAX_NDR_SAMPLE_SIZE = 3 * dr_submissions_count
    current_ndr_count = len(submissions_to_process)
    if current_ndr_count > MAX_NDR_SAMPLE_SIZE:
        random.shuffle(submissions_to_process)

        # b. Sample the required number of items from the shuffled list
        submissions_to_process = submissions_to_process[:MAX_NDR_SAMPLE_SIZE]

        print(f"Sampling Applied: Original NDR count ({current_ndr_count}) > Max allowed ({MAX_NDR_SAMPLE_SIZE}).")
        print(f"Final NDR sample size: {len(submissions_to_process)}")

    return submissions_to_process
