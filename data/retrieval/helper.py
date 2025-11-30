import random
import openreview.api

from typing import Optional, List, Dict, Any

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

    submissions_to_process = []
    for i, submission in enumerate(initial_accepted_papers):

        # 1. Check for mandatory PDF path
        pdf_path = get_note_value(submission, 'pdf')

        if pdf_path is None:
            print(f"Not Desk Rejected Submission:❌ Skipping Submission ID {submission.id} and {submission.content["title"]}: No main PDF path found.")
            continue

        comment_notes = client.get_all_notes(replyto=submission.id, details='content')

        for note in comment_notes:
            # We check for the presence of the required decision field
            if get_note_value(note=note, field="decision"):
                submissions_to_process.append({
                    'submission': submission,
                    'comment_note': None
                })
            else:
                print(f"Not Desk Rejected Submission:❌ Skipping Submission ID {submission.id} and {submission.content["title"]}: No Decision Note found.")

    MAX_NDR_SAMPLE_SIZE = 3 * dr_submissions_count
    current_ndr_count = len(submissions_to_process)
    if current_ndr_count > MAX_NDR_SAMPLE_SIZE:
        random.shuffle(submissions_to_process)

        # b. Sample the required number of items from the shuffled list
        submissions_to_process = submissions_to_process[:MAX_NDR_SAMPLE_SIZE]

        print(f"Sampling Applied: Original NDR count ({current_ndr_count}) > Max allowed ({MAX_NDR_SAMPLE_SIZE}).")
        print(f"Final NDR sample size: {len(submissions_to_process)}")

    return submissions_to_process
