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
            print(f"❌ Skipping Submission ID {submission.id} and {submission.content["title"]}: No main PDF path found.")
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
            print(f"❌ Skipping Submission ID {submission.id}: Found {len(desk_reject_notes)} desk reject notes (must be exactly 1).")
            continue

        # If all checks pass, add the note and its corresponding comment note to the processing list
        submissions_to_process.append({
            'submission': submission,
            'comment_note': desk_reject_notes[0], # If we reached this point, exactly one valid desk_reject_note was found.
            'index': i # Keep original index for unique directory naming
        })

    return submissions_to_process

def filter_proper_accepted_papers(client: openreview.api.OpenReviewClient, initial_accepted_papers: List[openreview.api.Note]) -> List[Dict[str, Any]]:
    submissions_to_process = []
    for i, submission in enumerate(initial_accepted_papers):

        # 1. Check for mandatory PDF path
        pdf_path = get_note_value(submission, 'pdf')

        if pdf_path is None:
            print(f"❌ Skipping Submission ID {submission.id} and {submission.content["title"]}: No main PDF path found.")
            continue

        comment_notes = client.get_all_notes(replyto=submission.id, details='content')

        for note in comment_notes:
            # We check for the presence of the required decision field
            if get_note_value(note=note, field="decision"):
                submissions_to_process.append(note)

    return submissions_to_process
