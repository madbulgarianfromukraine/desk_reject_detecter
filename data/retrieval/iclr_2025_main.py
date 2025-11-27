import os
import csv
import zipfile
import tarfile
import shutil
import urllib.parse
import openreview.api
from openreview import OpenReviewException
from dotenv import load_dotenv

from typing import Optional


#definin the environmental variables
load_dotenv(verbose=True)

USER_NAME = os.getenv("USER_NAME")
PASSWORD = os.getenv("USER_PASSWORD")
BASE_URL = "https://api2.openreview.net"

client = openreview.api.OpenReviewClient(
    baseurl=BASE_URL,
    username=USER_NAME,
    password=PASSWORD
)

# Assuming 'client' is initialized.
CONFERENCE_ID = 'ICLR.cc/2025/Conference'
DESK_REJECT_INVITATION = f'{CONFERENCE_ID}/-/Desk_Rejected_Submission'


# 1. Get all Notes posted using the correct master list invitation
initial_desk_rejections = client.get_all_notes(
    invitation=DESK_REJECT_INVITATION,
    details='content'
)

desk_rejection_comments = len(initial_desk_rejections)*[None]

# --- Global Data Structure for CSV Output ---
final_csv_data = []

# --- Helper Functions for Clean Code ---

def get_note_value(note: openreview.api.Note, field: str = "") -> Optional[str]:
    """Safely extracts the string 'value' from a nested note content dictionary."""
    return note.content.get(field, {}).get('value', None)

def download_file(client, note_id, field_note, output_path, is_pdf=False):
    """Handles API call, error checking, and saving for a single file."""
    try:
        if is_pdf:
            binary_data = client.get_pdf(note_id)
        else:
            binary_data = client.get_attachment(id=note_id, field_note=field_note)

        with open(output_path, 'wb') as f:
            f.write(binary_data)

        return True

    except OpenReviewException as e:
        # Catch 403 Forbidden, 404 Not Found, etc.
        print(f"    ❌ Download Failed: {field_note} for {note_id} is restricted/missing. Error: {e}")
        return False
    except Exception as e:
        # Catch file system or other errors
        print(f"    ❌ Download Failed: Non-API error for {field_note}. Error: {e}")
        return False

def main():
    submissions_to_process = []
    for i, submission in enumerate(initial_desk_rejections):

        # 1. Check for mandatory PDF path
        pdf_path = get_note_value(submission, 'pdf')
        if pdf_path is None:
            print(f"❌ Skipping Submission ID {submission.id} and {submission.content["title"]}: No main PDF path found.")
            continue

        # 2. Check for mandatory desk reject comment existence (metadata check)
        # We must find the comment note first to check its content
        comment_notes = client.get_all_notes(replyto=submission.id, details='content')
        desk_reject_comment_note = comment_notes[0] if comment_notes else None

        if desk_reject_comment_note is None:
            print(f"❌ Skipping Submission ID {submission.id}: No Desk Reject Comment Note found.")
            continue

        # 3. Check for mandatory desk reject content keys (weaknesses or summary)
        rejection_content = desk_reject_comment_note.content
        if not (rejection_content.get('weaknesses') or rejection_content.get('summary')):
            print(f"❌ Skipping Submission ID {submission.id}: Comment Note is empty or lacks reason keys (Weaknesses/Summary).")
            continue

        # If all checks pass, add the note and its corresponding comment note to the processing list
        submissions_to_process.append({
            'submission': submission,
            'comment_note': desk_reject_comment_note,
            'index': i # Keep original index for unique directory naming
        })


    print(f"\n--- Starting Download and Processing for {len(submissions_to_process)} Valid Submissions ---")

    for item in submissions_to_process:
        submission = item['submission']
        comment_note = item['comment_note']
        i = item['index']

        submission_id = submission.id

        # --- Data Extraction ---
        pdf_path = get_note_value(submission, 'pdf')
        title = get_note_value(submission, 'title')

        # Extract rejection reason (prioritizing Weaknesses > Summary)
        reason = get_note_value(comment_note, 'weaknesses')
        if not reason:
            reason = get_note_value(comment_note, 'summary')
        if not reason:
            reason = "Reason not explicitly tagged (Weaknesses/Summary missing)."

        # --- Directory Setup (Check 4) ---
        dir_name = f"submission_{i}_{submission_id}"
        base_dir = f"data/iclr/downloads/{dir_name}"
        supplemental_dir = os.path.join(base_dir, "supplemental_files")

        os.makedirs(base_dir, exist_ok=True)

        download_success_status = True

        print(f"\n[{i+1}] Processing Submission: {submission_id} ({title})")

        # --- 1. Download Main PDF ---
        pdf_filename = os.path.join(base_dir, f"main_paper.pdf")
        if not download_file(client, submission_id, 'pdf', pdf_filename, is_pdf=True):
            # If PDF fails, we cannot use this submission for analysis
            shutil.rmtree(base_dir, ignore_errors=True)
            download_success_status = False
            continue

            # --- 2. Download Supplementary Material (Optional) ---
        supplementary_material_path = get_note_value(submission, 'supplementary_material')
        supplemental_downloaded = False

        if supplementary_material_path:

            # Get extension for file saving
            parsed_path = urllib.parse.urlparse(supplementary_material_path).path
            _, file_ext = os.path.splitext(os.path.basename(parsed_path))

            # Use a temporary file for the download
            temp_filename = os.path.join(base_dir, f"supplemental_archive{file_ext}")

            # Attempt download
            if download_file(client, submission_id, 'supplementary_material', temp_filename, is_pdf=False):

                # Create subdirectory for extraction
                os.makedirs(supplemental_dir, exist_ok=True)
                supplemental_downloaded = True

                # Handle Extraction (ZIP/TAR/TGZ)
                if file_ext.lower() in ['.zip', '.tgz', '.tar']:
                    try:
                        if file_ext.lower() == '.zip':
                            with zipfile.ZipFile(temp_filename, 'r') as zip_ref:
                                zip_ref.extractall(supplemental_dir)
                        else: # Handle tar/tgz
                            import tarfile
                            mode = 'r:gz' if file_ext.lower() == '.tgz' else 'r'
                            with tarfile.open(temp_filename, mode) as tar_ref:
                                tar_ref.extractall(supplemental_dir)

                        print(f"    📦 Extracted {file_ext.upper()} into {supplemental_dir}.")
                        os.remove(temp_filename) # Delete archive after successful extraction

                    except Exception as e:
                        print(f"    ❌ Error extracting archive: {e}. Keeping original archive.")
                        shutil.move(temp_filename, os.path.join(supplemental_dir, f"supplemental_archive{file_ext}")) # Move unextracted archive to subdir

                else:
                    # If it's a single PDF/other file, just move it to the subdirectory
                    shutil.move(temp_filename, os.path.join(supplemental_dir, f"supplemental{file_ext}"))

        # --- 3. Collect Final CSV Data (Check 4) ---
        final_csv_data.append({
            'submission_id': submission_id,
            'directory_name': base_dir,
            'status': 'Desk Rejected',
            'desk_reject_comments': reason,
            'supplemental_downloaded': supplemental_downloaded
        })

    # --- Final CSV Output ---
    CSV_FILENAME = "desk_rejected_submissions_analysis.csv"
    if final_csv_data:
        csv_fieldnames = ['submission_id', 'directory_name', 'status', 'desk_reject_comments', 'supplemental_downloaded']

        try:
            with open(CSV_FILENAME, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_fieldnames)
                writer.writeheader()
                writer.writerows(final_csv_data)
            print(f"\n🎉 Successfully created final analysis CSV: **{CSV_FILENAME}** with {len(final_csv_data)} records.")
        except Exception as e:
            print(f"\n❌ Error writing final CSV file: {e}")
    else:
        print("\n⚠️ No valid submissions were processed for final CSV output.")

if __name__ == "__main__":
    main()


