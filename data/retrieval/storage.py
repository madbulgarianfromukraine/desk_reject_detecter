import openreview.api
import csv
import os
import shutil
import urllib.parse
import zipfile
from typing import Optional, List, Dict, Any
from openreview import OpenReviewException
import threading

from helper import get_note_value

CSV_WRITE_LOCK = threading.Lock()

def download_file(client: openreview.api.OpenReviewClient, note_id: Optional[str], field_name: Optional[str] = "", output_path : str = "main.pdf", is_pdf: bool=False) -> bool:
    """Handles API call, error checking, and saving for a single file."""
    try:
        if is_pdf:
            binary_data = client.get_pdf(note_id)
        else:
            binary_data = client.get_attachment(id=note_id, field_name=field_name)

        with open(output_path, 'wb') as f:
            f.write(binary_data)

        return True

    except OpenReviewException as e:
        # Catch 403 Forbidden, 404 Not Found, etc.
        print(f"    ❌ Download Failed: {field_name} for {note_id} is restricted/missing. Error: {e}")
        return False
    except Exception as e:
        # Catch file system or other errors
        print(f"    ❌ Download Failed: Non-API error for {field_name}. Error: {e}")
        return False


def store_main_and_supplemental_materials(client: openreview.api.OpenReviewClient,submissions_to_process: List[Dict[str, Any]], csv_data: List[Dict[str, Any]], desk_rejection: bool = False) -> None:
    for item in submissions_to_process:
        submission = item['submission']
        comment_note = item['comment_note']
        i = item['index']

        submission_id = submission.id
        pdf_path = get_note_value(submission, 'pdf')
        title = get_note_value(submission, 'title')

        dir_name = f"submission_{submission_id}"
        base_dir = f"data/iclr/data/{dir_name}"

        os.makedirs(base_dir, exist_ok=True)


        print(f"\n[{i+1}] Processing Submission: {submission_id} ({title})")

        # --- 1. Download Main PDF ---
        pdf_filename = os.path.join(base_dir, f"main_paper.pdf")
        if not download_file(client, submission_id, 'pdf', pdf_filename, is_pdf=True):
            # If PDF fails, we cannot use this submission for analysis
            shutil.rmtree(base_dir, ignore_errors=True)
            continue

        # --- 2. Download Supplementary Material if the link is there
        supplementary_material_path = get_note_value(submission, 'supplementary_material')
        supplemental_downloaded = False

        if supplementary_material_path:

            # Get extension for file saving
            parsed_path = urllib.parse.urlparse(supplementary_material_path).path
            _, file_ext = os.path.splitext(os.path.basename(parsed_path))

            # Use a temporary file for the download
            temp_filename = os.path.join(base_dir, f"supplemental_archive{file_ext}")

            # 1. Attempt the download (download_file returns True on success, False on failure)
            supplemental_download_successful = download_file(
                client,
                submission_id,
                'supplementary_material',
                temp_filename,
                is_pdf=False
            )

            if not supplemental_download_successful:
                shutil.rmtree(base_dir, ignore_errors=True)
                continue
            else:
                # --- Download Succeeded (Proceed with File Management) ---
                supplemental_dir = os.path.join(base_dir, "supplemental_files")
                # Create subdirectory for extraction
                os.makedirs(supplemental_dir, exist_ok=True)
                supplemental_downloaded = True # Mark for CSV logging

                # Handle Extraction (ZIP/TAR/TGZ)
                if file_ext.lower() == '.zip':
                    try:
                        with zipfile.ZipFile(temp_filename, 'r') as zip_ref:
                            zip_ref.extractall(supplemental_dir)

                        print(f"    📦 Extracted {file_ext.upper()} into {supplemental_dir}.")
                        os.remove(temp_filename)

                    except Exception as e:
                        print(f"    ❌ Error extracting archive: {e}. Keeping original archive in supplemental_files.")
                        # Move unextracted archive to subdirectory
                        shutil.move(temp_filename, os.path.join(supplemental_dir, f"supplemental_archive{file_ext}"))

                elif file_ext.lower() == ".pdf":
                    # If it's a single PDF/other file, just move it to the subdirectory
                    shutil.move(temp_filename, os.path.join(supplemental_dir, f"supplemental{file_ext}"))

                else:
                    print(f"❌ Error for the supplemental files format(either .zip or .pdf are allowed per conference requirements)")
                    shutil.rmtree(base_dir, ignore_errors=True)
                    continue

        with CSV_WRITE_LOCK:
            # --- 3. Collect Final CSV Data (Check 4) ---
            csv_data.append({
                'submission_id': submission_id,
                'directory_name': base_dir,
                'status': 'Desk Rejected' if desk_rejection else "Not Desk Rejected",
                'desk_reject_comments': comment_note.content["desk_reject_comments"] if desk_rejection else "",
                'supplemental_downloaded': supplemental_downloaded
            })


def write_to_csv(csv_data: List[Dict[str, Any]]) -> None:

    CSV_FILENAME = "data/iclr/data/desk_rejected_submissions.csv"
    if csv_data:
        csv_fieldnames = ['submission_id', 'directory_name', 'status', 'desk_reject_comments', 'supplemental_downloaded']

        try:
            with open(CSV_FILENAME, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)
            print(f"\n🎉 Successfully created final analysis CSV: **{CSV_FILENAME}** with {len(csv_data)} records.")
        except Exception as e:
            print(f"\n❌ Error writing final CSV file: {e}")
    else:
        print("\n⚠️ No valid submissions were processed for final CSV output.")