import openreview.api
import csv
import os
import shutil
import urllib.parse
import zipfile
from typing import Optional, List, Dict, Any
from openreview import OpenReviewException
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

from helper import get_note_value
from util.attachments import get_attachment


def download_file(client: openreview.api.OpenReviewClient,
                  note_id: Optional[str],
                  field_name: Optional[str] = "",
                  output_path: str = "main.pdf",
                  is_pdf: bool = False,
                  timeout: float = 30.,
                  is_reference: bool = False) -> bool:
    """Download a single file using the already-instantiated OpenReview client with a hard timeout.

    We execute the OpenReview API call in a worker thread and wait up to `timeout` seconds.
    On timeout or error, we skip and return False. Only after a successful fetch do we write to disk.
    """

    if not note_id:
        print("    ❌ Missing note_id for download")
        return False

    def _fetch() -> bytes:
        if is_pdf:
            if is_reference:
                return get_attachment(client=client, id=[note_id], field_name=field_name)
            else:
                return client.get_pdf(note_id)
        else:
            if not field_name:
                raise ValueError("field_name is required when downloading an attachment")
            if is_reference:
                return get_attachment(client=client,id=[note_id], field_name=field_name)
            else:
                return client.get_attachment(id=[note_id], field_name=field_name)

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"download_{note_id}") as executor:
        future = executor.submit(_fetch)
        try:
            data: bytes = future.result(timeout=timeout)
        except FuturesTimeout:
            print(f"    ⏱️ Download Timed Out after {timeout}s for {field_name or 'pdf'} of note {note_id}. Skipping.")
            return False
        except OpenReviewException as e:
            print(f"    ❌ OpenReview error during download of {field_name or 'pdf'} for note {note_id}: {e}")
            return False
        except Exception as e:
            print(f"    ❌ Unexpected error during download of {field_name or 'pdf'} for note {note_id}: {e}")
            return False

    # Ensure parent directory exists and write file only after successful fetch
    try:
        with open(output_path, "wb") as f:
            f.write(data)
    except Exception as e:
        print(f"    ❌ Failed to write output file '{output_path}': {e}")
        return False

    return True


def process_single_submission(client: openreview.api.OpenReviewClient, item: Dict[str, Any], csv_data: List[Dict[str, Any]],
                              desk_rejection: bool = False, timeout: float = 30.0, is_reference: bool = False) -> None:
    """Process a single submission using the latest revision/edit."""

    # item['submission'] here is expected to be the 'Note' from the last edit
    # as per your previous function logic.
    submission = item['submission']
    comment_note = item['comment_note']
    material_id = submission.id

    # --- 1. Identify if we are dealing with an Edit/Revision ---
    # We fetch edits to find the specific 'Edit ID' needed for 'is_reference' downloads
    if is_reference:
        note_edits = client.get_note_edits(note_id=material_id, sort='tmdate:desc')
        if not note_edits:
            print(f"No edits found for {material_id}. Proceeding with download of the pdf.")
            is_reference=False
        else:
            # The latest edit ID is what we use as the 'id' parameter in get_pdf/get_attachment
            # when is_reference is True.
            material_id = note_edits[-1].id
            print(f"Submission_id is {submission.id} is changed for {material_id}")

    title = get_note_value(submission, 'title')
    dir_name = f"submission_{material_id}"
    base_dir = f"data/iclr/data/{dir_name}"

    os.makedirs(base_dir, exist_ok=True)

    # --- 2. Download Main PDF (Using Edit ID as reference) ---
    pdf_filename = os.path.join(base_dir, f"main_paper.pdf")

    # Note: We pass last_edit_id instead of submission_id because is_reference=True
    # requires the ID of the Edit/Revision record.
    print(f'Attempting to download main_paper.pdf for submission={submission.id}')
    if not download_file(client, material_id, 'pdf', pdf_filename, is_pdf=True,
                         timeout=timeout, is_reference=is_reference):
        shutil.rmtree(base_dir, ignore_errors=True)
        return None

    # --- 3. Download Supplementary Material ---
    supplementary_material_path = get_note_value(submission, 'supplementary_material')
    supplemental_downloaded = False

    print(f'Attempting to download supplemental files for submission={submission.id}')
    if supplementary_material_path:
        parsed_path = urllib.parse.urlparse(supplementary_material_path).path
        _, file_ext = os.path.splitext(os.path.basename(parsed_path))
        temp_filename = os.path.join(base_dir, f"supplemental_archive{file_ext}")

        # Download supplemental using the Edit ID reference
        supplemental_download_successful = download_file(
            client,
            material_id,  # Reference the specific edit
            'supplementary_material',
            temp_filename,
            is_pdf=False,
            timeout=timeout,
            is_reference=is_reference
        )

        if not supplemental_download_successful:
            print(f"⚠️ Supplemental download failed for {submission.id} at revision {material_id}")
            shutil.rmtree(base_dir, ignore_errors=True)
            return None
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

    # --- 4. Record CSV Data ---
    csv_data.append({
        'material_id': material_id,
        'directory_name': base_dir,
        'status': 'Desk Rejected' if desk_rejection else "Not Desk Rejected",
        'desk_reject_comments': comment_note.content["desk_reject_comments"]["value"] if desk_rejection else "",
        'supplemental_downloaded': supplemental_downloaded
    })


def store_main_and_supplemental_materials(client: openreview.api.OpenReviewClient,submissions_to_process: List[Dict[str, Any]], csv_data: List[Dict[str, Any]], desk_rejection: bool = False, timeout: float = 30.0) -> None:
    """Backward-compatible wrapper that processes a list by delegating to process_single_submission.

    Prefer calling process_single_submission in the caller's loop to reduce long-held GIL segments.
    """
    for item in submissions_to_process:
        process_single_submission(client=client, item=item, csv_data=csv_data, desk_rejection=desk_rejection, timeout=timeout)


def write_to_csv(csv_data: List[Dict[str, Any]]) -> None:

    CSV_FILENAME = "data/iclr/data/submissions.csv"
    if csv_data:
        csv_fieldnames = ['material_id', 'directory_name', 'status', 'desk_reject_comments', 'supplemental_downloaded']

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