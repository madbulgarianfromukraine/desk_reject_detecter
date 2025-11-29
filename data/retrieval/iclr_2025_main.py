import os
from concurrent.futures import ThreadPoolExecutor

import openreview.api
from dotenv import load_dotenv

from data.retrieval.helper import filter_proper_desk_rejections
from data.retrieval.storage import store_main_and_supplemental_materials, write_to_csv


load_dotenv(verbose=True)
USER_NAME = os.getenv("USER_NAME")
PASSWORD = os.getenv("USER_PASSWORD")
BASE_URL = "https://api2.openreview.net"
CONFERENCE_ID = 'ICLR.cc/2025/Conference'
# --- Global Data Structure for CSV Output ---
final_csv_data = []

def main_rejection(client: openreview.api.OpenReviewClient) -> None:
    DESK_REJECT_INVITATION = f'{CONFERENCE_ID}/-/Desk_Rejected_Submission'

    # 1. Get all Notes posted using the correct master list invitation
    initial_desk_rejections = client.get_all_notes(
        invitation=DESK_REJECT_INVITATION,
        details='content'
    )

    print(f"\n--- Processing initial desk rejects ---")
    submissions_to_process = filter_proper_desk_rejections(client=client, initial_desk_rejections=initial_desk_rejections)

    print(f"\n--- Starting Download and Processing for {len(submissions_to_process)} Valid Submissions ---")

    store_main_and_supplemental_materials(client=client, submissions_to_process=submissions_to_process, csv_data=final_csv_data)



def main_accepted(client: openreview.api.OpenReviewClient) -> None:
    pass

if __name__ == "__main__":
    client = openreview.api.OpenReviewClient(
        baseurl=BASE_URL,
        username=USER_NAME,
        password=PASSWORD
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        # we execute a functions in a concurrent way for two reasons:
        # 1. it is faster(download takes incredibly long)
        # 2. it will randomize the entries of the papers
        executor.submit(main_rejection, client)
        executor.submit(main_accepted, client)


    write_to_csv(csv_data=final_csv_data)


