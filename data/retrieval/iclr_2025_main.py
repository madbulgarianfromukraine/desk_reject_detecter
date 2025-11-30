import os
from threading import Barrier
from concurrent.futures import ThreadPoolExecutor

import openreview.api
from dotenv import load_dotenv

from data.retrieval.helper import filter_proper_desk_rejections, filter_proper_accepted_papers
from data.retrieval.storage import process_single_submission, write_to_csv


load_dotenv(verbose=True)
USER_NAME = os.getenv("USER_NAME")
PASSWORD = os.getenv("USER_PASSWORD")
BASE_URL = "https://api2.openreview.net"
CONFERENCE_ID = 'ICLR.cc/2025/Conference'

DESK_REJECTION_IDS = []
WITHDRAWAL_IDS = []
ACCEPTED_RETREVING_BARRIER = Barrier(3) # 2 for rejection and withdrawal and 1 for accepted
final_csv_data = [] # --- Global Data Structure for CSV Output ---

def main_rejection(client: openreview.api.OpenReviewClient) -> None:
    global DESK_REJECTION_IDS, ACCEPTED_RETREVING_BARRIER
    DESK_REJECT_INVITATION = f'{CONFERENCE_ID}/-/Desk_Rejected_Submission'

    # 1. Get all Notes posted using the correct master list invitation
    initial_desk_rejections = client.get_all_notes(
        invitation=DESK_REJECT_INVITATION,
        details='content'
    )
    DESK_REJECTION_IDS = [submission.forum for submission in initial_desk_rejections]
    ACCEPTED_RETREVING_BARRIER.wait()
    print(f"\n--- Processing initial desk rejects ---")
    submissions_to_process = filter_proper_desk_rejections(client=client, initial_desk_rejections=initial_desk_rejections)

    print(f"\n--- Starting Download and Processing for {len(submissions_to_process)} Valid DR-Submissions ---")

    # Process each submission individually to avoid holding the GIL for long loops
    for item in submissions_to_process:
        process_single_submission(client=client, item=item, csv_data=final_csv_data, desk_rejection=True)



def main_accepted(client: openreview.api.OpenReviewClient) -> None:
    global DESK_REJECTION_IDS, WITHDRAWAL_IDS
    ACCEPTED_INVITATION = f'{CONFERENCE_ID}/-/Submission'

    initial_accepted_papers = []
    initial_accepted_papers += client.get_all_notes(
        invitation=ACCEPTED_INVITATION,
        details='content'
    )

    ACCEPTED_RETREVING_BARRIER.wait()
    print(f"\n--- Processing initially not desk rejects ---")
    submissions_to_process = filter_proper_accepted_papers(
        client=client,
        initial_accepted_papers=initial_accepted_papers,
        dr_submissions_count=len(DESK_REJECTION_IDS),
        desk_rejection_ids=DESK_REJECTION_IDS,
        withdrawal_ids=WITHDRAWAL_IDS,
    )

    print(f"\n--- Starting Download and Processing for {len(submissions_to_process)} Valid NDR-Submissions ---")

    # Process each submission individually to avoid holding the GIL for long loops
    for item in submissions_to_process:
        process_single_submission(client=client, item=item, csv_data=final_csv_data, desk_rejection=False)

def main_withdrawal(client: openreview.api.OpenReviewClient) -> None:
    global WITHDRAWAL_IDS, ACCEPTED_RETREVING_BARRIER
    WITHDRAWN_INVITATION = f'{CONFERENCE_ID}/-/Withdrawn_Submission'
    withdrawals = client.get_all_notes(
        invitation=WITHDRAWN_INVITATION,
        details='id'
    )

    WITHDRAWAL_IDS += [withdrawal.id for withdrawal in withdrawals]
    ACCEPTED_RETREVING_BARRIER.wait()

if __name__ == "__main__":
    client = openreview.api.OpenReviewClient(
        baseurl=BASE_URL,
        username=USER_NAME,
        password=PASSWORD
    )

    with ThreadPoolExecutor(max_workers=3) as executor:
        # we execute a functions in a concurrent way for two reasons:
        # 1. it is faster(download takes incredibly long)
        # 2. it will randomize the entries of the papers
        executor.submit(main_rejection, client)
        executor.submit(main_withdrawal, client)
        executor.submit(main_accepted, client)


    write_to_csv(csv_data=final_csv_data)


