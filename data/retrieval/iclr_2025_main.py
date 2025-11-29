import os
from threading import Barrier
from concurrent.futures import ThreadPoolExecutor

import openreview.api
from dotenv import load_dotenv

from data.retrieval.helper import filter_proper_desk_rejections, filter_proper_accepted_papers
from data.retrieval.storage import store_main_and_supplemental_materials, write_to_csv


load_dotenv(verbose=True)
USER_NAME = os.getenv("USER_NAME")
PASSWORD = os.getenv("USER_PASSWORD")
BASE_URL = "https://api2.openreview.net"
CONFERENCE_ID = 'ICLR.cc/2025/Conference'

DESK_REJECTION_IDS = []
WITHDRAWAL_IDS = []
ACCEPTED_RETREVING_BARRIER = Barrier(3) # 2 to wait for desk-rejected and withdrawn submissions and 1 to wait for the accepted, so that it can start
# --- Global Data Structure for CSV Output ---
final_csv_data = []

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

    print(f"\n--- Starting Download and Processing for {len(submissions_to_process)} Valid Submissions ---")

    store_main_and_supplemental_materials(client=client, submissions_to_process=submissions_to_process, csv_data=final_csv_data, desk_rejection=True)



def main_accepted(client: openreview.api.OpenReviewClient) -> None:
    global ACCEPTED_RETREVING_BARRIER
    ACCEPTED_INVITATION = f'{CONFERENCE_ID}/-/Submission'

    initial_accepted_papers = []
    initial_accepted_papers += client.get_all_notes(
        invitation=ACCEPTED_INVITATION,
        details='content'
    )

    ACCEPTED_RETREVING_BARRIER.wait()

    print(f"\n--- Processing initially not desk rejects ---")
    submissions_to_process = filter_proper_accepted_papers(client=client, initial_accepted_papers=initial_accepted_papers)

    print(f"\n--- Starting Download and Processing for {len(submissions_to_process)} Valid Submissions ---")

    store_main_and_supplemental_materials(client=client, submissions_to_process=submissions_to_process, csv_data=final_csv_data, desk_rejection=False)

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

    with ThreadPoolExecutor(max_workers=2) as executor:
        # we execute a functions in a concurrent way for two reasons:
        # 1. it is faster(download takes incredibly long)
        # 2. it will randomize the entries of the papers
        #executor.submit(main_rejection, client)
        executor.submit(main_accepted, client)


    write_to_csv(csv_data=final_csv_data)


