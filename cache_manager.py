import time
from typing import List, Optional
from pathlib import Path
import google.genai as genai
from concurrent.futures import ThreadPoolExecutor
from utils import STYLE_GUIDES_DEFAULT, SUPPORTED_MIME_TYPES
import mimetypes


def get_optimized_fallback_mime(file_path: str) -> str:
    mime, _ = mimetypes.guess_type(file_path)

    if mime in SUPPORTED_MIME_TYPES:
        return mime

    # 2. Structural pattern matching for closest-category fallbacks
    match mime.split('/') if mime else []:
        case ['video', _]:
            return 'video/mp4'  # Best fallback for all unsupported video
        case ['audio', _]:
            return 'audio/mpeg'  # Best fallback for all unsupported audio
        case ['image', _]:
            return 'image/jpeg'  # Best fallback for all unsupported images
        case ['text', _]:
            return 'text/plain'  # Catch-all for varied text (logs, csv, etc.)
        case _:
            return 'text/plain'  # Ultimate default for unknown binaries

def upload_single_file(client: genai.Client, file_path: Optional[Path | str] = None) -> Optional[str]:
    try:
        file = client.files.upload(file=file_path,
                                   config=genai.types.UploadFileConfig(mime_type=get_optimized_fallback_mime(file_path)))
        while file.state.name == "PROCESSING":
            print(f"Waiting for {file.display_name} to process...")
            time.sleep(2)
            file = client.files.get(name=file.name)
    except Exception as e:
        print(f"Failed to upload the file {file_path} due to {e}")
        return None

    return file.name

def create_paper_cache(
        pdf_path: str,
        style_guide_paths: List[str] = STYLE_GUIDES_DEFAULT,
        supplemental_paths: List[str] = [],
        model_name: str = "gemini-2.5-flash-lite"
):
    print("--- Uploading Files to Gemini Cache ---")
    client = genai.Client()
    # 1. Upload Main Paper
    print(f"Uploading main paper: {pdf_path}...")
    paper_file = upload_single_file(client, file_path=pdf_path)
    if paper_file is None:
        raise Exception("Have not been able to upload a main.pdf")

    print(f'Uploading supplemental paths: {supplemental_paths}...')
    supplemental_paths_uploaded = []
    with ThreadPoolExecutor(max_workers=10, thread_name_prefix="SupplementalPathThread") as executor:
        supplemental_paths_uploaded = list(executor.map(upload_single_file, [(client, supplemental_path) for supplemental_path in supplemental_paths]))

    if len(supplemental_paths_uploaded) != len(supplemental_paths):
        print("Have not been able to upload some supplemental files")

    print(f'Uploading style guide paths: {style_guide_paths}...')
    style_guide_paths_uploaded = []
    with ThreadPoolExecutor(max_workers=4, thread_name_prefix="StyleGuidePathThread") as executor:
        style_guide_paths_uploaded = list(executor.map(upload_single_file, [(client, supplemental_path) for supplemental_path in supplemental_paths]))

    if len(style_guide_paths_uploaded) != len(style_guide_paths):
        print("Have not been able to upload some style guide files")
    print("Creating Context Cache...")

    contents = [
        genai.types.Content(
            role="user",
            parts=[
                genai.types.Part.from_uri(file_uri=document1.uri, mime_type=document1.mime_type),
                genai.types.Part.from_uri(file_uri=document2.uri, mime_type=document2.mime_type),
            ]
        ),
    ]
    cache = client.caches.create(
        model=model_name,
        config=genai.types.CreateCachedContentConfig(
            contents=contents,
            system_instruction="You are an expert analyzing transcripts.",
        ),
    )
    print(f"Cache Created! Name: {cache.name}")
    cache = client.caches.create
    return cache.name