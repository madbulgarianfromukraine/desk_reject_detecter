import time
from typing import List, Optional
from pathlib import Path
import google.genai as genai
from concurrent.futures import ThreadPoolExecutor
from core.constants import STYLE_GUIDES_DEFAULT, SUPPORTED_MIME_TYPES
from core.log import LOG
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

def upload_single_file(client: genai.Client, file_path: Optional[Path | str] = None) -> Optional[genai.types.File]:
    try:
        file = client.files.upload(file=file_path,
                                   config=genai.types.UploadFileConfig(mime_type=get_optimized_fallback_mime(file_path)))
        while file.state.name == "PROCESSING":
            LOG.debug(f"Waiting for {file.display_name} to process...")
            time.sleep(2)
            file = client.files.get(name=file.name)
    except Exception as e:
        LOG.debug(f"Failed to upload the file {file_path} due to {e}")
        return None

    return file

def create_paper_cache(style_guide_paths: List[str] = STYLE_GUIDES_DEFAULT, model_name: str = "gemini-2.5-flash-lite"):
    LOG.debug("--- Uploading Files to Gemini Cache ---")
    client = genai.Client()

    LOG.debug(f'Uploading style guide paths: {style_guide_paths}...')
    style_guide_paths_uploaded = []
    with ThreadPoolExecutor(max_workers=4, thread_name_prefix="StyleGuidePathThread") as executor:
        style_guide_paths_uploaded = list(executor.map(upload_single_file, [(client, supplemental_path) for supplemental_path in style_guide_paths]))

    if len(style_guide_paths_uploaded) != len(style_guide_paths):
        LOG.warning("Have not been able to upload some style guide files")
    LOG.info("Creating Context Cache...")

    contents = [
        genai.types.Content(
            role="user",
            parts=[
                genai.types.Part.from_uri(file_uri=document.uri, mime_type=get_optimized_fallback_mime(document.uri)) for document in style_guide_paths_uploaded
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
    LOG.info(f"Cache Created! Name: {cache.name}")
    cache = client.caches.create
    return cache.name