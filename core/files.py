import mimetypes
import os
from typing import List, Union, Optional
from google.genai import types

from core.log import LOG
from core.constants import SKIP_DIRS, STYLE_GUIDES_DEFAULT, SUPPORTED_MIME_TYPES


__STYLE_GUIDES_CACHE: List[types.Part] = []


def get_style_guides_parts() -> List[types.Part]:
    """Get style guides as a list of Parts, using cache if available."""
    global __STYLE_GUIDES_CACHE
    if not __STYLE_GUIDES_CACHE:
        LOG.info("Loading style guides into the prompt")
        for style_guide in STYLE_GUIDES_DEFAULT:
            with open(style_guide, "rb") as f:
                __STYLE_GUIDES_CACHE.append(types.Part.from_bytes(
                    data=f.read(),
                    mime_type=get_optimized_fallback_mime(str(style_guide))
                ))
    return __STYLE_GUIDES_CACHE

def add_supplemental_files(supp_path: str) -> List[str]:
    """
    Recursively gathers supplemental file paths from the specified directory.

    :param supp_path: The root directory to search for supplemental files.
    :return: A list of absolute or relative file paths.
    """
    all_files = []
    for root, _, files in os.walk(supp_path):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files

def process_supplemental_files(supp_path: str, prompt_parts: List[types.Part]) -> None:
    """
    Reads supplemental files from a directory and appends them as Parts to the prompt list.

    :param supp_path: Path to the directory containing supplemental files.
    :param prompt_parts: The list of prompt parts to append files to.
    """
    if os.path.exists(supp_path):
        prompt_parts.append(types.Part.from_text(text="Here are the supplemental files for the paper"))
        supplemental_files = add_supplemental_files(supp_path)
        
        for s_file in supplemental_files:
            s_file_mime = get_optimized_fallback_mime(s_file)
            prompt_parts.append(types.Part.from_text(text=f"The file {s_file}:"))

            with open(s_file, "rb") as f:
                f_read = f.read()
                if len(f_read) <= 0:
                    continue

                if not s_file_mime:
                    file_part = try_decoding(binary_data=f_read)
                    if not file_part:
                        LOG.debug(f"The file '{s_file}' couldn't be uploaded due to unsupported mime_type.")
                        continue
                    prompt_parts.append(file_part)
                else:
                    prompt_parts.append(types.Part.from_bytes(
                        data=f_read,
                        mime_type=s_file_mime
                    ))

def get_optimized_fallback_mime(file_path: str) -> Optional[str]:
    """
    Determines the best supported MIME type for a given file, falling back to safe defaults
    if the exact type is not supported by the Gemini API.

    Rationale:
    - Gemini has a specific list of supported MIME types.
    - For unsupported media, we map to a "best-fit" supported type (e.g., any video -> video/mp4)
      to allow the model to attempt processing.
    - text/plain is used as the ultimate fallback for unknown or varied text formats.

    :param file_path: Path to the file.
    :return: A supported MIME type string.
    """
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
            return None  # Ultimate default for unknown binaries


def try_decoding(binary_data: bytes) -> Optional[types.Part]:
    """
    Attempts to decode binary data as UTF-8 text and return as a Part.

    :param binary_data: The binary data to decode.
    :return: A Part with text content if decoding succeeds, None otherwise.
    """
    try:
        # If it's YAML, Python, or Markdown, this works perfectly.
        text_content = binary_data.decode('utf-8')
        return types.Part.from_text(text=text_content)
    except UnicodeDecodeError:
        return None


def add_supplemental_files(path_to_supplemental_files: Union[os.PathLike, str]) -> List[Union[os.PathLike, str]]:
    """
    Recursively gathers all files from the supplemental files directory.

    Implementation Details:
    - Uses os.walk to traverse the directory tree.
    - Prunes the search tree by modifying 'dirs' in-place to skip hidden directories
      and those listed in SKIP_DIRS (e.g., .venv, __pycache__).
    - Ignores hidden files (starting with '.').

    :param path_to_supplemental_files: Path to the directory containing supplemental materials.
    :return: A list of full file paths.
    """
    supplemental_files_paths = []

    for root, dirs, files in os.walk(f"{path_to_supplemental_files}"):
        # Modifying dirs[:] in-place prunes the search tree
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith('.') and not d.startswith("_")]

        for file in files:
            if not file.startswith("."):
                supplemental_files_paths.append(os.path.join(root, file))

    return supplemental_files_paths
