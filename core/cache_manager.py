import base64
from typing import List, Optional, Dict, Any
from pathlib import Path
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
            
_STYLE_GUIDE_CACHE = None

def get_style_guide_content() -> List[Dict[str, Any]]:
    """
    Loads and caches the style guide content to be used as a prefix in OpenAI requests.
    This leverages OpenAI's automatic prompt caching.
    """
    global _STYLE_GUIDE_CACHE
    if _STYLE_GUIDE_CACHE is not None:
        return _STYLE_GUIDE_CACHE

    LOG.debug("--- Loading Style Guides for OpenAI Prompt Caching ---")
    content = []
    for path in STYLE_GUIDES_DEFAULT:
        try:
            mime = get_optimized_fallback_mime(str(path))
            with open(path, "rb") as f:
                file_data = base64.b64encode(f.read()).decode("utf-8")

            content.append({
                "type": "text",
                "text": f"The file named {path.split('/')[-1]} is one of the style guide files which will be referred to in one of the current agents"
            })# the explanational part

            content.append({
                "type": "file",
                "source_type": "base64",
                "mime_type": mime,
                "data": file_data,
            }) # and then the content part
            LOG.debug(f"Loaded style guide: {path} as {mime}")
        except Exception as e:
            LOG.error(f"Failed to load style guide {path}: {e}")
    
    _STYLE_GUIDE_CACHE = content
    return _STYLE_GUIDE_CACHE
