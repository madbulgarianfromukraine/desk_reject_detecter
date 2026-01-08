from pathlib import Path

__REQUIREMENTS_DIR = 'data/iclr/requirements/'
STYLE_GUIDES_DEFAULT = [f for f in Path(__REQUIREMENTS_DIR).iterdir() if f.is_file()]

# Exhaustive list of standard supported types for Gemini 2.5
SUPPORTED_MIME_TYPES = {
    'application/pdf', 'text/plain',
    'image/png', 'image/jpeg', 'image/webp', 'image/heic', 'image/heif',
    'video/mp4', 'video/mpeg', 'video/mov', 'video/avi', 'video/x-flv',
    'video/mpg', 'video/webm', 'video/wmv', 'video/3gpp',
    'audio/wav', 'audio/mp3', 'audio/aiff', 'audio/aac', 'audio/ogg',
    'audio/flac', 'audio/m4a', 'audio/mpga', 'audio/pcm'
}
#Skip dirs for efficient loading of supplemental files
SKIP_DIRS = {'.venv', 'CVS', '.git', '__pycache__', '.pytest_cache'}