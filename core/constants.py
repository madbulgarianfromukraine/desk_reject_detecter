from pathlib import Path

__REQUIREMENTS_DIR = 'data/iclr/requirements/'
STYLE_GUIDES_DEFAULT = [f for f in Path(__REQUIREMENTS_DIR).iterdir() if f.is_file()]

# Exhaustive list of standard supported types for OpenAI GPT-4o
SUPPORTED_MIME_TYPES = [
    # Documents & Text
    'application/pdf', 'text/plain', 'text/markdown', 'text/html',
    'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'application/vnd.openxmlformats-officedocument.presentationml.presentation',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'text/csv', 'application/json', 'application/xml', 'application/x-sh',

    # Images (Vision)
    'image/png', 'image/jpeg', 'image/webp', 'image/gif',
    'image/bmp', 'image/heic', 'image/heif', 'image/tiff',

    # Audio (Speech & Realtime)
    'audio/mpeg', 'audio/wav', 'audio/ogg', 'audio/aac',
    'audio/flac', 'audio/webm', 'audio/mp4', 'audio/opus',

    # Programming & Technical (File Search / Code Interpreter)
    'text/x-c', 'text/x-c++', 'text/x-csharp', 'text/x-java',
    'text/x-python', 'text/javascript', 'application/typescript',
    'text/x-php', 'text/x-ruby', 'text/x-swift', 'text/x-go',
    'text/x-tex', 'text/css', 'application/x-tar', 'application/zip'
]

#Skip dirs for efficient loading of supplemental files
SKIP_DIRS = {'.venv', 'CVS', '.git', '__pycache__', '.pytest_cache'}