from modiff.NodeBase import NodeBase
from modiff.media_import import import_web_media, import_youtube_media
from modiff.path_identifiers import resolve_runtime_input_path


class LocalMedia(NodeBase):
    """Select a backend-managed image, audio file, or video file."""

    label = "Local Media"
    category = "Media sources"
    params = {
        "file": {
            "label": "Media file",
            "display": "filebrowser",
            "type": "str",
            "fieldOptions": {"fileTypes": ["image", "audio", "video"], "multiple": False},
        },
        "path": {"label": "Local path", "display": "output", "type": "str"},
    }

    def execute(self, **kwargs):
        value = kwargs.get("file")
        value = value[0] if isinstance(value, list) and value else value
        path = resolve_runtime_input_path(str(value or "")).resolve()
        if not path.is_file():
            raise ValueError("Local Media needs an existing image, audio file, or video file.")
        return {"path": str(path)}


class WebMedia(NodeBase):
    """Download one public HTTP(S) media file into the backend import cache."""

    label = "Web Media"
    category = "Media sources"
    params = {
        "url": {"label": "Media URL", "type": "str", "default": "https://"},
        "max_size_mb": {"label": "Maximum size", "type": "int", "default": 256, "min": 1, "max": 2048},
        "path": {"label": "Cached path", "display": "output", "type": "str"},
    }

    def execute(self, **kwargs):
        limit = int(kwargs.get("max_size_mb") or 256) * 1024 * 1024
        return {"path": str(import_web_media(kwargs.get("url"), max_bytes=limit))}


class YouTubeMedia(NodeBase):
    """Download one user-authorized YouTube item into the backend import cache."""

    label = "YouTube Media"
    category = "Media sources"
    params = {
        "url": {"label": "YouTube URL", "type": "str", "default": "https://www.youtube.com/watch?v="},
        "media_kind": {"label": "Import as", "type": "string", "options": ["video", "audio"], "default": "video"},
        "max_duration_seconds": {"label": "Maximum duration", "type": "int", "default": 1800, "min": 1, "max": 21600},
        "max_size_mb": {"label": "Maximum size", "type": "int", "default": 512, "min": 1, "max": 4096},
        "rights_confirmed": {
            "label": "I have permission to use this media",
            "type": "bool",
            "default": False,
        },
        "path": {"label": "Cached path", "display": "output", "type": "str"},
    }

    def execute(self, **kwargs):
        if not kwargs.get("rights_confirmed"):
            raise ValueError("Confirm that you have permission to download and use this media.")
        path = import_youtube_media(
            kwargs.get("url"),
            media_kind=str(kwargs.get("media_kind") or "video"),
            max_duration_seconds=int(kwargs.get("max_duration_seconds") or 1800),
            max_bytes=int(kwargs.get("max_size_mb") or 512) * 1024 * 1024,
        )
        return {"path": str(path)}
