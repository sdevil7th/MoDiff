"""Safe, backend-owned imports for reusable graph media sources."""

from __future__ import annotations

import hashlib
import http.client
import ipaddress
import mimetypes
import os
import shutil
import socket
import tempfile
import threading
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPHandler, HTTPRedirectHandler, HTTPSHandler, ProxyHandler, Request, build_opener

from modiff.config import CONFIG


_MIME_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "audio/mpeg": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/flac": ".flac",
    "audio/ogg": ".ogg",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "video/quicktime": ".mov",
}
_MEDIA_PREFIXES = ("image/", "audio/", "video/")
_YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be"}
_AUDIO_CONVERSION_LOCKS = tuple(threading.Lock() for _ in range(64))


def _audio_conversion_lock(path: Path):
    return _AUDIO_CONVERSION_LOCKS[hash(str(path)) % len(_AUDIO_CONVERSION_LOCKS)]


def _file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def import_root(root: str | os.PathLike[str] | None = None) -> Path:
    # Config exposes the canonical application data directory as ``data``.
    # ``data_dir`` was never a supported key and broke non-WAV Audio.Load
    # inputs exactly when they needed the managed ffmpeg conversion cache.
    value = Path(root) if root is not None else Path(CONFIG.paths["data"]) / "imports"
    return value.expanduser().resolve()


def _public_address_info(host: str, port: int) -> list[tuple[int, tuple]]:
    try:
        resolved = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError("The media host could not be resolved.") from exc
    if not resolved:
        raise ValueError("The media host could not be resolved.")
    addresses = []
    seen = set()
    for family, _socktype, _proto, _canonname, sockaddr in resolved:
        address = sockaddr[0]
        ip = ipaddress.ip_address(address)
        if not ip.is_global:
            raise ValueError("Local and private network addresses cannot be imported from a graph.")
        key = (family, sockaddr)
        if key not in seen:
            seen.add(key)
            addresses.append(key)
    return addresses


def _public_http_url(value: str, *, youtube_only: bool = False) -> str:
    parsed = urlparse(str(value or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Enter a public HTTP or HTTPS media URL.")
    host = parsed.hostname.rstrip(".").lower()
    if youtube_only and host not in _YOUTUBE_HOSTS:
        raise ValueError("The YouTube source accepts youtube.com and youtu.be links only.")
    _public_address_info(host, parsed.port or (443 if parsed.scheme == "https" else 80))
    return parsed.geturl()


def _open_public_socket(host: str, port: int, timeout: float | object, source_address=None):
    """Resolve, validate, and connect to the exact validated address.

    Connecting to the validated numeric socket address prevents DNS rebinding
    between the SSRF check and the actual network request.
    """

    last_error = None
    for family, sockaddr in _public_address_info(host, port):
        sock = socket.socket(family, socket.SOCK_STREAM)
        try:
            if timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:
                sock.settimeout(timeout)
            if source_address:
                sock.bind(source_address)
            sock.connect(sockaddr)
            return sock
        except OSError as exc:
            last_error = exc
            sock.close()
    if last_error is not None:
        raise last_error
    raise OSError("The media host has no usable public address.")


class _PinnedHTTPConnection(http.client.HTTPConnection):
    def connect(self):
        self.sock = _open_public_socket(self.host, self.port, self.timeout, self.source_address)
        if self._tunnel_host:
            self._tunnel()


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def connect(self):
        self.sock = _open_public_socket(self.host, self.port, self.timeout, self.source_address)
        server_hostname = self.host
        if self._tunnel_host:
            self._tunnel()
            server_hostname = self._tunnel_host
        self.sock = self._context.wrap_socket(self.sock, server_hostname=server_hostname)


class _PinnedHTTPHandler(HTTPHandler):
    def http_open(self, req):
        return self.do_open(_PinnedHTTPConnection, req)


class _PinnedHTTPSHandler(HTTPSHandler):
    def https_open(self, req):
        return self.do_open(
            _PinnedHTTPSConnection,
            req,
            context=self._context,
            check_hostname=self._check_hostname,
        )


class _SafeRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return super().redirect_request(req, fp, code, msg, headers, _public_http_url(newurl))


def import_web_media(
    url: str,
    *,
    max_bytes: int = 256 * 1024 * 1024,
    timeout_seconds: float = 30,
    root: str | os.PathLike[str] | None = None,
) -> Path:
    """Download one public media response into the app cache with hard limits."""

    safe_url = _public_http_url(url)
    request = Request(safe_url, headers={"User-Agent": "MoDiff/1.0 media import"})
    # Do not delegate these graph-controlled URLs to an environment proxy: a
    # proxy can resolve the hostname differently and bypass the local address
    # validation. Each direct connection is pinned to its validated address.
    opener = build_opener(ProxyHandler({}), _PinnedHTTPHandler(), _PinnedHTTPSHandler(), _SafeRedirects())
    destination_root = import_root(root) / "web"
    destination_root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    temporary = None
    try:
        with opener.open(request, timeout=float(timeout_seconds)) as response:
            content_type = str(response.headers.get_content_type() or "").lower()
            if not content_type.startswith(_MEDIA_PREFIXES):
                raise ValueError("The URL did not return an image, audio file, or video file.")
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > max_bytes:
                raise ValueError(f"The media file exceeds the {max_bytes // (1024 * 1024)} MB import limit.")
            with tempfile.NamedTemporaryFile(dir=destination_root, prefix=".download-", delete=False) as handle:
                temporary = Path(handle.name)
                total = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError(f"The media file exceeds the {max_bytes // (1024 * 1024)} MB import limit.")
                    digest.update(chunk)
                    handle.write(chunk)
            suffix = _MIME_EXTENSIONS.get(content_type)
            if not suffix:
                suffix = Path(urlparse(response.geturl()).path).suffix.lower() or mimetypes.guess_extension(
                    content_type
                )
            if not suffix or len(suffix) > 8:
                suffix = ".bin"
            destination = destination_root / f"{digest.hexdigest()}{suffix}"
            if destination.exists():
                temporary.unlink(missing_ok=True)
            else:
                temporary.replace(destination)
            return destination
    except (HTTPError, URLError, TimeoutError) as exc:
        raise ValueError(f"The media URL could not be downloaded: {exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def import_youtube_media(
    url: str,
    *,
    media_kind: str = "video",
    max_duration_seconds: int = 1800,
    max_bytes: int = 512 * 1024 * 1024,
    root: str | os.PathLike[str] | None = None,
) -> Path:
    """Download one user-authorized YouTube item as video or WAV audio."""

    safe_url = _public_http_url(url, youtube_only=True)
    if media_kind not in {"video", "audio"}:
        raise ValueError("YouTube media kind must be video or audio.")
    try:
        from yt_dlp import YoutubeDL
    except ImportError as exc:
        raise RuntimeError("YouTube import needs the yt-dlp dependency.") from exc

    destination_root = import_root(root) / "youtube"
    destination_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=".youtube-", dir=destination_root))
    options = {
        "noplaylist": True,
        "playlist_items": "1",
        "paths": {"home": str(work)},
        "outtmpl": {"default": "%(id)s.%(ext)s"},
        "max_filesize": int(max_bytes),
        "match_filter": lambda info, *, incomplete=False: (
            f"Video exceeds the {max_duration_seconds}-second import limit"
            if not incomplete and float(info.get("duration") or 0) > max_duration_seconds
            else None
        ),
        "quiet": True,
        "no_warnings": True,
        "overwrites": False,
    }
    if media_kind == "audio":
        options.update(
            {
                "format": "bestaudio/best",
                "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
            }
        )
    else:
        options.update(
            {
                "format": "bv*[height<=1080]+ba/b[height<=1080]/b",
                "merge_output_format": "mp4",
            }
        )
    try:
        with YoutubeDL(options) as downloader:
            info = downloader.extract_info(safe_url, download=True)
        identifier = str((info or {}).get("id") or "youtube")
        candidates = [path for path in work.iterdir() if path.is_file() and not path.name.endswith(".part")]
        if not candidates:
            raise RuntimeError("YouTube import completed without a media file.")
        source = max(candidates, key=lambda path: path.stat().st_size)
        if source.stat().st_size > max_bytes:
            raise ValueError(f"The downloaded media exceeds the {max_bytes // (1024 * 1024)} MB import limit.")
        digest = _file_sha256(source)
        suffix = ".wav" if media_kind == "audio" else source.suffix.lower()
        destination = destination_root / f"{identifier}-{digest[:12]}{suffix}"
        if destination.exists():
            return destination
        shutil.move(str(source), destination)
        return destination
    finally:
        shutil.rmtree(work, ignore_errors=True)


def audio_as_wav(
    path: str | os.PathLike[str],
    *,
    sample_rate: int | None = None,
    channels: int | None = None,
    root: str | os.PathLike[str] | None = None,
) -> Path:
    """Return a model-ready WAV without altering the imported original.

    ``sample_rate`` and ``channels`` are intentionally caller-owned.  A model
    adapter should request the rate and channel layout its processor expects;
    the generic loader preserves the source representation.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {source}")
    target_sample_rate = int(sample_rate) if sample_rate is not None else None
    target_channels = int(channels) if channels is not None else None
    if target_sample_rate is not None and not 8000 <= target_sample_rate <= 384000:
        raise ValueError("Audio sample rate must be between 8 kHz and 384 kHz.")
    if target_channels is not None and not 1 <= target_channels <= 8:
        raise ValueError("Audio channel count must be between 1 and 8.")
    if source.suffix.lower() == ".wav" and target_sample_rate is None and target_channels is None:
        return source
    import subprocess

    from imageio_ffmpeg import get_ffmpeg_exe

    digest = _file_sha256(source)
    destination_root = import_root(root) / "audio"
    destination_root.mkdir(parents=True, exist_ok=True)
    conversion = f"{target_sample_rate or 'source'}-{target_channels or 'source'}"
    destination = destination_root / f"{digest}-{conversion}.wav"
    if destination.exists():
        return destination
    with _audio_conversion_lock(destination):
        if destination.exists():
            return destination
        with tempfile.NamedTemporaryFile(
            dir=destination_root,
            prefix=f".{destination.stem}-",
            suffix=".wav",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        try:
            command = [get_ffmpeg_exe(), "-y", "-v", "error", "-i", str(source), "-vn"]
            if target_sample_rate is not None:
                command.extend(["-ar", str(target_sample_rate)])
            if target_channels is not None:
                command.extend(["-ac", str(target_channels)])
            command.extend(["-c:a", "pcm_s16le", str(temporary)])
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode or not temporary.is_file():
                detail = (result.stderr or result.stdout or "unsupported audio format").strip()
                raise ValueError(f"The selected audio could not be converted to WAV: {detail}")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return destination
