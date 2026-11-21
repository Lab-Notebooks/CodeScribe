# Copyright (c) 2026 UChicago Argonne LLC
# SPDX-License-Identifier: Apache-2.0
# Full license and notices: see LICENSE and NOTICE in the repo root.

"""Media attachment: route files to provider-native content blocks.

The agent discovers media via read/glob. ReadTool returns a MEDIA_ATTACH
marker for binary media instead of mangled text, and the agent harness
(Agent.handle_tool_calls) turns the marker into content blocks on the next
turn via file_parts(): images and PDFs inline as base64, videos as sampled
frames (no chat API accepts video as a native input block).

Files are inlined as base64, which inflates the request body ~4/3x and trips
gateway body-size limits (nginx `client_max_body_size` -> HTTP 413). Oversized
images are downscaled before attaching, oversized PDFs are rejected with a
clear error, and the agent harness strips earlier attachments from the
message history so only the newest media is sent inline each turn.
"""

from __future__ import annotations

import base64
import mimetypes
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

__all__ = [
    "MEDIA_ATTACH_PREFIX",
    "MAX_INLINE_MEDIA_BYTES",
    "guess_media_mime",
    "media_attach_marker",
    "parse_media_marker",
    "is_inline_media_block",
    "strip_inline_media",
    "file_parts",
]

# ReadTool returns f"{MEDIA_ATTACH_PREFIX}{path}" for media files; the agent
# harness converts the marker into real content blocks on the next turn.
MEDIA_ATTACH_PREFIX = "MEDIA_ATTACH:"

# Max raw bytes of a file inlined as base64 into one request. Base64 inflates
# ~4/3x, so 4 MB raw -> ~5.3 MB in the body, under common 10 MB gateway caps.
# Override with CODESCRIBE_MAX_INLINE_MEDIA_BYTES.
MAX_INLINE_MEDIA_BYTES = int(
    os.getenv("CODESCRIBE_MAX_INLINE_MEDIA_BYTES", str(4_000_000))
)

# MIME prefixes/types the chat APIs accept as native input blocks.
_MEDIA_MIME_PREFIXES = ("image/", "video/")
_MEDIA_MIME_TYPES = {"application/pdf"}


def guess_media_mime(path: Union[str, Path]) -> Optional[str]:
    """MIME type of `path` if it is media an LLM can consume, else None."""
    mime = mimetypes.guess_type(str(path))[0] or ""
    if mime.startswith(_MEDIA_MIME_PREFIXES) or mime in _MEDIA_MIME_TYPES:
        return mime
    return None


def media_attach_marker(path: Union[str, Path]) -> str:
    return f"{MEDIA_ATTACH_PREFIX}{path}"


def parse_media_marker(output: str) -> Optional[str]:
    """The attached path if `output` is a media-attach marker, else None."""
    output = (output or "").strip()
    if output.startswith(MEDIA_ATTACH_PREFIX):
        return output[len(MEDIA_ATTACH_PREFIX) :]
    return None


def is_inline_media_block(block: Any) -> bool:
    """True if `block` is a content block carrying inline base64 media."""
    if not isinstance(block, dict):
        return False
    btype = block.get("type")
    if btype == "image_url":
        url = (block.get("image_url") or {}).get("url", "")
        return isinstance(url, str) and url.startswith("data:")
    if btype in ("image", "document"):
        return (block.get("source") or {}).get("type") == "base64"
    if btype == "input_file":
        file_data = block.get("file_data") or ""
        return isinstance(file_data, str) and file_data.startswith("data:")
    return False


def strip_inline_media(messages: List[Dict[str, Any]]) -> int:
    """Replace inline media blocks in `messages` with text placeholders.

    Attached media is base64-inlined into every request; leaving every
    attachment in the history grows the request body each turn until the
    gateway rejects it with 413. Mutates in place; returns blocks stripped.
    """
    stripped = 0
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        new_content: List[Dict[str, Any]] = []
        changed = False
        for block in content:
            if is_inline_media_block(block):
                new_content.append(
                    {
                        "type": "text",
                        "text": "[media attachment omitted from this request: "
                        "attached earlier in the conversation]",
                    }
                )
                changed = True
                stripped += 1
            else:
                new_content.append(block)
        if changed:
            msg["content"] = new_content
    return stripped


def _downscale_image(p: Path, max_bytes: int) -> Path:
    """Re-encode image `p` as a progressively smaller JPEG via ffmpeg.

    Returns a path to a JPEG no larger than `max_bytes`, or raises ValueError
    if even the smallest attempt exceeds the cap.
    """
    max_dim = 1568
    while max_dim >= 256:
        out_dir = Path(tempfile.mkdtemp(prefix="codescribe-img-"))
        out = out_dir / "scaled.jpg"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(p),
                "-vf",
                f"scale=min({max_dim}\\,iw):-2",
                "-q:v",
                "4",
                str(out),
            ],
            check=True,
            capture_output=True,
        )
        if out.stat().st_size <= max_bytes:
            return out
        max_dim //= 2
    raise ValueError(
        f"{p.name} exceeds the inline media cap of {max_bytes} bytes even after "
        "downscaling; the API gateway rejects oversized request bodies (413)"
    )


def file_parts(
    path: Union[str, Path], provider: str, nframes: int = 4
) -> List[Dict[str, Any]]:
    """Content blocks for `path`, routed by modality.

    `provider` is "anthropic" or "openai". Images and PDFs go inline as
    base64; videos are sampled to at most `nframes` evenly spaced frames via
    ffmpeg and each frame is attached as an image block.
    """
    p = Path(path)
    mime = guess_media_mime(p)
    if mime is None:
        raise ValueError(f"unsupported media type for {p.name}")

    def b64(f: Path) -> str:
        return base64.b64encode(f.read_bytes()).decode()

    if mime.startswith("image/"):
        src, src_mime = p, mime
        if p.stat().st_size > MAX_INLINE_MEDIA_BYTES:
            src = _downscale_image(p, MAX_INLINE_MEDIA_BYTES)
            src_mime = "image/jpeg"
        if provider == "anthropic":
            return [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": src_mime,
                        "data": b64(src),
                    },
                }
            ]
        return [
            {
                "type": "image_url",
                "image_url": {"url": f"data:{src_mime};base64,{b64(src)}"},
            }
        ]

    if mime == "application/pdf":
        if p.stat().st_size > MAX_INLINE_MEDIA_BYTES:
            raise ValueError(
                f"{p.name} is {p.stat().st_size} bytes, over the inline cap of "
                f"{MAX_INLINE_MEDIA_BYTES} bytes (base64 inflates ~4/3x and the "
                "gateway rejects large bodies with 413). Split the PDF or attach "
                "the relevant pages as images instead."
            )
        if provider == "anthropic":
            return [
                {
                    "type": "document",
                    "source": {"type": "base64", "media_type": mime, "data": b64(p)},
                }
            ]
        return [
            {
                "type": "input_file",
                "filename": p.name,
                "file_data": f"data:{mime};base64,{b64(p)}",
            }
        ]

    # video/*: extract frames, attach each as an image block. Frames are
    # downscaled to 1280px wide so each stays well under the inline cap.
    out_dir = Path(tempfile.mkdtemp(prefix="codescribe-frames-"))
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(p),
            "-vf",
            f"fps=1/{max(1, nframes)},scale=1280:-2",
            "-frames:v",
            str(nframes),
            str(out_dir / "f%02d.jpg"),
        ],
        check=True,
        capture_output=True,
    )
    parts: List[Dict[str, Any]] = []
    for frame in sorted(out_dir.glob("f*.jpg")):
        if frame.stat().st_size > MAX_INLINE_MEDIA_BYTES:
            frame = _downscale_image(frame, MAX_INLINE_MEDIA_BYTES)
        parts += file_parts(frame, provider, nframes)
    return parts
