#!/usr/bin/env python3
"""
Convert a Medium HTML export to Markdown.

Images are classified via the `claude` CLI (uses your existing Claude Code auth):
  - Math images  -> KaTeX display blocks
  - Code images  -> fenced code blocks
  - Other images -> downloaded locally to images/posts/<blog-name>/

Usage:
  python3 medium_to_markdown.py <input.html> [output.md] [--repo-root PATH]

Defaults:
  output.md  -> content/posts/<date>-<slug>.md  (derived from filename)
  repo-root  -> two levels up from this script (i.e. the site root)

Use --dry-run to skip image classification (useful for testing/previewing).
"""

import argparse
import base64
import io
import json
import os
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag

# Max image bytes to send to the vision API (5MB API limit; stay under it)
MAX_IMAGE_BYTES = 4 * 1024 * 1024


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"[\s]+", "-", text.strip())
    return text


def download_image(url: str) -> tuple[bytes, str]:
    """Return (raw_bytes, mime_type)."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        mime = resp.headers.get_content_type() or "image/jpeg"
        return resp.read(), mime


def cdn_filename(url: str) -> str:
    path = urllib.parse.urlparse(url).path
    return os.path.basename(path)


def descriptive_filename(
    url: str,
    caption: str = "",
    alt: str = "",
    fig_index: int = 0,
) -> str:
    """Derive a human-readable filename from caption/alt/index."""
    ext = os.path.splitext(urllib.parse.urlparse(url).path)[1] or ".png"
    label = caption or alt
    # Strip source URLs that appear in captions
    label = re.sub(r"source:\s*https?://\S+", "", label, flags=re.IGNORECASE)
    label = re.sub(r"https?://\S+", "", label)
    label = re.sub(r"\s+", " ", label).strip().rstrip(":").strip()
    if label:
        slug = label.lower()
        slug = re.sub(r"[^a-z0-9\s-]", "", slug)
        slug = re.sub(r"\s+", "-", slug.strip())
        slug = re.sub(r"-+", "-", slug).strip("-")[:50]
        return slug + ext
    return f"img-{fig_index:02d}{ext}"


def fix_url(url: str) -> str:
    if url.startswith("http"):
        return url
    return "https://" + url.lstrip("/")


def maybe_shrink(data: bytes, mime: str) -> tuple[bytes, str]:
    """Resize image if over MAX_IMAGE_BYTES so it fits within the API limit."""
    if len(data) <= MAX_IMAGE_BYTES:
        return data, mime
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(data))
        # Convert animated GIFs to first frame PNG
        if getattr(img, "is_animated", False):
            img.seek(0)
            img = img.convert("RGBA")
        # Scale down until it fits
        scale = 0.7
        while True:
            new_w = max(1, int(img.width * scale))
            new_h = max(1, int(img.height * scale))
            resized = img.resize((new_w, new_h), Image.LANCZOS)
            buf = io.BytesIO()
            fmt = "PNG" if mime in ("image/png", "image/gif") else "JPEG"
            resized.save(buf, format=fmt)
            candidate = buf.getvalue()
            if len(candidate) <= MAX_IMAGE_BYTES or scale < 0.05:
                new_mime = "image/png" if fmt == "PNG" else "image/jpeg"
                return candidate, new_mime
            scale *= 0.7
    except ImportError:
        print("  WARN: Pillow not installed; sending oversized image as-is", file=sys.stderr)
    return data, mime


# ---------------------------------------------------------------------------
# Claude CLI classifier
# ---------------------------------------------------------------------------

CLASSIFY_PROMPT = """\
Look at this image and classify it into exactly one of three categories:

1. MATH  - the image contains mathematical equations, formulas, or notation
           (Greek letters, integrals, sums, matrices, etc.)
2. CODE  - the image shows source code, a terminal listing, or a code snippet
3. IMAGE - anything else (diagrams, photos, charts, algorithm pseudocode boxes,
           animations, etc.)

Then, depending on the category:

If MATH: Transcribe every equation as KaTeX-compatible LaTeX.
  - Wrap display equations in \\[...\\]
  - Add \\tag{N} if equation numbers are visible in the image
  - Output ONLY the LaTeX blocks, no prose

If CODE: Extract the code exactly.
  - Detect the programming language
  - Output a fenced code block: ```<language>\\n<code>\\n```
  - Output ONLY the code block, no prose

If IMAGE: Output a single short alt-text description (15 words max).

Your response format must be exactly:
CATEGORY: <MATH|CODE|IMAGE>
CONTENT:
<the LaTeX / code block / alt-text>
"""


def classify_image_via_cli(image_data: bytes, mime: str) -> tuple[str, str]:
    """Use `claude -p` with stream-json to classify an image.
    Returns (category, content). category in {'MATH','CODE','IMAGE'}.
    """
    image_data, mime = maybe_shrink(image_data, mime)
    b64 = base64.standard_b64encode(image_data).decode()

    payload = json.dumps({
        "type": "user",
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": mime, "data": b64},
                },
                {"type": "text", "text": CLASSIFY_PROMPT},
            ],
        },
    })

    result = subprocess.run(
        [
            "claude", "-p",
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--verbose",
        ],
        input=payload,
        capture_output=True,
        text=True,
        timeout=60,
    )

    # Extract result text from stream-json output
    raw = ""
    for line in result.stdout.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if obj.get("type") == "result":
                raw = obj.get("result", "")
                break
        except json.JSONDecodeError:
            pass

    if not raw:
        stderr = result.stderr.strip()
        raise RuntimeError(f"claude CLI returned no result. stderr: {stderr[:200]}")

    cat_m = re.search(r"CATEGORY:\s*(MATH|CODE|IMAGE)", raw, re.IGNORECASE)
    content_m = re.search(r"CONTENT:\s*\n(.*)", raw, re.DOTALL)
    category = cat_m.group(1).upper() if cat_m else "IMAGE"
    content = content_m.group(1).strip() if content_m else raw
    return category, content


def is_animated_gif(data: bytes) -> bool:
    """Quick check: GIF files are animated media, always treat as IMAGE."""
    return data[:6] in (b"GIF87a", b"GIF89a")


# ---------------------------------------------------------------------------
# Inline HTML -> Markdown
# ---------------------------------------------------------------------------

def node_to_md(node) -> str:
    if isinstance(node, NavigableString):
        return str(node)
    if not isinstance(node, Tag):
        return ""
    tag = node.name
    inner = "".join(node_to_md(c) for c in node.children)
    if tag in ("strong", "b"):
        stripped = inner.strip()
        return f"**{stripped}**" if stripped else ""
    if tag in ("em", "i"):
        stripped = inner.strip()
        return f"*{stripped}*" if stripped else ""
    if tag == "code":
        return f"`{inner}`"
    if tag == "a":
        href = node.get("href", "")
        m = re.search(r"[?&]url=([^&]+)", href)
        if m:
            href = urllib.parse.unquote(m.group(1))
        return f"[{inner}]({href})"
    if tag == "br":
        return "  \n"
    return inner


def graf_text(graf: Tag) -> str:
    return "".join(node_to_md(c) for c in graf.children).strip()


# ---------------------------------------------------------------------------
# Figure processing
# ---------------------------------------------------------------------------

def process_figure(
    fig: Tag,
    img_dir: Path,
    img_url_prefix: str,
    dry_run: bool = False,
    caption_override: str = "",
    fig_index: int = 0,
) -> str:
    img_tag = fig.find("img")
    if not img_tag:
        return ""
    url = fix_url(img_tag.get("src", ""))
    if not url:
        return ""

    figcaption = fig.find("figcaption")
    caption = caption_override or (figcaption.get_text().strip() if figcaption else "")
    # caption_md preserves inline links/formatting (get_text would flatten <a> tags)
    caption_md = caption_override or (graf_text(figcaption) if figcaption else "")
    alt_attr = img_tag.get("alt", "").strip()

    if dry_run:
        filename = descriptive_filename(url, caption, alt_attr, fig_index)
        alt = caption or alt_attr or filename
        return f"![{alt}]({img_url_prefix}/{filename})\n\n"

    print(f"  -> fetching {url[-70:]}", flush=True)
    try:
        data, mime = download_image(url)
    except Exception as exc:
        print(f"     WARN: download failed: {exc}", file=sys.stderr)
        return f"<!-- image download failed: {url} -->\n\n"

    # Animated GIFs are always regular images (not math/code)
    if is_animated_gif(data):
        category, content = "IMAGE", caption or alt_attr or cdn_filename(url)
    else:
        try:
            category, content = classify_image_via_cli(data, mime)
            print(f"     classified as {category}", flush=True)
        except Exception as exc:
            print(f"     WARN: classification failed: {exc}", file=sys.stderr)
            category, content = "IMAGE", caption or alt_attr or cdn_filename(url)

    if category == "MATH":
        content = re.sub(r"\$\$(.+?)\$\$", r"\\[\1\\]", content, flags=re.DOTALL)
        return content + "\n\n"

    if category == "CODE":
        return content + "\n\n"

    # Regular image: save with descriptive name, reuse existing identical file
    filename = descriptive_filename(url, caption, alt_attr, fig_index)
    dest = img_dir / filename
    img_dir.mkdir(parents=True, exist_ok=True)
    # Avoid overwriting an identical existing file; deduplicate by hash
    if dest.exists():
        import hashlib
        existing_md5 = hashlib.md5(dest.read_bytes()).hexdigest()
        new_md5 = hashlib.md5(data).hexdigest()
        if existing_md5 != new_md5:
            base, ext = os.path.splitext(filename)
            n = 2
            while (img_dir / f"{base}-{n}{ext}").exists():
                n += 1
            filename = f"{base}-{n}{ext}"
            dest = img_dir / filename
    dest.write_bytes(data)
    rel_path = f"{img_url_prefix}/{filename}"
    alt = caption or alt_attr or filename
    if caption:
        return f"![{alt}]({rel_path})\n*{caption_md or caption}*\n\n"
    return f"![{alt}]({rel_path})\n\n"


def process_outset_row(
    row_div: Tag,
    img_dir: Path,
    img_url_prefix: str,
    dry_run: bool = False,
    fig_index: int = 0,
) -> str:
    """Side-by-side image layout: emit images in an HTML table."""
    figures = row_div.find_all("figure", recursive=True)

    shared_cap = ""
    for fig in figures:
        cap = fig.find("figcaption")
        if cap and cap.get_text().strip():
            shared_cap = cap.get_text().strip()
            break

    parts = []
    for idx, fig in enumerate(figures):
        parts.append(process_figure(fig, img_dir, img_url_prefix, dry_run, "", fig_index + idx))

    # If all outputs are image refs, wrap in HTML table for side-by-side
    all_images = all(p.strip().startswith("![") for p in parts if p.strip())
    if all_images and len(parts) > 1:
        cells = "".join(f"<td>{p.strip()}</td>" for p in parts if p.strip())
        result = f"<table><tr>{cells}</tr></table>\n"
        if shared_cap:
            result += f"*{shared_cap}*\n"
        return result + "\n"

    result = "".join(parts)
    if shared_cap and not any(shared_cap in p for p in parts):
        result += f"*{shared_cap}*\n\n"
    return result


# ---------------------------------------------------------------------------
# List helpers
# ---------------------------------------------------------------------------

def consume_list_run(children: list, start: int) -> tuple[str, int]:
    """Consume a consecutive run of graf--li elements."""
    items = []
    i = start
    is_ordered = False
    while i < len(children):
        child = children[i]
        if not isinstance(child, Tag):
            i += 1
            continue
        if "graf--li" not in child.get("class", []):
            break
        if child.parent and child.parent.name == "ol":
            is_ordered = True
        items.append(graf_text(child))
        i += 1
    lines = []
    for idx, item in enumerate(items):
        prefix = f"{idx + 1}." if is_ordered else "-"
        lines.append(f"{prefix} {item}")
    return "\n".join(lines) + "\n\n", i


# ---------------------------------------------------------------------------
# Inline-math + formatting post-processing
# ---------------------------------------------------------------------------

# Unicode subscripts -> LaTeX subscript content
_SUB = {
    '₀':'0','₁':'1','₂':'2','₃':'3','₄':'4','₅':'5','₆':'6','₇':'7','₈':'8','₉':'9',
    'ᵢ':'i','ⱼ':'j','ᵣ':'r','ₙ':'n','ₓ':'x','ₖ':'k','ₘ':'m','ₚ':'p','ₜ':'t','ₕ':'h',
    '₋':'-','₊':'+',
}
# Unicode superscripts -> LaTeX superscript content
_SUP = {
    '⁰':'0','¹':'1','²':'2','³':'3','⁴':'4','⁵':'5','⁶':'6','⁷':'7','⁸':'8','⁹':'9',
    'ᵀ':'T','ᵗ':'t','ʰ':'h','ᴺ':'N','ᵇ':'b',
}
# Standalone math symbols -> LaTeX commands
_SYM = {
    '∂':r'\partial ','ℓ':r'\ell ','𝐿':'L','𝓛':'L','ℝ':r'\mathbb{R}',
    'ℛ':r'\mathcal{R}','∈':r'\in ','≤':r'\le ','≥':r'\ge ','≠':r'\ne ',
    '⊙':r'\odot ','⨀':r'\bigodot ','⋅':r'\cdot ','·':r'\cdot ','×':r'\times ',
    'Σ':r'\sum ','∞':r'\infty ','≈':r'\approx ',
    '⌈':r'\lceil ','⌉':r'\rceil ','⌊':r'\lfloor ','⌋':r'\rfloor ',
    '←':r'\leftarrow ','→':r'\to ','…':r'\dots ',
    'τ':r'\tau ','𝜏':r'\tau ','θ':r'\theta ','σ':r'\sigma ','μ':r'\mu ',
    'α':r'\alpha ','β':r'\beta ','γ':r'\gamma ','δ':r'\delta ',
}
_COMBINING_TILDE = '̃'
_PRIME = '’'  # right single quote, used as a prime inside math
_MATH_SIGNAL = set(_SUB) | set(_SUP) | set(_SYM) | {_COMBINING_TILDE, '√'}
# Chars that may form an inline math run. '*' is excluded so markdown **bold**
# markers are never consumed.
_RUN_ASCII = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/=+_^{}.")


def _is_run_char(ch: str) -> bool:
    return ch in _RUN_ASCII or ch in _MATH_SIGNAL or ch == _PRIME


def _convert_run(run: str) -> str:
    """Convert a raw run (known to contain a math signal) to a LaTeX body."""
    out = []
    i, n = 0, len(run)
    while i < n:
        ch = run[i]
        if ch in _SUB or ch in _SUP:
            kind = _SUB if ch in _SUB else _SUP
            grp = []
            while i < n and run[i] in kind:
                grp.append(kind[run[i]])
                i += 1
            body = ''.join(grp)
            wrap = '_' if kind is _SUB else '^'
            out.append(f'{wrap}{body}' if len(body) == 1 else f'{wrap}{{{body}}}')
            continue
        if ch == '√':
            if i + 1 < n:
                out.append(r'\sqrt{' + run[i + 1] + '}')
                i += 2
            else:
                out.append(r'\sqrt')
                i += 1
            continue
        if ch in _SYM:
            out.append(_SYM[ch])
            i += 1
            continue
        if ch == _PRIME:
            out.append("'")
            i += 1
            continue
        if i + 1 < n and run[i + 1] == _COMBINING_TILDE:
            out.append(r'\tilde{' + ch + '}')
            i += 2
            continue
        out.append(ch)
        i += 1
    return re.sub(r'\s+', ' ', ''.join(out)).strip()


def _process_prose(s: str) -> str:
    result = []
    i, n = 0, len(s)
    while i < n:
        if _is_run_char(s[i]):
            j = i
            while j < n and _is_run_char(s[j]):
                j += 1
            run = s[i:j]
            if any(c in _MATH_SIGNAL for c in run):
                lead = 0
                while lead < len(run) and run[lead] in ".'":
                    lead += 1
                trail = len(run)
                while trail > lead and run[trail - 1] in ".'":
                    trail -= 1
                core = run[lead:trail]
                if core and any(c in _MATH_SIGNAL for c in core):
                    result.append(run[:lead] + r'\(' + _convert_run(core) + r'\)' + run[trail:])
                else:
                    result.append(run)
            else:
                result.append(run)
            i = j
        else:
            result.append(s[i])
            i += 1
    return ''.join(result)


# Spans left untouched by inline-math conversion.
_PROT = re.compile(
    r'```.*?```'                  # fenced code
    r'|`[^`\n]*`'                 # inline code
    r'|\\\[.*?\\\]'               # display math
    r'|\\\(.*?\\\)'               # inline math (already converted)
    r'|!\[[^\]]*\]\([^)]*\)'      # image
    r'|\[[^\]]*\]\([^)]*\)'       # link
    r'|<[^>\n]+>',                # html tag
    re.DOTALL,
)


def convert_inline_math(text: str) -> str:
    r"""Wrap inline Unicode math runs in \(...\), leaving code, display math,
    existing inline math, links/images and HTML intact."""
    out, pos = [], 0
    for m in _PROT.finditer(text):
        out.append(_process_prose(text[pos:m.start()]))
        out.append(m.group(0))
        pos = m.end()
    out.append(_process_prose(text[pos:]))
    return ''.join(out)


def normalize_display_math(text: str) -> str:
    r"""Ensure a blank line between back-to-back display-math blocks. Consecutive
    \] / \[ on adjacent lines otherwise merge into one block and only the first
    renders under KaTeX."""
    lines = text.split('\n')
    out = []
    for i, ln in enumerate(lines):
        out.append(ln)
        if i + 1 < len(lines) and ln.rstrip().endswith(r'\]') \
                and lines[i + 1].lstrip().startswith(r'\['):
            out.append('')
    return '\n'.join(out)


def number_figures(text: str) -> str:
    """Add sequential 'fig N:' labels to standalone image figures and ensure each
    has a matching caption line. (Numbers every top-level image; review the output
    if some images, e.g. portraits/hero banners, should be excluded.)"""
    lines = text.split('\n')
    out, n, i = [], 0, 0
    img_re = re.compile(r'^!\[(.*?)\]\((.*?)\)\s*$')
    cap_re = re.compile(r'^\*(.+)\*\s*$')
    strip_fig = re.compile(r'^fig\s*\d+:\s*', re.I)
    while i < len(lines):
        m = img_re.match(lines[i])
        if m:
            n += 1
            desc = strip_fig.sub('', m.group(1)).strip() or f'figure {n}'
            out.append(f'![fig {n}: {desc}]({m.group(2)})')
            cm = cap_re.match(lines[i + 1].strip()) if i + 1 < len(lines) else None
            if cm:
                cap = strip_fig.sub('', cm.group(1)).strip() or desc
                out.append(f'*fig {n}: {cap}*')
                i += 2
            else:
                out.append(f'*fig {n}: {desc}*')
                i += 1
            continue
        out.append(lines[i])
        i += 1
    return '\n'.join(out)


def number_code(text: str) -> str:
    """Add sequential 'Code N:' captions after substantial fenced code listings
    (those defining a function/class). Short inline excerpts and bare ``` output
    or diagram blocks are skipped."""
    lines = text.split('\n')
    out, n, i = [], 0, 0
    open_re = re.compile(r'^```(\w+)\s*$')
    close_re = re.compile(r'^```\s*$')
    name_re = re.compile(r'\b(?:def|class)\s+(\w+)')
    while i < len(lines):
        if open_re.match(lines[i]):
            block = [lines[i]]
            j = i + 1
            while j < len(lines) and not close_re.match(lines[j]):
                block.append(lines[j])
                j += 1
            if j < len(lines):
                block.append(lines[j])  # closing fence
            out.extend(block)
            nm = name_re.search('\n'.join(block))
            if nm:  # substantial listing -> number it
                n += 1
                out.append('')
                out.append(f'*Code {n}: `{nm.group(1)}`*')
            i = j + 1
            continue
        out.append(lines[i])
        i += 1
    return '\n'.join(out)


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(
    html_path: Path,
    output_path: Path,
    repo_root: Path,
    dry_run: bool = False,
    inline_math: bool = True,
    num_figures: bool = False,
    num_code: bool = False,
) -> None:
    with html_path.open(encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    # --- Metadata ---
    title_el = soup.find("h1", class_="p-name")
    title_text = title_el.get_text().strip() if title_el else html_path.stem

    time_el = soup.find("time", class_="dt-published")
    date_str = time_el.get("datetime", "")[:10] if time_el else ""

    summary_sec = soup.find("section", class_="p-summary")
    description = summary_sec.get_text().strip() if summary_sec else ""

    stem = output_path.stem
    slug_m = re.match(r"\d{4}-\d{2}-\d{2}-(.*)", stem)
    blog_slug = slug_m.group(1) if slug_m else slugify(title_text)

    img_url_prefix = f"/images/posts/{blog_slug}"
    img_dir = repo_root / "images" / "posts" / blog_slug

    # --- Content traversal ---
    lines: list[str] = []
    description_normalized = re.sub(r"\s+", " ", description).strip()
    fig_counter = [0]  # mutable counter shared across all figure calls

    section_inners = soup.find_all("div", class_="section-inner")

    for div in section_inners:
        div_classes = div.get("class", [])

        if "sectionLayout--outsetRow" in div_classes:
            print("Processing outset row (side-by-side)...", flush=True)
            n = len(div.find_all("figure", recursive=True))
            lines.append(process_outset_row(div, img_dir, img_url_prefix, dry_run, fig_counter[0]))
            fig_counter[0] += n
            continue

        children = list(div.children)
        i = 0
        while i < len(children):
            child = children[i]
            if not isinstance(child, Tag):
                i += 1
                continue

            classes = child.get("class", [])

            # Skip title (goes to frontmatter)
            if "graf--title" in classes:
                i += 1
                continue

            if "graf--h3" in classes:
                lines.append(f"## {graf_text(child)}\n\n")
                i += 1
                continue

            if "graf--h4" in classes:
                lines.append(f"### {graf_text(child)}\n\n")
                i += 1
                continue

            if child.name == "hr":
                lines.append("---\n\n")
                i += 1
                continue

            if "graf--pre" in classes:
                code_text = child.get_text()
                lang = detect_language(code_text)
                lines.append(f"```{lang}\n{code_text}\n```\n\n")
                i += 1
                continue

            if "graf--li" in classes:
                text, i = consume_list_run(children, i)
                lines.append(text)
                continue

            if child.name in ("ul", "ol"):
                items = child.find_all("li", class_="graf--li")
                is_ordered = child.name == "ol"
                list_lines = []
                for idx, li in enumerate(items):
                    prefix = f"{idx + 1}." if is_ordered else "-"
                    list_lines.append(f"{prefix} {graf_text(li)}")
                lines.append("\n".join(list_lines) + "\n\n")
                i += 1
                continue

            if "graf--p" in classes:
                text = graf_text(child)
                normalized = re.sub(r"\s+", " ", text).strip()
                if text and normalized != description_normalized:
                    lines.append(f"{text}\n\n")
                i += 1
                continue

            if "graf--figure" in classes:
                print("Processing figure...", flush=True)
                lines.append(process_figure(child, img_dir, img_url_prefix, dry_run, "", fig_counter[0]))
                fig_counter[0] += 1
                i += 1
                continue

            if "graf--mixtapeEmbed" in classes:
                a = child.find("a")
                if a:
                    href = a.get("href", "")
                    m = re.search(r"[?&]url=([^&]+)", href)
                    if m:
                        href = urllib.parse.unquote(m.group(1))
                    text = a.get_text().strip() or href
                    lines.append(f"[{text}]({href})\n\n")
                i += 1
                continue

            i += 1

    # --- Frontmatter ---
    fm_lines = [
        "---",
        f'title: "{title_text}"',
        f"date: {date_str}",
        f'description: "{description}"',
        f"slug: {blog_slug}",
        "draft: false",
        "---",
        "",
        "[TOC]",
        "",
    ]
    # --- Post-processing passes (operate on the body, not the frontmatter) ---
    body = "".join(lines)
    body = normalize_display_math(body)
    if inline_math:
        body = convert_inline_math(body)
    if num_figures:
        body = number_figures(body)
    if num_code:
        body = number_code(body)

    full_content = "\n".join(fm_lines) + "\n" + body

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(full_content, encoding="utf-8")
    print(f"\nWrote {output_path}", flush=True)


def detect_language(code: str) -> str:
    if re.search(r"\bimport\b|\bdef\b|\bself\b|\bprint\(|\belif\b", code):
        return "python"
    if re.search(r"@triton|tl\.\w+", code):
        return "python"
    if re.search(r"\b(#include|cout|cin|int\s+main)\b", code):
        return "cpp"
    if re.search(r"\b(const|let|var|function|=>|console\.log)\b", code):
        return "javascript"
    if re.search(r"\b(public|private|static|void|class)\b.*\{", code, re.DOTALL):
        return "java"
    if re.search(r"^\s*[$#]|\b(apt-get|chmod)\b", code, re.MULTILINE):
        return "bash"
    return ""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="Path to the Medium HTML export file")
    parser.add_argument("output", nargs="?", help="Output markdown path (auto-derived if omitted)")
    parser.add_argument("--repo-root", default=None, help="Repository root directory")
    parser.add_argument("--dry-run", action="store_true", help="Skip image classification; emit placeholder image refs")
    parser.add_argument("--no-inline-math", action="store_true", help="Do not convert inline Unicode math to \\(...\\)")
    parser.add_argument("--number-figures", action="store_true", help="Add sequential 'fig N:' labels and captions to images")
    parser.add_argument("--number-code", action="store_true", help="Add sequential 'Code N:' captions to substantial code listings")
    args = parser.parse_args()

    html_path = Path(args.input).resolve()
    if not html_path.exists():
        sys.exit(f"Input file not found: {html_path}")

    repo_root = (
        Path(args.repo_root).resolve()
        if args.repo_root
        else Path(__file__).resolve().parent.parent
    )

    if args.output:
        output_path = Path(args.output).resolve()
    else:
        stem = html_path.stem
        stem = re.sub(r"-[a-f0-9]{12}$", "", stem)
        stem = stem.replace("_", "-").lower()
        date_m = re.match(r"(\d{4}-\d{2}-\d{2})-(.*)", stem)
        if date_m:
            slug = slugify(date_m.group(2))
            stem = f"{date_m.group(1)}-{slug}"
        output_path = repo_root / "content" / "posts" / f"{stem}.md"

    print(f"Input:      {html_path}")
    print(f"Output:     {output_path}")
    print(f"Repo root:  {repo_root}")
    print(f"Dry run:    {args.dry_run}")
    print()

    convert(
        html_path,
        output_path,
        repo_root,
        dry_run=args.dry_run,
        inline_math=not args.no_inline_math,
        num_figures=args.number_figures,
        num_code=args.number_code,
    )


if __name__ == "__main__":
    main()
