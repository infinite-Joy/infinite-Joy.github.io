#!/usr/bin/env python3
"""Static blog generator: content/posts/*.md -> blog/<slug>/index.html + feed.xml"""

import argparse
import http.server
import math
import os
import re
import sys
import threading
import time
import xml.sax.saxutils as saxutils
from datetime import date, datetime
from pathlib import Path

import markdown
import yaml
from jinja2 import Environment, FileSystemLoader
from pygments.formatters import HtmlFormatter

ROOT = Path(__file__).parent
CONTENT_DIR = ROOT / "content" / "posts"
TEMPLATES_DIR = ROOT / "templates"
BLOG_DIR = ROOT / "blog"
ASSETS_CSS = ROOT / "assets" / "css"
BASE_URL = "https://infinite-joy.github.io"

PYGMENTS_DARK = "dracula"
PYGMENTS_LIGHT = "friendly"

MD_EXTENSIONS = [
    "pymdownx.arithmatex",
    "fenced_code",
    "codehilite",
    "tables",
    "toc",
    "footnotes",
    "attr_list",
    "md_in_html",
    "smarty",
    "sane_lists",
]
MD_EXTENSION_CONFIGS = {
    "codehilite": {"css_class": "codehilite", "guess_lang": False},
    "toc": {"permalink": "#", "permalink_title": "Link to this section"},
    "pymdownx.arithmatex": {"generic": True},
}


def parse_post(path: Path) -> dict | None:
    text = path.read_text(encoding="utf-8")
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            front = yaml.safe_load(parts[1]) or {}
            body_md = parts[2].lstrip("\n")
        else:
            front, body_md = {}, text
    else:
        front, body_md = {}, text

    stem = path.stem  # e.g. "2026-06-22-my-title"
    date_match = re.match(r"(\d{4}-\d{2}-\d{2})-(.+)", stem)

    raw_date = front.get("date")
    if raw_date is None and date_match:
        raw_date = date_match.group(1)
    if isinstance(raw_date, date) and not isinstance(raw_date, datetime):
        raw_date = raw_date.isoformat()
    elif isinstance(raw_date, datetime):
        raw_date = raw_date.date().isoformat()

    slug = front.get("slug")
    if slug is None:
        slug = date_match.group(2) if date_match else stem

    words = len(body_md.split())
    reading_time = max(1, math.ceil(words / 200))

    try:
        dt = datetime.strptime(raw_date, "%Y-%m-%d")
        date_human = dt.strftime("%-d %B %Y")
        date_iso = dt.date().isoformat()
    except (TypeError, ValueError):
        date_human = str(raw_date) if raw_date else ""
        date_iso = date_human

    md = markdown.Markdown(
        extensions=MD_EXTENSIONS, extension_configs=MD_EXTENSION_CONFIGS
    )
    body_html = md.convert(body_md)

    return {
        "title": front.get("title", slug.replace("-", " ").title()),
        "date_iso": date_iso,
        "date_human": date_human,
        "description": front.get("description", ""),
        "eyebrow": front.get("eyebrow", ""),
        "tags": front.get("tags", []),
        "slug": slug,
        "cover": front.get("cover", ""),
        "draft": bool(front.get("draft", False)),
        "reading_time": reading_time,
        "body": body_html,
        "canonical": f"{BASE_URL}/blog/{slug}/",
        "og_type": "article",
    }


def generate_highlight_css():
    dark_fmt = HtmlFormatter(style=PYGMENTS_DARK)
    light_fmt = HtmlFormatter(style=PYGMENTS_LIGHT)

    def scoped(formatter, selector_prefix):
        raw = formatter.get_style_defs(".codehilite")
        lines = []
        for line in raw.splitlines():
            if not line.strip() or line.strip().startswith("/*"):
                continue
            # skip bare element rules like "pre { ... }" that aren't .codehilite
            if re.match(r"^[a-z]", line.strip()):
                continue
            if selector_prefix:
                line = re.sub(r"^\.codehilite", f"{selector_prefix} .codehilite", line)
            lines.append(line)
        return "\n".join(lines)

    css = scoped(dark_fmt, "") + "\n\n" + scoped(light_fmt, '[data-theme="light"]')
    ASSETS_CSS.mkdir(parents=True, exist_ok=True)
    (ASSETS_CSS / "highlight.css").write_text(css, encoding="utf-8")
    print("  Generated assets/css/highlight.css")


def build(include_drafts: bool = False):
    CONTENT_DIR.mkdir(parents=True, exist_ok=True)
    BLOG_DIR.mkdir(parents=True, exist_ok=True)

    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=False)
    env.filters["tojson"] = __import__("json").dumps

    posts = []
    for md_file in sorted(CONTENT_DIR.glob("*.md"), reverse=True):
        if "bak" in md_file.name:
            continue
        post = parse_post(md_file)
        if post is None:
            continue
        if post["draft"] and not include_drafts:
            continue
        posts.append(post)

    post_tmpl = env.get_template("post.html.j2")
    for post in posts:
        out_dir = BLOG_DIR / post["slug"]
        out_dir.mkdir(parents=True, exist_ok=True)
        html = post_tmpl.render(**post)
        (out_dir / "index.html").write_text(html, encoding="utf-8")
        print(f"  Post: blog/{post['slug']}/index.html")

    index_tmpl = env.get_template("index.html.j2")
    index_html = index_tmpl.render(
        posts=posts,
        title="Blog â€” Joydeep Bhattacharjee",
        description="Writing on LLM inference, ML systems, and deep learning by Joydeep Bhattacharjee.",
        canonical=f"{BASE_URL}/blog/",
        cover="",
    )
    (BLOG_DIR / "index.html").write_text(index_html, encoding="utf-8")
    print("  Index: blog/index.html")

    _write_feed(posts)
    generate_highlight_css()
    print(f"Done â€” {len(posts)} post(s) built.")


def _write_feed(posts):
    def esc(s):
        return saxutils.escape(str(s))

    items = []
    for p in posts[:20]:
        items.append(
            f"""  <item>
    <title>{esc(p['title'])}</title>
    <link>{esc(p['canonical'])}</link>
    <guid>{esc(p['canonical'])}</guid>
    <pubDate>{p['date_iso']}</pubDate>
    <description>{esc(p['description'])}</description>
  </item>"""
        )

    feed = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">
  <channel>
    <title>Joydeep Bhattacharjee</title>
    <link>{BASE_URL}</link>
    <description>Writing on LLM inference, ML systems, and deep learning.</description>
    <atom:link href="{BASE_URL}/feed.xml" rel="self" type="application/rss+xml"/>
{chr(10).join(items)}
  </channel>
</rss>"""
    (ROOT / "feed.xml").write_text(feed, encoding="utf-8")
    print("  Feed: feed.xml")


def _watched_paths():
    """Yield all files that should trigger a rebuild when changed."""
    yield from (ROOT / "content" / "posts").glob("*.md")
    yield from (ROOT / "templates").glob("*.j2")
    yield from (ROOT / "assets" / "css").glob("*.css")
    yield ROOT / "build.py"


def _snapshot():
    return {p: p.stat().st_mtime for p in _watched_paths() if p.exists()}


def watch_and_rebuild(include_drafts: bool = False):
    last = _snapshot()
    print("Watching for changesâ€¦ (Ctrl+C to stop)")
    while True:
        time.sleep(1)
        current = _snapshot()
        changed = [p for p, t in current.items() if last.get(p) != t] + \
                  [p for p in last if p not in current]
        if changed:
            for p in changed:
                print(f"  changed: {p.relative_to(ROOT)}")
            try:
                build(include_drafts=include_drafts)
            except Exception as e:
                print(f"  Build error: {e}")
            last = _snapshot()


def main():
    parser = argparse.ArgumentParser(description="Build the blog.")
    parser.add_argument("--drafts", action="store_true", help="Include draft posts")
    parser.add_argument("--serve", action="store_true", help="Serve after building")
    parser.add_argument("--watch", action="store_true", help="Rebuild on file changes")
    args = parser.parse_args()

    build(include_drafts=args.drafts)

    if args.serve:
        os.chdir(ROOT)
        port = 8000
        print(f"\nServing at http://localhost:{port}/blog/  (Ctrl+C to stop)\n")
        handler = http.server.SimpleHTTPRequestHandler
        httpd = http.server.HTTPServer(("", port), handler)
        server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        server_thread.start()

        if args.watch:
            try:
                watch_and_rebuild(include_drafts=args.drafts)
            except KeyboardInterrupt:
                pass
        else:
            try:
                server_thread.join()
            except KeyboardInterrupt:
                pass
        httpd.shutdown()
    elif args.watch:
        try:
            watch_and_rebuild(include_drafts=args.drafts)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()