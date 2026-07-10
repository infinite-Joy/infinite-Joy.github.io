---
name: numbering-images
description: Use when a markdown post has multiple images and the prose refers to them vaguely ("the above image", "the diagram below", "the second image") instead of by number
---

# Numbering Images

## Overview

Label every image in a markdown file with a sequential "Image N" caption, then rewrite prose that points at an image vaguely ("above", "below", "the following", "this diagram") into an explicit reference like "Image 5". Always use the word "Image", never "Figure" or "picture", so captions and prose match.

## When to Use

- A post has multiple `![alt](path)` images and refers to them with position words ("above", "below", "the following diagram") or ordinal words ("the second image", "the last image").
- Every image counts, including banner/hero images with no descriptive alt text and images that are reused later in the post (same file referenced twice still gets its own number at each position it appears).
- Don't use for posts with only one image, or where images are pure decoration never mentioned in prose.

## Process

0. **Check for pre-existing numbering first**: `grep -n '^\*Image [0-9]*'` (and also `grep -n -i 'image [0-9]'` in prose). If any exist, treat them as **stale, not authoritative** — renumber everything to match a fresh complete top-to-bottom pass.
1. **Find every image in order**: `grep -n '^!\[' path/to/post.md`. That's your image list, top to bottom, in document order.
2. **Learn this file's caption convention before writing anything.** Read a few lines after each `![...]` line. Two conventions are common in this blog:
   - An italic caption line immediately below the image, usually repeating the alt text: `*alt text*`.
   - No caption at all (common for banner/hero images with generic filenames like `hero.png` or `img-00.png`).
   Match whatever the file already does; don't invent a new caption style.
3. **Number every image 1..N in document order**, regardless of whether it already has a caption:
   - If a caption line exists, prepend `Image N: ` inside the italics: `*alt text*` → `*Image N: alt text*`.
   - If no caption line exists, insert a minimal one directly below the image: `*Image N*`.
4. **Rewrite vague references** in the surrounding prose. Search for `above|below|following (image|diagram|figure|chart|graph)|this image|the image|second image|last image` and replace with the specific image number:
   - "The above image illustrates X" → "Image 8 illustrates X".
   - "as shown in the diagram below" → "as shown in Image 12".
   - Resolve "above"/"below" by finding the nearest image in that direction in the document, not by guessing.
   - If a paragraph uses "second image" / "last image" to mean different **panels or rows of the same composite figure** (not separate images), don't invent extra image numbers — write "the second panel of Image 7" / "the last panel of Image 7" instead.
   - Only touch sentences that describe an *image*. Leave references to code blocks alone (see the `numbering-code-blocks` skill for those).
5. **Verify** with `grep -n '^\*Image [0-9]*'` — confirm the numbers are sequential with no gaps, and `grep -c '^!\['` vs `grep -c '^\*Image [0-9]*'` match (or the difference equals the number of intentionally uncaptioned images, if this file's convention keeps some uncaptioned — decide this per file in step 2, don't mix conventions within one file).

## Common Mistakes

- **Edit tool string-match failures from invisible characters**: alt text and captions in these posts often contain non-breaking spaces (`\xa0`), thin-space em dashes (` — `), or curly quotes that look identical to regular characters when read but make the `Edit` tool's exact-string match fail with no indication why. When an `Edit` fails on a caption or prose line that looks correct, inspect the raw bytes first:
  ```python
  with open(path, encoding='utf-8') as f:
      print(repr(f.readlines()[line_no - 1]))
  ```
  Then perform the replacement with a small Python script (read lines, edit by index, write back) instead of retrying `Edit` with cosmetically-different quoting.
- **Skipping banner/hero images**: a title image with a non-descriptive filename (`hero.png`, `img-00.png`) still gets a number if the file's convention is to caption all images; don't silently exclude it just because it has no natural caption text.
- **Confusing panels of one image with separate images**: prose describing a single multi-panel figure ("top row shows X... the second image shows Y... the last image shows Z") is usually describing rows/panels of one image, not three different images. Check how many `![...]` lines actually exist in that section before assigning new numbers.
- **Renumbering after the fact**: number in one top-to-bottom pass before editing prose, so references don't drift as you go.
