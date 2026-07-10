---
name: numbering-equations
description: Use when a markdown post has multiple display-math (\[...\]) equations and the prose refers to them vaguely ("the equation above", "as shown below", "these overall ideas") instead of by number
---

# Numbering Equations

## Overview

Label every display-math block (`\[...\]`) in a markdown file with a sequential number using `\tag{N}` **inside** the block, then rewrite prose that points at equations vaguely ("above", "below", "the following", "this equation") or with stale/inconsistent numbers into an explicit lowercase reference like "equation 4". This mirrors `numbering-code-blocks` and `numbering-images`, but the mechanics are different and stricter because of how this blog's math renderer works — read the whole Process section before editing, don't just pattern-match the other two skills.

## When to Use

- A post has multiple `\[ ... \]` display-math blocks and refers to them with position words ("the equation above", "as shown below") or with numbers that don't actually match document order (common in math-heavy posts drafted from slides/notebooks, where the author wrote "equation 2", "equation 3" loosely and the numbers drift or restart).
- Don't use for posts with only one or two equations never cross-referenced, or where equations are pure reference material never mentioned in prose.

## How this blog renders equations (read first)

`build.py` runs markdown through `pymdownx.arithmatex` (generic mode), which turns a `\[...\]` block into `<div class="arithmatex">...</div>` for client-side KaTeX rendering (see `templates/post.html.j2`). Arithmatex's block processor only recognizes a `\[...\]` as math if **the entire markdown block — the text between two blank lines — contains nothing else**. This has one critical consequence:

- **Never put a caption/label on its own line right after `\]`** (e.g. `*Equation 4*` under the block, the way `numbering-images` captions images). Even with a blank line before it, if there's no blank line separating `\]` from the caption line, the caption text and the equation collapse into one markdown block, arithmatex's regex no longer matches the whole block, and the equation silently falls through as **literal unrendered text** (e.g. the page shows `[ y = wx ] Equation 4` instead of rendered math). This is easy to miss because the markdown source looks fine — you only see the bug by building the site and inspecting the HTML.
- The correct pattern (used throughout this blog, e.g. `content/posts/2026-03-08-flash-attention.md`) is to put the number **inside** the equation as a LaTeX `\tag{N}`, on the last content line before the closing `\]`:
  ```
  \[
  y = wx \tag{4}
  \]
  ```
  For single-line equations, it goes inline before the closing delimiter: `\[ y = wx \tag{4} \]`.
- Prose references use **lowercase** "equation N" / "equations N–M" (not "Equation"), except where normal English capitalizes the first word of a sentence — matching the existing convention in `flash-attention.md` ("as seen in the below algorithm... equation 1", "equations 2–6").
- **Every math symbol mentioned in prose gets wrapped in `\(...\)` — no bare Greek letters or variables.** `flash-attention.md` wraps every single mention (`\(\tau\)`, `\(m_i\)`, `\(\ell_i\)`, `\(\ell'_i\)`); the only bare occurrences are inside code fences/backticks, which is a different (code, not math) context. Don't take a post's own existing bare-Greek-letter habit (e.g. repeated bare `Δ`, `ϕ`, `κ` in a pre-numbering draft) as the convention to preserve — check `flash-attention.md`, not the file being edited, since older drafts predate this convention.

## Process

0. **Check for pre-existing numbering first**: `grep -n '\\tag{' path.md` and `grep -noiE 'equation ?[0-9]+' path.md`. Treat any existing numbers as **stale, not authoritative** — in derivation-heavy posts the author often labels only some equations, or reuses the same number loosely across a multi-step derivation. Plan to renumber everything in one fresh top-to-bottom pass.
1. **Find every display-math block in order**: `grep -n '^\\\[$\|\\\[ .*\\\]$' path.md` (matches both multi-line `\[` / `\]` blocks and single-line `\[ ... \]` blocks). This is your equation list, top to bottom, in document order.
2. **Decide the numbering granularity before touching anything, especially for derivation-heavy sections.** A single algebraic derivation (e.g. solving for a variable step by step) is often split across many separate `\[...\]` blocks — sometimes 10+ for one "named" result. Two valid approaches:
   - **Every block gets its own number** (most literal, matches "number the equations" taken strictly; used when the user hasn't indicated otherwise).
   - **Group intermediate algebra steps under one number**, matching the coarser grain the author's own prose often already implies (e.g. "as shown in equation 5" covering a 4-block derivation chain).
   If the post has long multi-step derivations and the request is ambiguous, ask the user which granularity they want before doing the full rewrite — the scope and amount of prose rewriting differs enormously between the two.
3. **Number the blocks 1..N in document order.** For each block, append `\tag{N}` to the last non-blank content line before the closing `\]` (or inline before the closing `\]` for single-line blocks). Do not add any text outside the `\[...\]` delimiters.
4. **Rewrite every reference to an equation in the surrounding prose** to cite the correct number, in lowercase ("equation N"), including:
   - Vague position references: "as shown in the equation above/below" → "as shown in equation 4".
   - Stale/incorrect existing numbers from a prior loose-numbering pass — these are common and easy to miss; re-derive the correct number from the block's actual document position, don't trust the old label.
   - Narrated derivation steps that don't currently cite a number at all ("in the third line...", "rearranging some terms gives us...") — if you chose per-block numbering in step 2, these need an explicit "equation N" citation too, since every numbered equation should be referenced by number per the task.
   - Ranges: "equations 12–15" for a group, matching this blog's existing style for citing a contiguous group.
5. **Verify the source is internally consistent** before rebuilding:
   ```bash
   grep -o '\\tag{[0-9]*}' path.md | grep -o '[0-9]*' | sort -n | uniq -d   # duplicates (should be empty)
   python3 -c "
   import re
   nums=[int(x) for x in re.findall(r'\\\\tag\{(\d+)\}', open('path.md').read())]
   print(len(nums), min(nums), max(nums), sorted(set(range(1,max(nums)+1))-set(nums)))
   "
   grep -noiE 'equation ?[0-9]+' path.md   # eyeball that every citation looks right
   ```
6. **Rebuild and verify the actual rendered output** — this is not optional, since the source can look correct while still failing to render (see the caption-line trap above):
   ```bash
   ./venv/bin/python3 build.py
   grep -c 'class="arithmatex"' blog/<slug>/index.html   # should be >= number of equations (inline math adds more)
   grep -n '<p>\\\[' blog/<slug>/index.html              # any hit means an equation fell through unrendered
   ```

## Common Mistakes

- **Captioning equations like images/code** (a separate `*Equation N*` line after the block). This is the single most likely mistake — it looks identical in the markdown source to the working `\tag{N}` version but silently breaks rendering. Always use `\tag{N}` inside the delimiters instead.
- **Trusting the source without rebuilding**: because the caption-line bug produces valid-looking markdown, you must render the actual HTML and grep for `class="arithmatex"` / stray `<p>\[` to confirm every equation really became math and didn't fall through as text.
- **Capitalizing "Equation" everywhere**: this blog uses lowercase "equation N" in the middle of sentences; only capitalize when it's genuinely the first word of a sentence.
- **Edit tool string-match failures from invisible characters**: prose in these posts sometimes uses a hair-space + em-dash + hair-space sequence (` — `) instead of a plain " — ". Retyping what looks like the same dash in an `Edit` call will silently fail to match. When an edit spanning such text fails, extract the exact substring from the file with a small Python script (`re.search` / slicing on the read content) instead of retyping it, and do the replacement via `content.replace(old, new)` with an `assert content.count(old) == 1` check rather than repeated `Edit` retries.
- **Not renumbering stale in-prose references**: a derivation-heavy post frequently already contains loose "equation N" mentions from the author (grouped coarsely, or numbered per-section instead of globally). Don't leave these as-is just because they look like they're "already numbered" — recompute every citation against the true top-to-bottom block position from step 1.
- **Splitting one multi-panel/multi-step explanation into the wrong number of equations**: check step 2's granularity decision was actually followed consistently — don't accidentally group some derivations and number others block-by-block within the same post.
