---
name: numbering-code-blocks
description: Use when a markdown post has multiple fenced code blocks and the prose refers to them vaguely ("the code above", "as shown below", "the following snippet") instead of by number
---

# Numbering Code Blocks

## Overview

Label every fenced code block in a markdown file with a sequential "Code N" caption, then rewrite prose that points at code vaguely ("above", "below", "the following", "this code") into an explicit reference like "Code 3". Always use the word "Code", never "Listing" or "snippet", so labels and prose match.

## When to Use

- A post has 3+ fenced code blocks (```` ``` ````) and refers to them with position words ("above", "below", "the following code").
- Output/result blocks (tables, terminal output) count as code blocks too if they're fenced — number them in the same sequence.
- Don't use for posts with only one code block, or where fenced blocks are pure reference material never mentioned in prose.

## Process

0. **Check for pre-existing numbering first**: `grep -n '\*\*Code [0-9]*\*\*\|(Code [0-9]*)\|^\*Code [0-9]*'` — bold captions, inline `(Code N)` citations, and italic `*Code N: description*` captions below a block are all the same labeling scheme, just in different spots. If any exist, treat them as **stale, not authoritative** — a prior partial pass may have only numbered some blocks (e.g. just the "interesting" ones) and skipped others (pseudocode, output tables, ASCII diagrams). Renumber every existing label to match the new complete top-to-bottom sequence; don't leave two contradictory numbering schemes in the same document.
1. **Find every fenced block in order**: `grep -n '^```' path/to/post.md`. Pair up start/end lines — that's your block list, top to bottom.
2. **Number them 1..N in document order**, regardless of language tag or whether it's real code vs. an output table. Don't skip any fenced block.
3. **Insert a caption immediately above each opening fence**, on its own line with a blank line on each side:
   ```markdown
   Some lead-in sentence.

   **Code 3**

   ```python
   ...
   ```
   ```
4. **Rewrite vague references** in the surrounding prose. Search for `above|below|following code|this code|the code|code snippet` and replace position words with the block number:
   - "In the above code, we..." → "In Code 1, we..."
   - "The above code is to..." → "Code 2 is to..."
   - "as shown in the code above" → "as shown in Code 16"
   - "Below are the results." → keep, but add "(Code 10)" if a table follows.
   - Only touch sentences that describe a *code block*. Leave references to images/figures ("the above image/diagram") alone.
5. **Verify** with `grep -n '\*\*Code [0-9]*\*\*'` — confirm the numbers are sequential and each one is immediately followed by a ` ``` ` fence (`grep -n -A2 '\*\*Code [0-9]*\*\*' path.md`).

## Common Mistakes

- **Edit tool string-match failures on long spans**: when replacing a large old_string that spans a heading + full code block + following prose, an invisible character mismatch (curly quote, em dash, non-breaking space) makes the whole edit fail with no clue which part mismatched. Fix: split into two smaller edits — one for the caption insertion (few lines around the fence), one for the prose sentence — each anchored on a short, exact snippet.
- **Skipping output/table blocks**: a fenced block showing a results table (no language tag) is still a code block for numbering purposes — don't reserve numbers only for blocks tagged ` ```python `.
- **Renumbering after the fact**: number in one top-to-bottom pass before editing prose, so references don't drift as you go.
- **Partial prior numbering**: a caption style isn't always bold-above-the-fence — some posts caption code the same way they caption images, with an italic line *below* the block (`*Code N: description*`). Search for both styles (see step 0) before assuming a file is unlabeled.
