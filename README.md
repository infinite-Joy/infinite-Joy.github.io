# infinite-Joy.github.io

Personal website for [Joydeep Bhattacharjee](https://infinite-joy.github.io) — Lead ML Engineer at Adobe, author, and mentor.

## Stack

- Static HTML/CSS/JS, served by GitHub Pages
- No build step for the homepage — edit `index.html` directly
- Blog pipeline (in progress): Markdown → Python → raw HTML (see `PLAN.md`)

## Commands

### Homepage — local preview

```bash
# Serve the site locally
python -m http.server 8000
# Then open http://localhost:8000
```

### Blog pipeline — setup

```bash
# Install dependencies (one-time)
pip install -r requirements.txt
```

### Blog pipeline — build & preview

```bash
# Build published posts only
python build.py

# Build including drafts
python build.py --drafts

# Build and serve at http://localhost:8000/blog/
python build.py --serve
```

### Writing a new post

```bash
# Create a new post file (edit content inside)
touch content/posts/YYYY-MM-DD-my-post-title.md
```

## Repo layout

```
index.html          # homepage
css/                # third-party CSS (Bootstrap)
js/                 # third-party JS
images/             # photos
assets/             # generated site CSS, JS, and images (blog pipeline)
content/posts/      # Markdown source for blog posts
templates/          # Jinja2 templates (blog pipeline)
blog/               # generated blog HTML — do not hand-edit
feed.xml            # generated RSS feed
```

# Blogs