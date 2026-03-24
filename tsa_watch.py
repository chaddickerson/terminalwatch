#!/usr/bin/env python3
"""TSA wait time monitor — scrapes Reddit and Bluesky for real-time reports.

Usage:
    python3 scripts/tsa_watch.py LGA --terminal C --hours 24
    python3 scripts/tsa_watch.py JFK --hours 12
    python3 scripts/tsa_watch.py LGA --terminal C --hours 48 --json
"""

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime, timezone, timedelta

# Common airport name mappings for search broadening
AIRPORT_NAMES = {
    "LGA": ["LaGuardia", "La Guardia", "LGA"],
    "JFK": ["JFK", "Kennedy", "John F Kennedy"],
    "EWR": ["Newark", "EWR", "Newark Liberty"],
    "ORD": ["O'Hare", "OHare", "ORD"],
    "LAX": ["LAX", "Los Angeles"],
    "SFO": ["SFO", "San Francisco"],
    "ATL": ["ATL", "Atlanta", "Hartsfield"],
    "DFW": ["DFW", "Dallas"],
    "DEN": ["DEN", "Denver"],
    "SEA": ["SEA", "Seattle", "SeaTac"],
    "BOS": ["BOS", "Boston", "Logan"],
    "MIA": ["MIA", "Miami"],
    "TYS": ["TYS", "Knoxville", "McGhee Tyson"],
}

# Subreddits searched for every airport
SUBREDDITS = [
    "TSA", "delta", "flying", "americanairlines", "unitedairlines",
    "travel", "airports", "jetblue", "spiritair", "SouthwestAirlines",
]

# Additional subs searched only for the specific airport
AIRPORT_SUBREDDITS = {
    "LGA": ["nyc", "newyorkcity"],
    "JFK": ["nyc", "newyorkcity", "JFKAirport"],
    "EWR": ["newjersey", "Newark"],
    "ORD": ["chicago", "OHareAirport"],
    "LAX": ["LosAngeles", "LAX"],
    "SFO": ["sanfrancisco", "bayarea"],
    "ATL": ["Atlanta"],
    "DFW": ["Dallas", "FortWorth"],
    "DEN": ["Denver"],
    "SEA": ["Seattle"],
    "BOS": ["boston"],
    "MIA": ["Miami"],
}

# Patterns used for extracting explicit wait times from text (for display only)
WAIT_TIME_PATTERNS = [
    r'\b(\d+)\s*(min|minute|mins|minutes|hr|hour|hours)\b',
]

# LLM backend: use Anthropic API if ANTHROPIC_API_KEY is set, else claude -p
CLAUDE_CMD = os.path.expanduser("~/.local/bin/claude")
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"


def _llm(prompt, input_text=None):
    """Call an LLM with the given prompt. Uses Anthropic API if key is set, else claude -p."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        return _anthropic_api(prompt, input_text, api_key)
    else:
        return _claude_cli(prompt, input_text)


def _anthropic_api(prompt, input_text, api_key):
    """Call the Anthropic Messages API directly via urllib."""
    user_content = prompt
    if input_text:
        user_content = f"{prompt}\n\n---\n\n{input_text}"

    payload = json.dumps({
        "model": ANTHROPIC_MODEL,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": user_content}],
    }).encode()

    req = urllib.request.Request(
        ANTHROPIC_API_URL,
        data=payload,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            # Extract text from the response
            for block in data.get("content", []):
                if block.get("type") == "text":
                    return block["text"]
            return None
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as e:
        print(f"  [warn] Anthropic API failed: {e}", file=sys.stderr)
        return None


def _claude_cli(prompt, input_text=None):
    """Run claude -p with the given prompt. Returns the text response."""
    cmd = [CLAUDE_CMD, "-p", prompt, "--output-format", "json"]
    try:
        result = subprocess.run(
            cmd,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            print(f"  [warn] claude -p failed: {result.stderr[:200]}", file=sys.stderr)
            return None
        data = json.loads(result.stdout)
        return data.get("result", result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError) as e:
        print(f"  [warn] claude -p error: {e}", file=sys.stderr)
        return None


def llm_filter_posts(posts, airport_code, terminal=None):
    """Use claude -p to filter posts to only those with actionable wait time info."""
    if not posts:
        return []

    # Build a compact representation for the LLM
    items = []
    for i, p in enumerate(posts):
        text = f"{p.get('title', '')} {p.get('body', '')}".strip()
        if len(text) > 400:
            text = text[:400]
        items.append({"id": i, "text": text})

    terminal_note = f" Terminal {terminal.upper()}" if terminal else ""
    prompt = f"""You are filtering social media posts about TSA wait times at {airport_code}{terminal_note}.

For each post, decide: INCLUDE or EXCLUDE.

INCLUDE posts where someone reports firsthand wait time observations:
- Explicit wait times ("took 2 hours", "45 min precheck", "got through in 20 minutes")
- Specific line length descriptions ("line wrapped around the terminal twice", "line goes back to the rideshare area")
- Personal reports of how long they waited or how long it took to get through

EXCLUDE:
- News articles about TSA funding, politics, or policy
- Political commentary mentioning TSA or the airport
- Travel waivers, airport closures, or flight delay announcements
- General complaints without specific wait information ("lines are insane" with no detail)
- Posts about plane crashes, incidents, or safety issues
- Posts asking questions about wait times without reporting any

Respond with ONLY a JSON array of the numeric IDs to INCLUDE. Example: [0, 3, 7]
If none should be included, respond with: []"""

    input_text = json.dumps(items)
    response = _llm(prompt, input_text)
    if response is None:
        # Fallback: include everything (degrade gracefully)
        print("  [warn] LLM filter failed, including all posts", file=sys.stderr)
        return posts

    # Parse the response — extract JSON array
    try:
        # The response might have markdown or extra text, find the array
        match = re.search(r'\[[\d\s,]*\]', response)
        if match:
            include_ids = set(json.loads(match.group()))
        else:
            print(f"  [warn] Could not parse LLM filter response, including all", file=sys.stderr)
            return posts
    except (json.JSONDecodeError, ValueError):
        print(f"  [warn] Could not parse LLM filter response, including all", file=sys.stderr)
        return posts

    filtered = [p for i, p in enumerate(posts) if i in include_ids]
    print(f"  LLM filter: {len(posts)} → {len(filtered)} posts", file=sys.stderr)
    return filtered


def llm_generate_summary(posts, airport_code, terminal=None):
    """Use claude -p to generate an HTML summary of the wait time reports."""
    if not posts:
        return "<p>No wait time reports found for this period.</p>"

    # Build context
    items = []
    for p in posts:
        text = f"{p.get('title', '')} {p.get('body', '')}".strip()
        if len(text) > 500:
            text = text[:500]
        ts = datetime.fromisoformat(p["timestamp"]).astimezone()
        terminal_info = ""
        if p.get("detected_terminals"):
            terminal_info = f" [Terminal {', '.join(p['detected_terminals'])}]"
        items.append(f"[{ts.strftime('%a %b %d %I:%M %p')}]{terminal_info} {text}")

    terminal_note = f" Terminal {terminal.upper()}" if terminal else ""
    prompt = f"""You are writing a brief summary of TSA wait time reports at {airport_code}{terminal_note} for a traveler.

Write 2-4 short paragraphs in HTML (use <p> tags only, no headings). Be direct and practical:

1. Lead with the bottom line: what should someone expect right now? Give specific time ranges.
2. Break out by terminal if the data supports it. Note which terminal is better/worse.
3. Note PreCheck vs general line differences if reported.
4. Note any time-of-day patterns (early morning vs afternoon).
5. End with a practical recommendation (how early to arrive).

IMPORTANT: Every time you reference a report or data point, include the specific day and time in parentheses. Never use relative terms like "this morning," "recent," "earlier today," or "yesterday" without also stating the exact date and time. Examples:
- "Terminal B waits hit 2 hours (Tue Mar 24, 6:08 AM)"
- "PreCheck was 20 minutes (Sun Mar 22, 1:51 PM)"
- "Reports from Sunday afternoon through Tuesday morning (Sun Mar 22 1 PM – Tue Mar 24 7 AM) show..."

Be concise. Use bold (<b>) for key numbers. Do not editorialize about politics or TSA funding — just report what travelers are seeing on the ground."""

    input_text = "\n".join(items)
    response = _llm(prompt, input_text)
    if response is None:
        return "<p>Summary unavailable.</p>"

    # Clean up — the response should be HTML paragraphs
    # Strip any markdown code fences if present
    response = re.sub(r'```html\s*', '', response)
    response = re.sub(r'```\s*', '', response)
    return response.strip()


def _request(url, headers=None):
    """Make an HTTP GET request with a browser-like User-Agent."""
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (tsa-watch/1.0)",
        **(headers or {}),
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as e:
        print(f"  [warn] Request failed for {url[:80]}: {e}", file=sys.stderr)
        return None


def search_reddit(airport_code, terminal=None, hours=24):
    """Search Reddit for recent TSA/airport posts."""
    results = []
    names = AIRPORT_NAMES.get(airport_code, [airport_code])
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

    # Combine general subs with airport-specific local subs
    all_subs = list(SUBREDDITS) + AIRPORT_SUBREDDITS.get(airport_code, [])
    # Deduplicate while preserving order
    seen_subs = set()
    unique_subs = []
    for s in all_subs:
        if s.lower() not in seen_subs:
            seen_subs.add(s.lower())
            unique_subs.append(s)

    for subreddit in unique_subs:
        for name in names[:2]:  # limit to avoid rate limiting
            query = urllib.parse.quote(f"{name} TSA security")
            url = (
                f"https://www.reddit.com/r/{subreddit}/search.json"
                f"?q={query}&sort=new&restrict_sr=on&t=week&limit=25"
            )
            data = _request(url)
            if not data or "data" not in data:
                continue

            for post in data["data"].get("children", []):
                p = post.get("data", {})
                created = datetime.fromtimestamp(p.get("created_utc", 0), tz=timezone.utc)
                if created < cutoff:
                    continue

                title = p.get("title", "")
                body = p.get("selftext", "")
                text = f"{title} {body}".lower()

                # Check if it mentions the airport
                if not any(n.lower() in text for n in names):
                    continue

                results.append({
                    "source": "reddit",
                    "subreddit": f"r/{subreddit}",
                    "title": title,
                    "body": body[:500] if body else "",
                    "url": f"https://reddit.com{p.get('permalink', '')}",
                    "timestamp": created.isoformat(),
                    "timestamp_local": created.astimezone().strftime("%a %b %d %I:%M %p"),
                    "score": p.get("score", 0),
                    "num_comments": p.get("num_comments", 0),
                    "relevance": 0,
                    "terminal_match": _terminal_match(text, terminal),
                })

    # Deduplicate by URL
    seen = set()
    deduped = []
    for r in results:
        if r["url"] not in seen:
            seen.add(r["url"])
            deduped.append(r)

    return deduped


def search_reddit_comments(airport_code, terminal=None, hours=24):
    """Search Reddit comments (via search API) for more granular reports."""
    results = []
    names = AIRPORT_NAMES.get(airport_code, [airport_code])
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

    for name in names[:2]:
        query = urllib.parse.quote(f"{name} TSA wait")
        url = (
            f"https://www.reddit.com/search.json"
            f"?q={query}&sort=new&type=comment&t=week&limit=25"
        )
        data = _request(url)
        if not data or "data" not in data:
            continue

        for comment in data["data"].get("children", []):
            c = comment.get("data", {})
            created = datetime.fromtimestamp(c.get("created_utc", 0), tz=timezone.utc)
            if created < cutoff:
                continue

            body = c.get("body", "")
            text = body.lower()

            if not any(n.lower() in text for n in names):
                continue

            permalink = c.get("permalink", "")
            results.append({
                "source": "reddit_comment",
                "subreddit": f"r/{c.get('subreddit', '?')}",
                "title": f"Comment in r/{c.get('subreddit', '?')}",
                "body": body[:500],
                "url": f"https://reddit.com{permalink}" if permalink else "",
                "timestamp": created.isoformat(),
                "timestamp_local": created.astimezone().strftime("%a %b %d %I:%M %p"),
                "score": c.get("score", 0),
                "num_comments": 0,
                "relevance": 0,
                "terminal_match": _terminal_match(text, terminal),
            })

    return results


def search_bluesky(airport_code, terminal=None, hours=24):
    """Search Bluesky public API for airport/TSA posts."""
    results = []
    names = AIRPORT_NAMES.get(airport_code, [airport_code])
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

    for name in names[:2]:
        query = urllib.parse.quote(f"{name} TSA")
        url = f"https://api.bsky.app/xrpc/app.bsky.feed.searchPosts?q={query}&sort=latest&limit=25"
        data = _request(url, headers={"Accept": "application/json"})
        if not data:
            # Bluesky public search may require auth or be rate-limited
            continue
        if not data or "posts" not in data:
            continue

        for post in data["posts"]:
            record = post.get("record", {})
            created_str = record.get("createdAt", "")
            try:
                created = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                continue

            if created < cutoff:
                continue

            text = record.get("text", "")
            text_lower = text.lower()

            author = post.get("author", {})
            handle = author.get("handle", "unknown")

            if not any(n.lower() in text_lower for n in names):
                continue

            # Skip trending/bot accounts
            if any(skip in handle for skip in ["nowbreezing", "trendsbot", "trending"]):
                continue

            uri = post.get("uri", "")
            # Convert AT URI to web URL
            rkey = uri.split("/")[-1] if uri else ""
            web_url = f"https://bsky.app/profile/{handle}/post/{rkey}" if rkey else ""

            results.append({
                "source": "bluesky",
                "subreddit": f"@{handle}",
                "title": text[:100],
                "body": text[:500],
                "url": web_url,
                "timestamp": created.isoformat(),
                "timestamp_local": created.astimezone().strftime("%a %b %d %I:%M %p"),
                "score": post.get("likeCount", 0),
                "num_comments": post.get("replyCount", 0),
                "relevance": 0,
                "terminal_match": _terminal_match(text_lower, terminal),
            })

    return results


def _terminal_match(text, terminal):
    """Check if text mentions a specific terminal."""
    if not terminal:
        return None
    terminal = terminal.upper()
    patterns = [
        rf'\bterminal\s*{terminal}\b',
        rf'\bterm\.?\s*{terminal}\b',
        rf'\bt{terminal}\b',
        rf'\bconcourse\s*{terminal}\b',
    ]
    return any(re.search(p, text, re.I) for p in patterns)


def _detect_terminal(text):
    """Detect which terminal(s) are mentioned in text. Returns a set."""
    terminals = set()
    for label in ["A", "B", "C", "D", "E", "F", "1", "2", "3", "4", "5", "6", "7", "8"]:
        patterns = [
            rf'\bterminal\s*{label}\b',
            rf'\bterm\.?\s*{label}\b',
            rf'\bconcourse\s*{label}\b',
        ]
        if any(re.search(p, text, re.I) for p in patterns):
            terminals.add(label)
    return terminals


def extract_wait_times(text):
    """Try to extract explicit wait time mentions from text."""
    times = []
    for m in re.finditer(r'(\d+)\s*(min|minute|mins|minutes)', text, re.I):
        times.append(int(m.group(1)))
    for m in re.finditer(r'(\d+)\s*(hr|hour|hours)', text, re.I):
        times.append(int(m.group(1)) * 60)
    return times


def _summarize_terminal(posts):
    """Generate a bullet-point summary for a set of posts about one terminal."""
    all_times = []
    precheck_times = []
    general_times = []
    for r in posts:
        text = f"{r['title']} {r['body']}".lower()
        times = extract_wait_times(f"{r['title']} {r['body']}")
        all_times.extend(times)
        if "precheck" in text or "pre-check" in text or "pre check" in text:
            precheck_times.extend(times)
        elif times:
            general_times.extend(times)

    lines = []
    if general_times:
        lines.append(f"- General line: {min(general_times)}-{max(general_times)} min reported "
                      f"({len(general_times)} data points)")
    if precheck_times:
        lines.append(f"- PreCheck: {min(precheck_times)}-{max(precheck_times)} min reported "
                      f"({len(precheck_times)} data points)")
    if not all_times:
        lines.append("- No explicit wait times reported; see posts below for qualitative descriptions")

    return "\n".join(lines)


def format_results(results, airport_code, terminal=None):
    """Format results grouped by day, then by terminal."""
    if not results:
        print(f"\nNo reports found for {airport_code}" +
              (f" Terminal {terminal}" if terminal else "") + ".\n")
        return

    # Detect terminals for each result
    for r in results:
        text = f"{r['title']} {r['body']}".lower()
        detected = _detect_terminal(text)
        if detected:
            r["detected_terminals"] = detected
        elif r.get("terminal_match"):
            r["detected_terminals"] = {terminal.upper()} if terminal else set()
        else:
            r["detected_terminals"] = set()

    # Parse dates and group by day
    from collections import defaultdict
    by_day = defaultdict(list)
    for r in results:
        ts = datetime.fromisoformat(r["timestamp"])
        day_key = ts.astimezone().strftime("%A %b %d, %Y")
        r["_day_key"] = day_key
        r["_sort_ts"] = ts
        by_day[day_key].append(r)

    # Sort days most recent first
    sorted_days = sorted(by_day.keys(),
                         key=lambda d: by_day[d][0]["_sort_ts"],
                         reverse=True)

    # Header
    header = f"# TSA Watch: {airport_code}"
    if terminal:
        header += f" (focus: Terminal {terminal.upper()})"
    print(header)
    print(f"{len(results)} reports across {len(sorted_days)} days\n")

    # All-up wait time summary
    all_times = []
    for r in results:
        all_times.extend(extract_wait_times(f"{r['title']} {r['body']}"))
    if all_times:
        print(f"**Overall reported wait times:** {min(all_times)}-{max(all_times)} min "
              f"(avg {sum(all_times)//len(all_times)} min, {len(all_times)} data points)\n")

    for day in sorted_days:
        day_posts = sorted(by_day[day], key=lambda r: r["_sort_ts"], reverse=True)
        print(f"---\n## {day}\n")

        # Group by terminal within the day
        terminal_groups = defaultdict(list)
        no_terminal = []
        for r in day_posts:
            if r["detected_terminals"]:
                for t in r["detected_terminals"]:
                    terminal_groups[t].append(r)
            else:
                no_terminal.append(r)

        # Sort terminals: prioritized terminal first, then alphabetical
        sorted_terminals = sorted(terminal_groups.keys(),
                                  key=lambda t: (0 if terminal and t == terminal.upper() else 1, t))

        for term in sorted_terminals:
            posts = terminal_groups[term]
            star = " ⭐" if terminal and term == terminal.upper() else ""
            print(f"### Terminal {term}{star}\n")
            print(_summarize_terminal(posts))
            print()
            _print_posts(posts)

        if no_terminal:
            print(f"### General / Terminal not specified\n")
            print(_summarize_terminal(no_terminal))
            print()
            _print_posts(no_terminal)


def _print_posts(posts):
    """Print a list of posts with URLs."""
    for r in posts:
        source_tag = r["source"].upper().replace("_", " ")
        time_str = datetime.fromisoformat(r["timestamp"]).astimezone().strftime("%I:%M %p")

        # Build one-line summary
        if r["source"] == "reddit" and r["title"]:
            summary = r["title"]
        else:
            body = r["body"].replace("\n", " ").strip()
            summary = body[:150] + "..." if len(body) > 150 else body

        wait_times = extract_wait_times(f"{r['title']} {r['body']}")
        wait_str = ""
        if wait_times:
            wait_str = f" — ⏱️ {', '.join(str(t) + 'm' for t in wait_times)}"

        score_str = ""
        if r["score"] > 0:
            score_str = f" ({r['score']}↑)"

        print(f"- **{time_str}** [{source_tag}]{score_str} {summary}{wait_str}")
        print(f"  {r['url']}")
    print()


def format_html(results, airport_code, terminal=None, summary_html=None):
    """Format results as a self-contained HTML file grouped by day, then terminal."""
    from html import escape
    from collections import defaultdict

    if not results:
        return f"<html><body><p>No reports found for {airport_code}.</p></body></html>"

    # Detect terminals for each result
    for r in results:
        text = f"{r['title']} {r['body']}".lower()
        detected = _detect_terminal(text)
        if detected:
            r["detected_terminals"] = detected
        elif r.get("terminal_match"):
            r["detected_terminals"] = {terminal.upper()} if terminal else set()
        else:
            r["detected_terminals"] = set()

    # Parse dates and group by day
    by_day = defaultdict(list)
    for r in results:
        ts = datetime.fromisoformat(r["timestamp"])
        day_key = ts.astimezone().strftime("%A %b %d, %Y")
        r["_day_key"] = day_key
        r["_sort_ts"] = ts
        by_day[day_key].append(r)

    sorted_days = sorted(by_day.keys(),
                         key=lambda d: by_day[d][0]["_sort_ts"],
                         reverse=True)

    # Overall wait times
    all_times = []
    for r in results:
        all_times.extend(extract_wait_times(f"{r['title']} {r['body']}"))

    focus_label = f" (focus: Terminal {terminal.upper()})" if terminal else ""
    # Always display in US Eastern time (relevant for NYC airports)
    from zoneinfo import ZoneInfo
    eastern = ZoneInfo("America/New_York")
    now_str = datetime.now(eastern).strftime("%b %d, %Y %I:%M %p ET")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TSA Watch: {airport_code}{focus_label}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 720px; margin: 0 auto; padding: 20px; background: #f8f9fa; color: #1a1a1a; }}
  h1 {{ margin-bottom: 4px; }}
  .subtitle {{ color: #666; margin-bottom: 16px; }}
  .summary-box {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin-bottom: 24px; }}
  .summary-box .big {{ font-size: 1.4em; font-weight: 600; }}
  .day-header {{ background: #1a1a1a; color: #fff; padding: 10px 16px; border-radius: 6px; margin-top: 28px; margin-bottom: 12px; }}
  .terminal-section {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
  .terminal-section.focus {{ border-left: 4px solid #0066cc; }}
  .terminal-header {{ font-size: 1.1em; font-weight: 600; margin-bottom: 8px; }}
  .terminal-header .star {{ color: #f5a623; }}
  .wait-summary {{ color: #555; margin-bottom: 12px; font-size: 0.95em; }}
  .wait-summary li {{ margin-bottom: 2px; }}
  .post {{ padding: 8px 0; border-bottom: 1px solid #eee; }}
  .post:last-child {{ border-bottom: none; }}
  .post-time {{ font-weight: 600; color: #333; }}
  .post-source {{ display: inline-block; background: #e9ecef; border-radius: 3px; padding: 1px 6px; font-size: 0.8em; color: #555; margin-left: 4px; }}
  .post-source.reddit {{ background: #ff4500; color: #fff; }}
  .post-source.bluesky {{ background: #0085ff; color: #fff; }}
  .post-score {{ color: #888; font-size: 0.85em; }}
  .post-text {{ margin: 4px 0; }}
  .post-wait {{ background: #fff3cd; border-radius: 4px; padding: 2px 8px; font-size: 0.9em; font-weight: 500; display: inline-block; margin-top: 2px; }}
  .post-link {{ font-size: 0.85em; }}
  a {{ color: #0066cc; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<h1>TSA Watch: {escape(airport_code)}{escape(focus_label)}</h1>
<p class="subtitle"><b>Last updated: {now_str}</b> &mdash; {len(results)} reports across {len(sorted_days)} days</p>
"""

    if summary_html:
        html += f'<div class="summary-box">{summary_html}</div>\n'
    elif all_times:
        html += f"""<div class="summary-box">
<span class="big">{min(all_times)}&ndash;{max(all_times)} min</span> reported wait times
(avg {sum(all_times)//len(all_times)} min, {len(all_times)} data points)
</div>
"""

    for day in sorted_days:
        day_posts = sorted(by_day[day], key=lambda r: r["_sort_ts"], reverse=True)
        html += f'<div class="day-header">{escape(day)}</div>\n'

        terminal_groups = defaultdict(list)
        no_terminal = []
        for r in day_posts:
            if r["detected_terminals"]:
                for t in r["detected_terminals"]:
                    terminal_groups[t].append(r)
            else:
                no_terminal.append(r)

        sorted_terminals = sorted(terminal_groups.keys(),
                                  key=lambda t: (0 if terminal and t == terminal.upper() else 1, t))

        for term in sorted_terminals:
            posts = terminal_groups[term]
            is_focus = terminal and term == terminal.upper()
            focus_cls = " focus" if is_focus else ""
            star = ' <span class="star">★</span>' if is_focus else ""
            html += f'<div class="terminal-section{focus_cls}">\n'
            html += f'<div class="terminal-header">Terminal {escape(term)}{star}</div>\n'
            html += _html_summary(posts)
            html += _html_posts(posts)
            html += '</div>\n'

        if no_terminal:
            html += '<div class="terminal-section">\n'
            html += '<div class="terminal-header">General / Terminal not specified</div>\n'
            html += _html_summary(no_terminal)
            html += _html_posts(no_terminal)
            html += '</div>\n'

    html += "</body></html>"
    return html


def _html_summary(posts):
    """Generate HTML summary bullets for a terminal group."""
    from html import escape
    all_times = []
    precheck_times = []
    general_times = []
    for r in posts:
        text = f"{r['title']} {r['body']}".lower()
        times = extract_wait_times(f"{r['title']} {r['body']}")
        all_times.extend(times)
        if "precheck" in text or "pre-check" in text or "pre check" in text:
            precheck_times.extend(times)
        elif times:
            general_times.extend(times)

    html = '<ul class="wait-summary">\n'
    if general_times:
        html += f'<li>General line: {min(general_times)}&ndash;{max(general_times)} min ({len(general_times)} data points)</li>\n'
    if precheck_times:
        html += f'<li>PreCheck: {min(precheck_times)}&ndash;{max(precheck_times)} min ({len(precheck_times)} data points)</li>\n'
    if not all_times:
        html += '<li>No explicit wait times; see posts below</li>\n'
    html += '</ul>\n'
    return html


def _html_posts(posts):
    """Generate HTML for a list of posts."""
    from html import escape
    html = ""
    for r in posts:
        source = r["source"].replace("_", " ")
        source_cls = "reddit" if "reddit" in r["source"] else "bluesky"
        time_str = datetime.fromisoformat(r["timestamp"]).astimezone().strftime("%I:%M %p")

        if r["source"] == "reddit" and r["title"]:
            summary = r["title"]
        else:
            body = r["body"].replace("\n", " ").strip()
            summary = body[:200] + "..." if len(body) > 200 else body

        wait_times = extract_wait_times(f"{r['title']} {r['body']}")
        wait_html = ""
        if wait_times:
            wait_html = f' <span class="post-wait">⏱️ {", ".join(str(t) + "m" for t in wait_times)}</span>'

        score_html = ""
        if r["score"] > 0:
            score_html = f' <span class="post-score">({r["score"]}↑)</span>'

        html += f"""<div class="post">
<span class="post-time">{escape(time_str)}</span>
<span class="post-source {source_cls}">{escape(source.upper())}</span>{score_html}
<div class="post-text">{escape(summary)}{wait_html}</div>
<div class="post-link"><a href="{escape(r['url'])}" target="_blank">{escape(r['url'])}</a></div>
</div>
"""
    return html


def main():
    parser = argparse.ArgumentParser(
        description="Monitor TSA wait times from social media",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 scripts/tsa_watch.py LGA --terminal C --hours 24
  python3 scripts/tsa_watch.py JFK --hours 12
  python3 scripts/tsa_watch.py LGA --terminal C --hours 48 --html
        """,
    )
    parser.add_argument("airport", help="Airport code (e.g., LGA, JFK, ORD)")
    parser.add_argument("--terminal", "-t", help="Terminal letter/number (e.g., C, 1)")
    parser.add_argument("--hours", "-H", type=int, default=24, help="Look back N hours (default: 24)")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    parser.add_argument("--html", action="store_true", help="Output HTML file and open in browser")

    args = parser.parse_args()
    airport = args.airport.upper()

    print(f"Searching for TSA reports at {airport}" +
          (f" Terminal {args.terminal.upper()}" if args.terminal else "") +
          f" (last {args.hours}h)...", file=sys.stderr)

    # Gather from all sources, tracking counts
    stats = {"last_run": datetime.now(timezone.utc).isoformat()}

    print("  Searching Reddit posts...", file=sys.stderr)
    reddit_posts = search_reddit(airport, args.terminal, args.hours)
    stats["reddit_posts"] = len(reddit_posts)

    print("  Searching Reddit comments...", file=sys.stderr)
    reddit_comments = search_reddit_comments(airport, args.terminal, args.hours)
    stats["reddit_comments"] = len(reddit_comments)

    print("  Searching Bluesky...", file=sys.stderr)
    bluesky_posts = search_bluesky(airport, args.terminal, args.hours)
    stats["bluesky"] = len(bluesky_posts)

    all_results = reddit_posts + reddit_comments + bluesky_posts
    stats["total_raw"] = len(all_results)

    print(f"  Found {len(all_results)} total reports "
          f"(reddit posts: {stats['reddit_posts']}, "
          f"reddit comments: {stats['reddit_comments']}, "
          f"bluesky: {stats['bluesky']}).", file=sys.stderr)

    if args.json:
        print(json.dumps(all_results, indent=2))
        return

    # LLM filter: keep only posts with actionable wait time info
    print("  Filtering with LLM...", file=sys.stderr)
    all_results = llm_filter_posts(all_results, airport, args.terminal)
    stats["after_llm_filter"] = len(all_results)

    # Write stats
    stats_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stats.json")
    # Also check for stats.json in cwd (for GitHub Actions where script is in repo root)
    if not os.path.exists(os.path.dirname(stats_path)):
        stats_path = "stats.json"
    try:
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  Wrote {stats_path}", file=sys.stderr)
    except OSError:
        # Try cwd as fallback
        with open("stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        print("  Wrote stats.json", file=sys.stderr)

    if args.html:
        # Generate LLM summary
        print("  Generating summary...", file=sys.stderr)
        # Need to detect terminals before summary (summary uses them)
        for r in all_results:
            text = f"{r['title']} {r['body']}".lower()
            r["detected_terminals"] = _detect_terminal(text) or set()
        summary_html = llm_generate_summary(all_results, airport, args.terminal)
        html = format_html(all_results, airport, args.terminal, summary_html=summary_html)
        out_path = f"/tmp/tsa-watch-{airport.lower()}.html"
        with open(out_path, "w") as f:
            f.write(html)
        print(f"  Wrote {out_path}", file=sys.stderr)
        subprocess.run(["open", out_path])
    else:
        format_results(all_results, airport, args.terminal)


if __name__ == "__main__":
    main()
