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

def _eastern():
    """Return the US/Eastern timezone."""
    from zoneinfo import ZoneInfo
    return ZoneInfo("America/New_York")


# Common airport name mappings for search broadening
AIRPORT_NAMES = {
    "ATL": ["ATL", "Atlanta", "Hartsfield"],
    "DFW": ["DFW", "Dallas Fort Worth", "DFW Airport"],
    "DEN": ["DEN", "Denver", "Denver International"],
    "ORD": ["O'Hare", "OHare", "ORD"],
    "LAX": ["LAX", "Los Angeles"],
    "JFK": ["JFK", "Kennedy", "John F Kennedy"],
    "CLT": ["CLT", "Charlotte", "Charlotte Douglas"],
    "LAS": ["LAS", "Las Vegas", "Harry Reid"],
    "MCO": ["MCO", "Orlando", "Orlando International"],
    "MIA": ["MIA", "Miami"],
    "LGA": ["LaGuardia", "La Guardia", "LGA"],
    "EWR": ["Newark", "EWR", "Newark Liberty"],
    # Additional airports (for local/manual use)
    "SFO": ["SFO", "San Francisco"],
    "SEA": ["SEA", "Seattle", "SeaTac"],
    "BOS": ["BOS", "Boston", "Logan"],
    "TYS": ["TYS", "Knoxville", "McGhee Tyson"],
}

# Subreddits searched for every airport
SUBREDDITS = [
    # TSA / security
    "TSA", "tsaprecheck", "GlobalEntry",
    # Airlines
    "delta", "americanairlines", "unitedairlines", "jetblue",
    "SouthwestAirlines", "AlaskaAirlines", "FrontierAirlines", "spiritair",
    # General travel/flying
    "flying", "Flights", "travel", "airports",
]

# Additional subs searched only for the specific airport
AIRPORT_SUBREDDITS = {
    "ATL": ["Atlanta", "ATLairport"],
    "DFW": ["Dallas", "FortWorth", "dfwarea"],
    "DEN": ["Denver", "DENairport", "DIA"],
    "ORD": ["chicago", "OHareAirport"],
    "LAX": ["LosAngeles", "LAX"],
    "JFK": ["nyc", "newyorkcity", "JFKAirport"],
    "CLT": ["Charlotte", "CharlotteAirport"],
    "LAS": ["LasVegas", "vegas"],
    "MCO": ["orlando"],
    "MIA": ["Miami"],
    "LGA": ["nyc", "newyorkcity", "astoria", "Queens", "LGAairport"],
    "EWR": ["newjersey", "Newark"],
    "SFO": ["sanfrancisco", "bayarea"],
    "SEA": ["Seattle"],
    "BOS": ["boston"],
}

# Full display names for landing page
AIRPORT_DISPLAY = {
    "ATL": "Atlanta (Hartsfield-Jackson)",
    "DFW": "Dallas/Fort Worth",
    "DEN": "Denver International",
    "ORD": "Chicago O'Hare",
    "LAX": "Los Angeles (LAX)",
    "JFK": "New York JFK",
    "CLT": "Charlotte Douglas",
    "LAS": "Las Vegas (Harry Reid)",
    "MCO": "Orlando International",
    "MIA": "Miami International",
    "LGA": "New York LaGuardia",
    "EWR": "Newark Liberty",
}

# Default set of airports for multi-airport mode
DEFAULT_AIRPORTS = ["ATL", "DFW", "DEN", "ORD", "LAX", "JFK", "CLT", "LAS", "MCO", "MIA", "LGA", "EWR"]

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

    # Build context — sort newest first so LLM sees most recent data first
    from zoneinfo import ZoneInfo
    eastern = ZoneInfo("America/New_York")
    posts_sorted = sorted(posts, key=lambda p: p["timestamp"], reverse=True)
    items = []
    for p in posts_sorted:
        text = f"{p.get('title', '')} {p.get('body', '')}".strip()
        if len(text) > 500:
            text = text[:500]
        ts = datetime.fromisoformat(p["timestamp"]).astimezone(eastern)
        terminal_info = ""
        if p.get("detected_terminals"):
            terminal_info = f" [Terminal {', '.join(p['detected_terminals'])}]"
        source = p.get("source", "unknown").capitalize()
        url = p.get("url", "")
        items.append(f"[{source}: {ts.strftime('%a %b %d, %I:%M %p ET')}]({url}){terminal_info} {text}")

    terminal_note = f" Terminal {terminal.upper()}" if terminal else ""
    today_str = datetime.now(eastern).strftime("%a %b %d, %Y")
    prompt = f"""You are writing a brief summary of TSA wait time reports at {airport_code}{terminal_note} for a traveler. Today's date is {today_str}.

Write in HTML (use <p> tags only, no headings). Structure the summary as follows:

1. **Terminal-specific data first.** Group reports by terminal (e.g., Terminal B, Terminal 1). Within each terminal group, present the most recent reports first — newest data is the most valuable. Include specific wait times and note differences between PreCheck, CLEAR, and regular lines when reported.

2. **General (non-terminal-specific) reports next.** After the terminal-specific paragraphs, add a separate paragraph for any reports that don't mention a specific terminal. Again, newest first.

3. **Practical recommendations last.** End with a short recommendation on how early to arrive. Make it terminal-specific if the data supports it (e.g., "Terminal C lines are moving faster than Terminal B"). Note any differences between PreCheck/CLEAR and regular lines in your recommendation.

Within all sections, always present newer information before older information.

IMPORTANT CITATION FORMAT: Every time you reference a report or data point, place the citation in parentheses AFTER the claim, with the entire "Source - Day Mon DD, H:MM PM" text linked to the original post URL. The input data is formatted as [Source: Day Mon DD, HH:MM AM/PM ET](url). Use this to build linked citations.

Correct citation style — source and time in parentheses after the fact, linked together:
- "PreCheck was approximately <b>30 minutes</b> during lunchtime (<a href="https://bsky.app/..." target="_blank">Bluesky - Tue Mar 24, 1:24 PM</a>)"
- "a traveler waited <b>1 hour 37 minutes</b> in the regular security line (<a href="https://twitter.com/..." target="_blank">Twitter - Tue Mar 24, 11:44 AM</a>)"

WRONG — do NOT lead sentences with "On Source, Date":
- WRONG: "On Bluesky, Tue Mar 24, 1:24 PM, PreCheck was 30 minutes"
- WRONG: "On Twitter, Tue Mar 24, a traveler reported..."

Also: if a report's date matches today's date, say "today" naturally in the prose (e.g., "PreCheck was 30 minutes today") but still include the full date in the parenthetical citation. Never say "that same day" or "earlier that same day" — say "today" or "earlier today" when the date is today.

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
                    "timestamp_local": created.astimezone(_eastern()).strftime("%a %b %d %I:%M %p"),
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
                "timestamp_local": created.astimezone(_eastern()).strftime("%a %b %d %I:%M %p"),
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
                "timestamp_local": created.astimezone(_eastern()).strftime("%a %b %d %I:%M %p"),
                "score": post.get("likeCount", 0),
                "num_comments": post.get("replyCount", 0),
                "relevance": 0,
                "terminal_match": _terminal_match(text_lower, terminal),
            })

    return results


def search_twitter(airport_code, terminal=None, hours=24):
    """Search X/Twitter for recent TSA/airport posts via the v2 API."""
    bearer = os.environ.get("TWITTER_BEARER_TOKEN")
    if not bearer:
        print("  [skip] TWITTER_BEARER_TOKEN not set", file=sys.stderr)
        return []

    results = []
    names = AIRPORT_NAMES.get(airport_code, [airport_code])
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

    for name in names[:2]:
        query = urllib.parse.quote(f"{name} TSA -is:retweet lang:en")
        url = (
            f"https://api.x.com/2/tweets/search/recent"
            f"?query={query}&max_results=25"
            f"&tweet.fields=created_at,public_metrics,text"
            f"&expansions=author_id&user.fields=username"
        )
        req = urllib.request.Request(url, headers={
            "Authorization": f"Bearer {bearer}",
        })
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as e:
            print(f"  [warn] Twitter search failed: {e}", file=sys.stderr)
            continue

        # Build user lookup
        users = {}
        for u in data.get("includes", {}).get("users", []):
            users[u["id"]] = u["username"]

        for tweet in data.get("data", []):
            created_str = tweet.get("created_at", "")
            try:
                created = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                continue

            if created < cutoff:
                continue

            text = tweet.get("text", "")
            text_lower = text.lower()

            if not any(n.lower() in text_lower for n in names):
                continue

            author = users.get(tweet.get("author_id"), "unknown")
            metrics = tweet.get("public_metrics", {})
            tweet_id = tweet.get("id", "")
            web_url = f"https://x.com/{author}/status/{tweet_id}" if tweet_id else ""

            results.append({
                "source": "twitter",
                "subreddit": f"@{author}",
                "title": "",
                "body": text[:500],
                "url": web_url,
                "timestamp": created.isoformat(),
                "timestamp_local": created.astimezone(_eastern()).strftime("%a %b %d %I:%M %p"),
                "score": metrics.get("like_count", 0),
                "num_comments": metrics.get("reply_count", 0),
                "relevance": 0,
                "terminal_match": _terminal_match(text_lower, terminal),
            })

    # Deduplicate by tweet ID
    seen = set()
    deduped = []
    for r in results:
        if r["url"] not in seen:
            seen.add(r["url"])
            deduped.append(r)

    return deduped


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
        time_str = datetime.fromisoformat(r["timestamp"]).astimezone(_eastern()).strftime("%I:%M %p")

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


def format_html(results, airport_code, terminal=None, summary_html=None, archive_files=None):
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
<script async src="https://www.googletagmanager.com/gtag/js?id=G-6BJC8T5KKQ"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){{dataLayer.push(arguments);}}
  gtag('js', new Date());
  gtag('config', 'G-6BJC8T5KKQ');
</script>
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<meta property="og:title" content="TSA Wait Times: {airport_code} — Terminal Watch">
<meta property="og:description" content="Crowdsourced TSA wait times at {airport_code}. Updated hourly from Bluesky and Twitter.">
<meta property="og:image" content="https://terminalwatch.info/og-image.png">
<meta property="og:url" content="https://terminalwatch.info/{airport_code.lower()}/">
<meta property="og:type" content="website">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="TSA Wait Times: {airport_code} — Terminal Watch">
<meta name="twitter:description" content="Crowdsourced TSA wait times at {airport_code}. Updated hourly from Bluesky and Twitter.">
<meta name="twitter:image" content="https://terminalwatch.info/og-image.png">
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
  .post-source {{ display: inline-block; background: #e9ecef; border-radius: 3px; padding: 1px 6px; font-size: 0.8em; color: #555; margin-left: 4px; text-decoration: none; }}
  .post-source:hover {{ opacity: 0.85; text-decoration: none; }}
  .post-source.reddit {{ background: #ff4500; color: #fff; }}
  .post-source.bluesky {{ background: #0085ff; color: #fff; }}
  .post-source.twitter {{ background: #000; color: #fff; }}
  .post-handle {{ font-size: 0.8em; color: #666; margin-left: 4px; }}
  .post-profile {{ font-size: 0.9em; color: #888; }}
  .post-text {{ margin: 4px 0; }}
  .post-wait {{ background: #fff3cd; border-radius: 4px; padding: 2px 8px; font-size: 0.9em; font-weight: 500; display: inline-block; margin-top: 2px; }}
  .post-link {{ font-size: 0.85em; color: #888; }}
  a {{ color: #0066cc; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  .archive {{ font-size: 0.85em; color: #888; margin-bottom: 16px; line-height: 1.8; }}
  .archive-label {{ color: #666; font-weight: 500; }}
  .archive a {{ color: #888; }}
  .archive a:hover {{ color: #0066cc; }}
  .archive .sep {{ color: #ccc; margin: 0 4px; }}
</style>
</head>
<body>
<p style="margin-bottom:4px"><a href="../">&larr; All airports</a></p>
<h1>TSA Watch: {escape(airport_code)}{escape(focus_label)}</h1>
<p class="subtitle"><b>Last updated: {now_str}</b> &mdash; {len(results)} reports across {len(sorted_days)} days</p>
"""

    # Archive links — last 24 hours only, grouped by date
    if archive_files:
        from zoneinfo import ZoneInfo as ZI
        et = ZI("America/New_York")
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        # Parse and filter to last 24 hours
        parsed = []
        for af in archive_files:
            fname = os.path.basename(af).replace(".html", "")
            try:
                ts = datetime.strptime(fname, "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            if ts >= cutoff:
                parsed.append((ts, af))
        if parsed:
            # Group by date (in ET), most recent date first
            from collections import OrderedDict
            by_date = OrderedDict()
            for ts, af in sorted(parsed, key=lambda x: x[0], reverse=True):
                ts_et = ts.astimezone(et)
                date_label = ts_et.strftime("%b %-d")
                time_label = ts_et.strftime("%-I:%M %p")
                link = f'<a href="archive/{os.path.basename(af)}">{time_label}</a>'
                by_date.setdefault(date_label, []).append(link)
            sep = '<span class="sep">|</span>'
            lines = []
            for date_label, time_links in by_date.items():
                lines.append(f'<span class="archive-label">{date_label}:</span> {sep.join(time_links)}')
            html += f'<div class="archive">{"<br>".join(lines)}</div>\n'

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
        source_cls = "reddit" if "reddit" in r["source"] else "twitter" if r["source"] == "twitter" else "bluesky"
        time_str = datetime.fromisoformat(r["timestamp"]).astimezone(_eastern()).strftime("%I:%M %p")

        if r["source"] == "reddit" and r["title"]:
            summary = r["title"]
        else:
            body = r["body"].replace("\n", " ").strip()
            summary = body

        wait_times = extract_wait_times(f"{r['title']} {r['body']}")
        wait_html = ""
        if wait_times:
            wait_html = f' <span class="post-wait">⏱️ {", ".join(str(t) + "m" for t in wait_times)}</span>'

        # Build profile link and handle display
        handle_html = ""
        sub = r.get("subreddit", "")
        if r["source"] == "bluesky" and sub.startswith("@"):
            handle = sub[1:]  # strip @
            profile_url = f"https://bsky.app/profile/{handle}"
            handle_html = f' <span class="post-handle">{escape(sub)} <a href="{escape(profile_url)}" target="_blank" class="post-profile">(profile)</a></span>'
        elif r["source"] == "twitter" and sub.startswith("@"):
            handle = sub[1:]
            profile_url = f"https://x.com/{handle}"
            handle_html = f' <span class="post-handle">{escape(sub)} <a href="{escape(profile_url)}" target="_blank" class="post-profile">(profile)</a></span>'
        elif r["source"] == "reddit" and sub.startswith("r/"):
            profile_url = f"https://reddit.com/{sub}"
            handle_html = f' <span class="post-handle">{escape(sub)} <a href="{escape(profile_url)}" target="_blank" class="post-profile">(profile)</a></span>'

        # Post link goes at end of text, before wait time
        link_html = f' <a href="{escape(r["url"])}" target="_blank" class="post-link">(link)</a>'

        html += f"""<div class="post">
<span class="post-time">{escape(time_str)}</span>
<span class="post-source {source_cls}">{escape(source.upper())}</span>{handle_html}
<div class="post-text">{escape(summary)}{link_html}{wait_html}</div>
</div>
"""
    return html


def generate_landing_page(output_dir, airport_stats):
    """Generate the index.html landing page linking to all airport pages."""
    from html import escape
    from zoneinfo import ZoneInfo
    eastern = ZoneInfo("America/New_York")
    now_str = datetime.now(eastern).strftime("%b %d, %Y %I:%M %p ET")

    cards = []
    for code in DEFAULT_AIRPORTS:
        display = AIRPORT_DISPLAY.get(code, code)
        stats = airport_stats.get(code, {})
        report_count = stats.get("after_llm_filter", 0)
        badge = f"{report_count} reports" if report_count else "no reports"

        cards.append(f"""<a href="{code.lower()}/index.html" class="card">
<div class="card-code">{escape(code)}</div>
<div class="card-name">{escape(display)}</div>
<div class="card-badge">{escape(badge)}</div>
</a>""")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<script async src="https://www.googletagmanager.com/gtag/js?id=G-6BJC8T5KKQ"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){{dataLayer.push(arguments);}}
  gtag('js', new Date());
  gtag('config', 'G-6BJC8T5KKQ');
</script>
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<meta property="og:title" content="Terminal Watch — Crowdsourced TSA Wait Times">
<meta property="og:description" content="Real-time TSA wait times crowdsourced from Bluesky and Twitter. Updated hourly.">
<meta property="og:image" content="https://terminalwatch.info/og-image.png">
<meta property="og:url" content="https://terminalwatch.info">
<meta property="og:type" content="website">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="Terminal Watch — Crowdsourced TSA Wait Times">
<meta name="twitter:description" content="Real-time TSA wait times crowdsourced from Bluesky and Twitter. Updated hourly.">
<meta name="twitter:image" content="https://terminalwatch.info/og-image.png">
<title>Terminal Watch — Crowdsourced TSA Wait Times</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background: #f8f9fa; color: #1a1a1a; }}
  h1 {{ margin-bottom: 4px; }}
  .subtitle {{ color: #666; margin-bottom: 24px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 12px; }}
  .card {{ display: block; background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 16px; text-decoration: none; color: inherit; transition: border-color 0.15s; }}
  .card:hover {{ border-color: #0066cc; }}
  .card-code {{ font-size: 1.6em; font-weight: 700; color: #1a1a1a; }}
  .card-name {{ font-size: 0.9em; color: #555; margin: 4px 0 8px; }}
  .card-badge {{ font-size: 0.8em; color: #888; background: #f0f0f0; display: inline-block; padding: 2px 8px; border-radius: 3px; }}
  .footer {{ margin-top: 32px; font-size: 0.85em; color: #999; }}
</style>
</head>
<body>
<h1>Terminal Watch</h1>
<p style="font-size: 0.85em; color: #555; margin: 4px 0 8px;"><em>This is very much a beta web site.</em></p>
<p style="font-size: 0.85em; color: #555; margin: 4px 0 12px;">Want to support this project? <a href="https://donate.stripe.com/3cI6oIdl9bhKcLG7MI6AM00" target="_blank" style="color: #0066cc;">Make a donation of any size.</a></p>
<p class="subtitle"><b>Last updated: {now_str}</b> &mdash; Crowdsourced TSA wait times from Bluesky and Twitter</p>
<div class="grid">
{"".join(cards)}
</div>
<div class="footer">Updated hourly. Data sourced from public social media posts. <a href="/about/" style="color: #0066cc;">About this project</a></div>
</body>
</html>"""

    out_path = os.path.join(output_dir, "index.html")
    with open(out_path, "w") as f:
        f.write(html)
    print(f"  Wrote landing page: {out_path}", file=sys.stderr)


def run_single_airport(airport, terminal, hours, output_dir=None, archive_base=None):
    """Run the full pipeline for one airport. Returns stats dict."""
    import time

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"  {airport}" +
          (f" Terminal {terminal.upper()}" if terminal else "") +
          f" (last {hours}h)", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    stats = {
        "airport": airport,
        "last_run": datetime.now(timezone.utc).isoformat(),
    }

    print("  Searching Reddit posts...", file=sys.stderr)
    reddit_posts = search_reddit(airport, terminal, hours)
    stats["reddit_posts"] = len(reddit_posts)

    print("  Searching Reddit comments...", file=sys.stderr)
    reddit_comments = search_reddit_comments(airport, terminal, hours)
    stats["reddit_comments"] = len(reddit_comments)

    print("  Searching Bluesky...", file=sys.stderr)
    bluesky_posts = search_bluesky(airport, terminal, hours)
    stats["bluesky"] = len(bluesky_posts)

    print("  Searching X/Twitter...", file=sys.stderr)
    twitter_posts = search_twitter(airport, terminal, hours)
    stats["twitter"] = len(twitter_posts)

    all_results = reddit_posts + reddit_comments + bluesky_posts + twitter_posts
    stats["total_raw"] = len(all_results)

    print(f"  Found {len(all_results)} total "
          f"(reddit: {stats['reddit_posts']}+{stats['reddit_comments']}, "
          f"bluesky: {stats['bluesky']}, twitter: {stats['twitter']})", file=sys.stderr)

    # LLM filter
    print("  Filtering with LLM...", file=sys.stderr)
    all_results = llm_filter_posts(all_results, airport, terminal)
    stats["after_llm_filter"] = len(all_results)

    if output_dir:
        # Generate HTML
        print("  Generating summary...", file=sys.stderr)
        for r in all_results:
            text = f"{r['title']} {r['body']}".lower()
            r["detected_terminals"] = _detect_terminal(text) or set()
        summary_html = llm_generate_summary(all_results, airport, terminal)

        # Collect archive files
        archive_files = []
        airport_archive = os.path.join(archive_base or output_dir, "archive") if archive_base else None
        if airport_archive and os.path.isdir(airport_archive):
            archive_files = sorted(
                [f for f in os.listdir(airport_archive) if f.endswith(".html")],
                reverse=True,
            )

        html = format_html(all_results, airport, terminal,
                           summary_html=summary_html, archive_files=archive_files)

        # Write to airport subdirectory
        airport_dir = os.path.join(output_dir, airport.lower())
        os.makedirs(airport_dir, exist_ok=True)
        out_path = os.path.join(airport_dir, "index.html")
        with open(out_path, "w") as f:
            f.write(html)
        print(f"  Wrote {out_path}", file=sys.stderr)
    else:
        format_results(all_results, airport, terminal)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Monitor TSA wait times from social media",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 tsa_watch.py LGA --terminal C --hours 24
  python3 tsa_watch.py JFK --hours 12 --html
  python3 tsa_watch.py --multi --output-dir /tmp/terminalwatch --hours 72
        """,
    )
    parser.add_argument("airport", nargs="?", help="Airport code (e.g., LGA, JFK, ORD). Omit with --multi.")
    parser.add_argument("--terminal", "-t", help="Terminal letter/number (e.g., C, 1)")
    parser.add_argument("--hours", "-H", type=int, default=72, help="Look back N hours (default: 72)")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    parser.add_argument("--html", action="store_true", help="Output HTML file and open in browser")
    parser.add_argument("--multi", action="store_true", help="Run all default airports, generate site")
    parser.add_argument("--output-dir", help="Output directory for --multi mode (required with --multi)")
    parser.add_argument("--archive-dir", help="Base archive directory (for --multi, archives are per-airport)")

    args = parser.parse_args()

    if args.multi:
        if not args.output_dir:
            print("Error: --output-dir is required with --multi", file=sys.stderr)
            sys.exit(1)

        os.makedirs(args.output_dir, exist_ok=True)
        all_stats = {}

        for airport in DEFAULT_AIRPORTS:
            # Set up per-airport archive dir
            airport_archive = None
            if args.archive_dir:
                airport_archive = os.path.join(args.archive_dir, airport.lower())

            stats = run_single_airport(
                airport=airport,
                terminal=None,  # no terminal focus in multi mode
                hours=args.hours,
                output_dir=args.output_dir,
                archive_base=airport_archive,
            )
            all_stats[airport] = stats

        # Generate landing page
        generate_landing_page(args.output_dir, all_stats)

        # Write combined stats
        combined_stats = {
            "last_run": datetime.now(timezone.utc).isoformat(),
            "airports": all_stats,
        }
        stats_path = os.path.join(args.output_dir, "stats.json")
        with open(stats_path, "w") as f:
            json.dump(combined_stats, f, indent=2)
        print(f"\n  Wrote {stats_path}", file=sys.stderr)
        print(f"  Done: {len(DEFAULT_AIRPORTS)} airports processed.", file=sys.stderr)

    elif args.airport:
        airport = args.airport.upper()

        if args.json:
            # JSON mode: gather and dump, no LLM
            print(f"Searching for TSA reports at {airport}...", file=sys.stderr)
            all_results = []
            all_results.extend(search_reddit(airport, args.terminal, args.hours))
            all_results.extend(search_reddit_comments(airport, args.terminal, args.hours))
            all_results.extend(search_bluesky(airport, args.terminal, args.hours))
            all_results.extend(search_twitter(airport, args.terminal, args.hours))
            print(json.dumps(all_results, indent=2))
        elif args.html:
            # Single airport HTML
            archive_dir = args.archive_dir
            archive_files = []
            if archive_dir and os.path.isdir(archive_dir):
                archive_files = sorted(
                    [f for f in os.listdir(archive_dir) if f.endswith(".html")],
                    reverse=True,
                )

            stats = run_single_airport(airport, args.terminal, args.hours,
                                       output_dir="/tmp", archive_base=args.archive_dir)

            # Move from /tmp/{code}/index.html to /tmp/tsa-watch-{code}.html for compat
            src = f"/tmp/{airport.lower()}/index.html"
            dst = f"/tmp/tsa-watch-{airport.lower()}.html"
            if os.path.exists(src):
                import shutil
                shutil.copy2(src, dst)
                print(f"  Copied to {dst}", file=sys.stderr)

            # Write stats
            try:
                with open("stats.json", "w") as f:
                    json.dump(stats, f, indent=2)
            except OSError:
                pass

            subprocess.run(["open", dst])
        else:
            # Text mode
            run_single_airport(airport, args.terminal, args.hours)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
