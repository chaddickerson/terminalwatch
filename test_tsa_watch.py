"""Unit tests for tsa_watch.py core functionality."""

import json
import os
import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

import tsa_watch


def _recent_ts(hours_ago=1):
    """Return an ISO timestamp for `hours_ago` hours in the past."""
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()


def _recent_utc(hours_ago=1):
    """Return a UTC epoch timestamp for `hours_ago` hours in the past."""
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).timestamp()


# ---------------------------------------------------------------------------
# extract_wait_times
# ---------------------------------------------------------------------------
class TestExtractWaitTimes(unittest.TestCase):
    def test_minutes(self):
        self.assertEqual(tsa_watch.extract_wait_times("took 45 minutes"), [45])

    def test_multiple(self):
        times = tsa_watch.extract_wait_times("waited 20 min, precheck was 5 mins")
        self.assertEqual(sorted(times), [5, 20])

    def test_hours(self):
        self.assertEqual(tsa_watch.extract_wait_times("waited 2 hours"), [120])

    def test_mixed(self):
        times = tsa_watch.extract_wait_times("1 hour and 30 minutes")
        self.assertEqual(sorted(times), [30, 60])

    def test_no_times(self):
        self.assertEqual(tsa_watch.extract_wait_times("security was fine"), [])

    def test_case_insensitive(self):
        self.assertEqual(tsa_watch.extract_wait_times("took 10 Minutes"), [10])


# ---------------------------------------------------------------------------
# _terminal_match
# ---------------------------------------------------------------------------
class TestTerminalMatch(unittest.TestCase):
    def test_match_terminal_c(self):
        self.assertTrue(tsa_watch._terminal_match("long line at terminal c today", "C"))

    def test_match_concourse(self):
        self.assertTrue(tsa_watch._terminal_match("concourse B was empty", "B"))

    def test_match_abbreviation(self):
        self.assertTrue(tsa_watch._terminal_match("T1 security is backed up", "1"))

    def test_no_match(self):
        self.assertFalse(tsa_watch._terminal_match("terminal A was fine", "C"))

    def test_none_terminal(self):
        self.assertIsNone(tsa_watch._terminal_match("anything", None))


# ---------------------------------------------------------------------------
# _detect_terminal
# ---------------------------------------------------------------------------
class TestDetectTerminal(unittest.TestCase):
    def test_single_terminal(self):
        self.assertEqual(tsa_watch._detect_terminal("terminal 3 was packed"), {"3"})

    def test_multiple_terminals(self):
        result = tsa_watch._detect_terminal("terminal A is worse than terminal B")
        self.assertEqual(result, {"A", "B"})

    def test_concourse(self):
        self.assertEqual(tsa_watch._detect_terminal("concourse C precheck"), {"C"})

    def test_no_terminal(self):
        self.assertEqual(tsa_watch._detect_terminal("security line is long"), set())


# ---------------------------------------------------------------------------
# search_bluesky
# ---------------------------------------------------------------------------
class TestSearchBluesky(unittest.TestCase):
    def _make_bsky_response(self, text, handle="user.bsky.social", hours_ago=1):
        ts = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).strftime(
            "%Y-%m-%dT%H:%M:%S.000Z"
        )
        return {
            "posts": [
                {
                    "uri": f"at://did:plc:abc/app.bsky.feed.post/post123",
                    "record": {"text": text, "createdAt": ts},
                    "author": {"handle": handle},
                    "likeCount": 5,
                    "replyCount": 1,
                }
            ]
        }

    @patch("tsa_watch._request")
    def test_returns_matching_post(self, mock_request):
        mock_request.return_value = self._make_bsky_response(
            "ORD TSA precheck took 15 minutes"
        )
        results = tsa_watch.search_bluesky("ORD", hours=24)
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0]["source"], "bluesky")
        self.assertIn("15 minutes", results[0]["body"])

    @patch("tsa_watch._request")
    def test_filters_old_posts(self, mock_request):
        mock_request.return_value = self._make_bsky_response(
            "ORD TSA was fast", hours_ago=48
        )
        results = tsa_watch.search_bluesky("ORD", hours=24)
        self.assertEqual(len(results), 0)

    @patch("tsa_watch._request")
    def test_filters_wrong_airport(self, mock_request):
        mock_request.return_value = self._make_bsky_response("JFK TSA was slow")
        results = tsa_watch.search_bluesky("ORD", hours=24)
        self.assertEqual(len(results), 0)

    @patch("tsa_watch._request")
    def test_skips_bot_accounts(self, mock_request):
        mock_request.return_value = self._make_bsky_response(
            "ORD TSA trending", handle="nowbreezing.bsky.social"
        )
        results = tsa_watch.search_bluesky("ORD", hours=24)
        self.assertEqual(len(results), 0)

    @patch("tsa_watch._request")
    def test_handles_api_failure(self, mock_request):
        mock_request.return_value = None
        results = tsa_watch.search_bluesky("ORD", hours=24)
        self.assertEqual(results, [])

    @patch("tsa_watch._request")
    def test_security_query_match(self, mock_request):
        """Posts mentioning 'security' without 'TSA' should still be found."""
        mock_request.return_value = self._make_bsky_response(
            "O'Hare security line was 30 minutes this morning"
        )
        results = tsa_watch.search_bluesky("ORD", hours=24)
        self.assertTrue(len(results) > 0)


# ---------------------------------------------------------------------------
# search_twitter
# ---------------------------------------------------------------------------
class TestSearchTwitter(unittest.TestCase):
    def _make_twitter_response(self, text, hours_ago=1):
        ts = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        return json.dumps({
            "data": [
                {
                    "id": "123456",
                    "text": text,
                    "created_at": ts,
                    "author_id": "user1",
                    "public_metrics": {"like_count": 3, "reply_count": 0},
                }
            ],
            "includes": {"users": [{"id": "user1", "username": "traveler"}]},
        }).encode()

    @patch("urllib.request.urlopen")
    def test_returns_matching_tweet(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = self._make_twitter_response(
            "LAX TSA precheck was 20 minutes"
        )
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        with patch.dict(os.environ, {"TWITTER_BEARER_TOKEN": "fake-token"}):
            results = tsa_watch.search_twitter("LAX", hours=24)
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0]["source"], "twitter")

    def test_skips_without_bearer_token(self):
        with patch.dict(os.environ, {}, clear=True):
            # Make sure TWITTER_BEARER_TOKEN is not set
            os.environ.pop("TWITTER_BEARER_TOKEN", None)
            results = tsa_watch.search_twitter("LAX", hours=24)
        self.assertEqual(results, [])


# ---------------------------------------------------------------------------
# _reddit_auth_headers
# ---------------------------------------------------------------------------
class TestRedditAuth(unittest.TestCase):
    def setUp(self):
        # Reset token cache between tests
        tsa_watch._reddit_token_cache["token"] = None
        tsa_watch._reddit_token_cache["expires_at"] = 0

    def test_returns_none_without_credentials(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("REDDIT_CLIENT_ID", None)
            os.environ.pop("REDDIT_CLIENT_SECRET", None)
            self.assertIsNone(tsa_watch._reddit_auth_headers())

    @patch("urllib.request.urlopen")
    def test_returns_bearer_headers(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "access_token": "test-token-123",
            "token_type": "bearer",
            "expires_in": 3600,
        }).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        with patch.dict(os.environ, {
            "REDDIT_CLIENT_ID": "my-client-id",
            "REDDIT_CLIENT_SECRET": "my-secret",
        }):
            headers = tsa_watch._reddit_auth_headers()
        self.assertEqual(headers["Authorization"], "bearer test-token-123")
        self.assertIn("tsa-watch", headers["User-Agent"])

    @patch("urllib.request.urlopen")
    def test_caches_token(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "access_token": "cached-token",
            "token_type": "bearer",
            "expires_in": 3600,
        }).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        with patch.dict(os.environ, {
            "REDDIT_CLIENT_ID": "my-client-id",
            "REDDIT_CLIENT_SECRET": "my-secret",
        }):
            headers1 = tsa_watch._reddit_auth_headers()
            headers2 = tsa_watch._reddit_auth_headers()
        # Should only call the API once
        self.assertEqual(mock_urlopen.call_count, 1)
        self.assertEqual(headers1["Authorization"], headers2["Authorization"])

    @patch("urllib.request.urlopen")
    def test_returns_none_on_api_error(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.URLError("connection refused")
        with patch.dict(os.environ, {
            "REDDIT_CLIENT_ID": "my-client-id",
            "REDDIT_CLIENT_SECRET": "my-secret",
        }):
            self.assertIsNone(tsa_watch._reddit_auth_headers())

    @patch("tsa_watch._request")
    def test_search_reddit_uses_oauth_when_available(self, mock_request):
        """When Reddit credentials are set, search should use oauth.reddit.com."""
        mock_request.return_value = {
            "data": {
                "children": [{
                    "data": {
                        "title": "ATL TSA security 30 min",
                        "selftext": "",
                        "created_utc": _recent_utc(1),
                        "permalink": "/r/TSA/comments/abc/test",
                        "score": 5,
                        "num_comments": 1,
                    }
                }]
            }
        }
        tsa_watch._reddit_token_cache["token"] = "fake-token"
        import time
        tsa_watch._reddit_token_cache["expires_at"] = time.time() + 3600

        with patch.dict(os.environ, {
            "REDDIT_CLIENT_ID": "my-client-id",
            "REDDIT_CLIENT_SECRET": "my-secret",
        }):
            results = tsa_watch.search_reddit("ATL", hours=24)

        # Verify oauth.reddit.com was used (check the URL passed to _request)
        called_url = mock_request.call_args_list[0][0][0]
        self.assertIn("oauth.reddit.com", called_url)
        self.assertNotIn(".json", called_url)


# ---------------------------------------------------------------------------
# search_reddit
# ---------------------------------------------------------------------------
class TestSearchReddit(unittest.TestCase):
    def _make_reddit_response(self, title, body="", hours_ago=1):
        return {
            "data": {
                "children": [
                    {
                        "data": {
                            "title": title,
                            "selftext": body,
                            "created_utc": _recent_utc(hours_ago),
                            "permalink": "/r/TSA/comments/abc/test",
                            "score": 10,
                            "num_comments": 3,
                        }
                    }
                ]
            }
        }

    @patch("tsa_watch._request")
    def test_returns_matching_post(self, mock_request):
        mock_request.return_value = self._make_reddit_response(
            "ATL TSA security was 45 minutes"
        )
        results = tsa_watch.search_reddit("ATL", hours=24)
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0]["source"], "reddit")

    @patch("tsa_watch._request")
    def test_filters_old_posts(self, mock_request):
        mock_request.return_value = self._make_reddit_response(
            "ATL TSA line was long", hours_ago=48
        )
        results = tsa_watch.search_reddit("ATL", hours=24)
        self.assertEqual(len(results), 0)

    @patch("tsa_watch._request")
    def test_filters_wrong_airport(self, mock_request):
        mock_request.return_value = self._make_reddit_response("JFK TSA was slow")
        results = tsa_watch.search_reddit("ATL", hours=24)
        self.assertEqual(len(results), 0)

    @patch("tsa_watch._request")
    def test_deduplicates_by_url(self, mock_request):
        resp = self._make_reddit_response("ATL TSA took forever")
        mock_request.return_value = resp
        results = tsa_watch.search_reddit("ATL", hours=24)
        urls = [r["url"] for r in results]
        self.assertEqual(len(urls), len(set(urls)))


# ---------------------------------------------------------------------------
# search_reddit_comments
# ---------------------------------------------------------------------------
class TestSearchRedditComments(unittest.TestCase):
    @patch("tsa_watch._request")
    def test_returns_matching_comment(self, mock_request):
        mock_request.return_value = {
            "data": {
                "children": [
                    {
                        "data": {
                            "body": "ORD TSA wait was about 20 minutes in terminal 1",
                            "created_utc": _recent_utc(1),
                            "permalink": "/r/flying/comments/abc/test/def",
                            "subreddit": "flying",
                            "score": 5,
                        }
                    }
                ]
            }
        }
        results = tsa_watch.search_reddit_comments("ORD", hours=24)
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0]["source"], "reddit_comment")

    @patch("tsa_watch._request")
    def test_filters_wrong_airport(self, mock_request):
        mock_request.return_value = {
            "data": {
                "children": [
                    {
                        "data": {
                            "body": "JFK security was a nightmare",
                            "created_utc": _recent_utc(1),
                            "permalink": "/r/flying/comments/abc/test/def",
                            "subreddit": "flying",
                            "score": 5,
                        }
                    }
                ]
            }
        }
        results = tsa_watch.search_reddit_comments("ORD", hours=24)
        self.assertEqual(len(results), 0)


# ---------------------------------------------------------------------------
# llm_filter_posts
# ---------------------------------------------------------------------------
class TestLlmFilterPosts(unittest.TestCase):
    def _make_posts(self):
        return [
            {"title": "TSA took 30 min", "body": "at ORD terminal 1", "timestamp": _recent_ts()},
            {"title": "TSA politics bad", "body": "funding is terrible", "timestamp": _recent_ts()},
            {"title": "PreCheck 5 min", "body": "breezed through", "timestamp": _recent_ts()},
        ]

    @patch("tsa_watch._llm")
    def test_filters_by_llm_response(self, mock_llm):
        mock_llm.return_value = "[0, 2]"
        posts = self._make_posts()
        result = tsa_watch.llm_filter_posts(posts, "ORD")
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["title"], "TSA took 30 min")
        self.assertEqual(result[1]["title"], "PreCheck 5 min")

    @patch("tsa_watch._llm")
    def test_returns_all_on_llm_failure(self, mock_llm):
        mock_llm.return_value = None
        posts = self._make_posts()
        result = tsa_watch.llm_filter_posts(posts, "ORD")
        self.assertEqual(len(result), 3)

    @patch("tsa_watch._llm")
    def test_returns_all_on_unparseable_response(self, mock_llm):
        mock_llm.return_value = "I cannot determine which posts to include"
        posts = self._make_posts()
        result = tsa_watch.llm_filter_posts(posts, "ORD")
        self.assertEqual(len(result), 3)

    @patch("tsa_watch._llm")
    def test_handles_empty_include(self, mock_llm):
        mock_llm.return_value = "[]"
        posts = self._make_posts()
        result = tsa_watch.llm_filter_posts(posts, "ORD")
        self.assertEqual(len(result), 0)

    def test_empty_posts(self):
        result = tsa_watch.llm_filter_posts([], "ORD")
        self.assertEqual(result, [])

    @patch("tsa_watch._llm")
    def test_handles_markdown_wrapped_response(self, mock_llm):
        mock_llm.return_value = "Here are the relevant posts:\n[0, 1]\n"
        posts = self._make_posts()
        result = tsa_watch.llm_filter_posts(posts, "ORD")
        self.assertEqual(len(result), 2)


# ---------------------------------------------------------------------------
# llm_generate_summary
# ---------------------------------------------------------------------------
class TestLlmGenerateSummary(unittest.TestCase):
    def _make_posts(self):
        return [
            {
                "title": "TSA precheck 10 min",
                "body": "Terminal 1 was quick",
                "timestamp": _recent_ts(),
                "source": "bluesky",
                "url": "https://bsky.app/post/123",
                "detected_terminals": {"1"},
            },
        ]

    @patch("tsa_watch._llm")
    def test_returns_summary_html(self, mock_llm):
        mock_llm.return_value = "<p>Terminal 1 PreCheck was <b>10 minutes</b>.</p>"
        result = tsa_watch.llm_generate_summary(self._make_posts(), "ORD")
        self.assertIn("<p>", result)
        self.assertIn("10 minutes", result)

    @patch("tsa_watch._llm")
    def test_strips_code_fences(self, mock_llm):
        mock_llm.return_value = "```html\n<p>Summary here.</p>\n```"
        result = tsa_watch.llm_generate_summary(self._make_posts(), "ORD")
        self.assertNotIn("```", result)
        self.assertIn("<p>Summary here.</p>", result)

    @patch("tsa_watch._llm")
    def test_returns_fallback_on_failure(self, mock_llm):
        mock_llm.return_value = None
        result = tsa_watch.llm_generate_summary(self._make_posts(), "ORD")
        self.assertEqual(result, "<p>Summary unavailable.</p>")

    def test_empty_posts(self):
        result = tsa_watch.llm_generate_summary([], "ORD")
        self.assertIn("No wait time reports", result)


# ---------------------------------------------------------------------------
# _request
# ---------------------------------------------------------------------------
class TestRequest(unittest.TestCase):
    @patch("urllib.request.urlopen")
    def test_returns_parsed_json(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"data": "test"}'
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = tsa_watch._request("https://example.com/api")
        self.assertEqual(result, {"data": "test"})

    @patch("urllib.request.urlopen")
    def test_returns_none_on_error(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.URLError("timeout")
        result = tsa_watch._request("https://example.com/api")
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# _summarize_terminal
# ---------------------------------------------------------------------------
class TestSummarizeTerminal(unittest.TestCase):
    def test_general_and_precheck(self):
        posts = [
            {"title": "TSA was 30 min", "body": "regular line"},
            {"title": "PreCheck took 10 minutes", "body": "not bad"},
        ]
        result = tsa_watch._summarize_terminal(posts)
        self.assertIn("General", result)
        self.assertIn("PreCheck", result)

    def test_no_times(self):
        posts = [{"title": "Long line", "body": "no specific time"}]
        result = tsa_watch._summarize_terminal(posts)
        self.assertIn("No explicit", result)


import urllib.error

if __name__ == "__main__":
    unittest.main()
