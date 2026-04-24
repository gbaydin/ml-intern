"""
Web Search Tool — General-purpose web search via DuckDuckGo.

Fills the gap between the papers tool (Semantic Scholar / HF) and docs tools
(HF / Gradio) by providing access to the broader web: StackOverflow answers,
library documentation (pymatgen, scikit-learn, PyTorch, ...), GitHub issues,
blog posts, tutorials, and any other publicly available page.

Uses the ``duckduckgo-search`` package (DDGS class) when available, falling
back to a lightweight httpx-based HTML scraper otherwise.
"""

import logging
import re
from typing import Any

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MAX_RESULTS = 8
MAX_RESULTS_CAP = 20
FETCH_TOP_N = 3
MAX_CONTENT_PER_PAGE = 6000

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# Elements to strip when extracting readable content
_STRIP_TAGS = {"nav", "header", "footer", "script", "style", "noscript", "svg", "iframe"}


# ---------------------------------------------------------------------------
# Search backends
# ---------------------------------------------------------------------------


async def _search_ddgs(query: str, max_results: int) -> list[dict[str, str]]:
    """Search using the ``duckduckgo-search`` package (DDGS.text).

    Returns a list of dicts with keys: title, url, snippet.
    Raises ImportError if the package is not installed.
    """
    from duckduckgo_search import DDGS  # type: ignore[import-untyped]

    results: list[dict[str, str]] = []
    # DDGS is synchronous — run in the default executor to avoid blocking
    import asyncio

    def _run() -> list[dict[str, Any]]:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))

    raw = await asyncio.get_running_loop().run_in_executor(None, _run)
    for item in raw:
        results.append({
            "title": item.get("title", ""),
            "url": item.get("href", item.get("link", "")),
            "snippet": item.get("body", item.get("snippet", "")),
        })
    return results


async def _search_httpx(query: str, max_results: int) -> list[dict[str, str]]:
    """Fallback: scrape DuckDuckGo HTML search results with httpx."""
    params = {"q": query}
    async with httpx.AsyncClient(
        headers=_HEADERS, timeout=15, follow_redirects=True
    ) as client:
        resp = await client.get("https://html.duckduckgo.com/html/", params=params)
        resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    results: list[dict[str, str]] = []

    for result_div in soup.select(".result"):
        if len(results) >= max_results:
            break

        link_tag = result_div.select_one("a.result__a")
        snippet_tag = result_div.select_one(".result__snippet")
        if not link_tag:
            continue

        title = link_tag.get_text(strip=True)
        href = link_tag.get("href", "")

        # DuckDuckGo HTML wraps URLs in a redirect; extract the actual URL
        if "uddg=" in str(href):
            import urllib.parse

            parsed = urllib.parse.parse_qs(urllib.parse.urlparse(str(href)).query)
            href = parsed.get("uddg", [str(href)])[0]

        snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
        if title and href:
            results.append({"title": title, "url": str(href), "snippet": snippet})

    return results


async def _search(query: str, max_results: int) -> list[dict[str, str]]:
    """Try DDGS first, fall back to httpx scraping."""
    try:
        return await _search_ddgs(query, max_results)
    except ImportError:
        logger.debug("duckduckgo-search not installed; falling back to httpx scraper")
    except Exception as exc:
        logger.warning("DDGS search failed (%s); falling back to httpx scraper", exc)
    return await _search_httpx(query, max_results)


# ---------------------------------------------------------------------------
# Content fetching
# ---------------------------------------------------------------------------


def _extract_readable_text(html: str) -> str:
    """Extract readable body text from an HTML page.

    Strips nav, header, footer, script, style, and other boilerplate elements.
    Returns plain text capped at ``MAX_CONTENT_PER_PAGE`` characters.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove unwanted elements
    for tag_name in _STRIP_TAGS:
        for el in soup.find_all(tag_name):
            el.decompose()

    # Also remove common boilerplate by role / class patterns
    for el in soup.find_all(attrs={"role": re.compile(r"^(navigation|banner|contentinfo)$")}):
        el.decompose()

    # Prefer <main> or <article> if present
    main = soup.find("main") or soup.find("article")
    target = main if main else soup.body if soup.body else soup

    text = target.get_text(separator="\n", strip=True)

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    if len(text) > MAX_CONTENT_PER_PAGE:
        text = text[:MAX_CONTENT_PER_PAGE] + f"\n\n... (truncated at {MAX_CONTENT_PER_PAGE} chars)"

    return text


async def _fetch_page_content(client: httpx.AsyncClient, url: str) -> str | None:
    """Fetch a URL and extract readable text.  Returns None on failure."""
    try:
        resp = await client.get(url, headers=_HEADERS)
        resp.raise_for_status()
        return _extract_readable_text(resp.text)
    except Exception as exc:
        logger.debug("Failed to fetch %s: %s", url, exc)
        return None


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _format_results(
    query: str,
    results: list[dict[str, str]],
    fetched_contents: dict[str, str] | None = None,
) -> str:
    """Format search results (and optional page contents) as readable text."""
    lines = [f"# Web Search: '{query}'"]
    lines.append(f"Found {len(results)} result(s)\n")

    for i, r in enumerate(results, 1):
        lines.append(f"## {i}. {r['title']}")
        lines.append(f"URL: {r['url']}")
        if r.get("snippet"):
            lines.append(f"Snippet: {r['snippet']}")

        if fetched_contents and r["url"] in fetched_contents:
            content = fetched_contents[r["url"]]
            lines.append(f"\n### Page Content\n{content}")

        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool spec
# ---------------------------------------------------------------------------

WEB_SEARCH_TOOL_SPEC: dict[str, Any] = {
    "name": "web_search",
    "description": (
        "General-purpose web search via DuckDuckGo. Use this for questions that "
        "go beyond ML papers and HF documentation: library APIs (pymatgen, scikit-learn, "
        "PyTorch, NumPy, ASE, etc.), StackOverflow answers, GitHub issues, tutorials, "
        "blog posts, implementation how-tos, error messages, and any other publicly "
        "available information.\n\n"
        "NOT for ML papers (use hf_papers) or HF library docs (use explore_hf_docs).\n\n"
        "Set fetch_content=true to also retrieve the full text of the top result "
        "pages — useful for reading documentation, StackOverflow answers, or "
        "detailed tutorials without a separate fetch step."
    ),
    "parameters": {
        "type": "object",
        "required": ["query"],
        "additionalProperties": False,
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Search query. Be specific — include library names, function "
                    "names, error messages, or version numbers when relevant. "
                    "Examples: 'pymatgen Structure from_file CIF', "
                    "'torch.compile dynamic shapes tutorial', "
                    "'scikit-learn ColumnTransformer remainder passthrough'."
                ),
            },
            "max_results": {
                "type": "integer",
                "description": (
                    f"Maximum number of search results to return (default {DEFAULT_MAX_RESULTS}, "
                    f"max {MAX_RESULTS_CAP})."
                ),
            },
            "fetch_content": {
                "type": "boolean",
                "description": (
                    "When true, fetch and extract readable text from the top "
                    f"{FETCH_TOP_N} result URLs. Adds latency but gives you the "
                    "actual page content (documentation, answers, code examples) "
                    "directly. Default: false."
                ),
            },
        },
    },
}


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


async def web_search_handler(args: dict[str, Any], **_kw: Any) -> tuple[str, bool]:
    """Handle a web_search tool call.

    Returns ``(formatted_output, success_bool)``.
    """
    query = args.get("query", "").strip()
    if not query:
        return "Error: 'query' parameter is required.", False

    max_results = min(args.get("max_results", DEFAULT_MAX_RESULTS), MAX_RESULTS_CAP)
    fetch_content = bool(args.get("fetch_content", False))

    try:
        results = await _search(query, max_results)
    except Exception as exc:
        return f"Search failed: {exc}", False

    if not results:
        return f"No web results found for '{query}'.", False

    # Optionally fetch full page content for top results
    fetched_contents: dict[str, str] | None = None
    if fetch_content:
        fetched_contents = {}
        urls_to_fetch = [r["url"] for r in results[:FETCH_TOP_N] if r.get("url")]
        async with httpx.AsyncClient(
            timeout=20, follow_redirects=True
        ) as client:
            for url in urls_to_fetch:
                content = await _fetch_page_content(client, url)
                if content:
                    fetched_contents[url] = content

    formatted = _format_results(query, results, fetched_contents)
    return formatted, True
