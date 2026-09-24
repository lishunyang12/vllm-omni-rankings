#!/usr/bin/env python3
"""Fetch every open PR using cursor pagination, then atomically publish a snapshot."""

import argparse
import base64
import collections
import datetime as dt
import json
import os
import re
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent


def fetch_committers(gh):
    """Use the public, authoritative governance roster, not org membership."""
    result = subprocess.run(
        [gh, "api", "repos/vllm-project/vllm-omni/contents/docs/community/governance.md"],
        capture_output=True, text=True, timeout=90, check=True,
    )
    content = json.loads(result.stdout)
    document = base64.b64decode(content["content"]).decode("utf-8")
    members = set()
    for heading in ("Lead Maintainers", "Active Committers"):
        section = re.search(r"^### " + re.escape(heading) + r"\s*\n(.*?)(?=^##|\Z)", document, re.M | re.S)
        if not section:
            raise RuntimeError("Governance roster format changed; existing snapshot was preserved")
        logins = re.findall(r"https://github\.com/([A-Za-z0-9-]+)\)", section.group(1))
        if not logins:
            raise RuntimeError("Empty governance roster; existing snapshot was preserved")
        members.update(logins)
    return {
        "logins": sorted(members, key=str.casefold),
        "source": "https://github.com/vllm-project/vllm-omni/blob/main/docs/community/governance.md",
        "blobSha": content["sha"], "updatedAt": dt.datetime.now(dt.timezone.utc).isoformat(),
        "definition": "Active Committers and Lead Maintainers in the public governance roster",
    }


def graphql(gh, query, variables=None):
    payload = json.dumps({"query": query, "variables": variables or {}})
    for attempt in range(3):
        result = subprocess.run(
            [gh, "api", "graphql", "--input", "-"], input=payload,
            capture_output=True, text=True, timeout=90,
        )
        if result.returncode == 0:
            response = json.loads(result.stdout)
            if not response.get("errors"):
                return response["data"]
        print(f"GitHub request retry {attempt + 1}: {result.stderr[:1500]}", flush=True)
        if result.stdout:
            try:
                print(json.dumps(json.loads(result.stdout).get('errors', [])), flush=True)
            except ValueError:
                pass
        if attempt < 2:
            time.sleep(2 ** attempt)
    raise RuntimeError("GitHub GraphQL request failed; existing snapshot was preserved")


def fetch_recent(gh, until=None):
    """Count creation events across all PR states in a fixed rolling 14-day window."""
    until = until or dt.datetime.now(dt.timezone.utc)
    since = until - dt.timedelta(days=14)
    cursor, pages, previous = None, 0, None
    seen, counts = set(), collections.Counter()
    while True:
        data = graphql(gh, (ROOT / "recent.graphql").read_text(), {"cursor": cursor})
        connection = data["repository"]["pullRequests"]
        reached_cutoff = False
        for raw in connection["nodes"]:
            if raw["number"] in seen:
                raise RuntimeError("Duplicate PR in recent submissions traversal")
            seen.add(raw["number"])
            created = dt.datetime.fromisoformat(raw["createdAt"].replace("Z", "+00:00"))
            if previous is not None and created > previous:
                raise RuntimeError("Recent PRs were not ordered by creation time")
            previous = created
            if created < since:
                reached_cutoff = True
            elif created <= until:
                counts[(raw["author"] or {}).get("login", "deleted-account")] += 1
        pages += 1
        print(f"recent page {pages}: {sum(counts.values())} submissions", flush=True)
        info = connection["pageInfo"]
        if reached_cutoff or not info["hasNextPage"]:
            break
        if not info["endCursor"] or info["endCursor"] == cursor:
            raise RuntimeError("Recent PR cursor did not advance")
        cursor = info["endCursor"]
    return {
        "days": 14, "since": since.isoformat(), "until": until.isoformat(),
        "total": sum(counts.values()), "byAuthor": dict(sorted(counts.items())),
        "pages": pages, "complete": True,
        "scope": "PRs created in [since, until], including open, merged and closed",
    }


def fetch(gh):
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    cursor, prs, pages, totals = None, {}, 0, []
    while True:
        data = graphql(gh, (ROOT / "query.graphql").read_text(), {"cursor": cursor})
        connection = data["repository"]["pullRequests"]
        totals.append(connection["totalCount"])
        for raw in connection["nodes"]:
            if raw["number"] in prs:
                raise RuntimeError("Duplicate PR encountered while paginating")
            author = raw["author"] or {
                "login": "deleted-account", "url": "https://github.com/ghost",
                "avatarUrl": "https://github.com/ghost.png", "__typename": "User",
            }
            if raw["labels"]["pageInfo"]["hasNextPage"]:
                raise RuntimeError("PR labels exceeded page size; existing snapshot was preserved")
            prs[raw["number"]] = {
                "number": raw["number"], "title": raw["title"], "url": raw["url"],
                "author": author["login"], "avatar": author["avatarUrl"],
                "authorUrl": author["url"], "bot": author["__typename"] == "Bot",
                "draft": raw["isDraft"], "created": raw["createdAt"], "updated": raw["updatedAt"],
                "labels": [label["name"] for label in raw["labels"]["nodes"]],
            }
        pages += 1
        print(f"page {pages}: {len(prs)} / {connection['totalCount']} PRs", flush=True)
        info = connection["pageInfo"]
        if not info["hasNextPage"]:
            break
        if not info["endCursor"] or info["endCursor"] == cursor:
            raise RuntimeError("Cursor did not advance")
        cursor = info["endCursor"]
    count_query = '{repository(owner:"vllm-project",name:"vllm-omni"){pullRequests(states:OPEN){totalCount}}}'
    total = graphql(gh, count_query)["repository"]["pullRequests"]["totalCount"]
    if any(t != total for t in totals) or len(prs) != total:
        return None
    items = sorted(prs.values(), key=lambda p: p["number"], reverse=True)
    authors = collections.Counter(p["author"] for p in items)
    recent = fetch_recent(gh)
    committers = fetch_committers(gh)
    return {
        "schema": 1, "repository": "vllm-project/vllm-omni",
        "source": "https://github.com/vllm-project/vllm-omni/pulls",
        "startedAt": started, "updatedAt": dt.datetime.now(dt.timezone.utc).isoformat(),
        "total": total, "authors": len(authors), "pages": pages, "complete": True,
        "method": "GitHub repository.pullRequests(states:OPEN), all cursor pages; counts rechecked after fetch",
        "pullRequests": items,
        "recentSubmissions": recent,
        "committers": committers,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gh", default="gh")
    parser.add_argument("--output", type=Path, default=ROOT / "data.json")
    args = parser.parse_args()
    for attempt in range(3):
        snapshot = fetch(args.gh)
        if snapshot is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.output.with_suffix(".tmp")
            temporary.write_text(json.dumps(snapshot, ensure_ascii=False, separators=(",", ":")) + "\n")
            os.replace(temporary, args.output)
            print(f"Saved {snapshot['total']} open PRs / {snapshot['authors']} authors")
            return
        print(f"PR count changed during capture; retry {attempt + 1}/3", flush=True)
    raise RuntimeError("Repository changed during all three captures; existing snapshot was preserved")


if __name__ == "__main__":
    main()
