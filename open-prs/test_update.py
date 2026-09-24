import base64
import datetime as dt
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("updater", Path(__file__).with_name("update.py"))
updater = importlib.util.module_from_spec(spec)
spec.loader.exec_module(updater)


def page(nodes, more=False, cursor=None):
    return {"repository": {"pullRequests": {
        "nodes": nodes, "pageInfo": {"hasNextPage": more, "endCursor": cursor},
    }}}


def pr(number, created, author="alice"):
    return {"number": number, "createdAt": created,
            "author": {"login": author} if author else None}


class SnapshotTests(unittest.TestCase):
    def test_recent_window_boundaries_and_pagination(self):
        until = dt.datetime(2026, 9, 24, 12, tzinfo=dt.timezone.utc)
        responses = [
            page([
                pr(6, "2026-09-24T12:00:01Z"),
                pr(5, "2026-09-24T12:00:00Z"),
                pr(4, "2026-09-11T10:00:00Z", "bob"),
            ], True, "page-2"),
            page([
                pr(3, "2026-09-10T12:00:00Z"),
                pr(2, "2026-09-10T12:00:00Z", None),
                pr(1, "2026-09-10T11:59:59Z"),
            ], True, "unused-older-page"),
        ]
        with patch.object(updater, "graphql", side_effect=responses) as request:
            result = updater.fetch_recent("gh", until)
        self.assertEqual(result["total"], 4)
        self.assertEqual(result["byAuthor"], {"alice": 2, "bob": 1, "deleted-account": 1})
        self.assertEqual(result["pages"], 2)
        self.assertTrue(result["complete"])
        self.assertEqual(request.call_args.args[2], {"cursor": "page-2"})

    def test_duplicate_recent_pr_rejected(self):
        node = pr(2, "2026-09-20T12:00:00Z")
        with patch.object(updater, "graphql", return_value=page([node, node])):
            with self.assertRaisesRegex(RuntimeError, "Duplicate"):
                updater.fetch_recent("gh", dt.datetime(2026, 9, 24, tzinfo=dt.timezone.utc))

    def test_recent_order_violation_rejected(self):
        nodes = [pr(1, "2026-09-12T12:00:00Z"), pr(2, "2026-09-20T12:00:00Z")]
        with patch.object(updater, "graphql", return_value=page(nodes)):
            with self.assertRaisesRegex(RuntimeError, "ordered"):
                updater.fetch_recent("gh", dt.datetime(2026, 9, 24, tzinfo=dt.timezone.utc))

    def test_roster_uses_only_authoritative_role_sections(self):
        text = """# Governance
### Lead Maintainers
- [@Lead](https://github.com/Lead)
### Active Committers
- [@Writer](https://github.com/Writer): role
## Reviewer Routing
- [@ReviewerOnly](https://github.com/ReviewerOnly)
"""
        response = {"sha": "example", "content": base64.b64encode(text.encode()).decode()}
        with patch.object(updater.subprocess, "run", return_value=SimpleNamespace(stdout=json.dumps(response))):
            result = updater.fetch_committers("gh")
        self.assertEqual(result["logins"], ["Lead", "Writer"])

    def test_missing_roster_section_rejected(self):
        response = {"sha": "example", "content": base64.b64encode(b"# New governance format").decode()}
        with patch.object(updater.subprocess, "run", return_value=SimpleNamespace(stdout=json.dumps(response))):
            with self.assertRaisesRegex(RuntimeError, "format changed"):
                updater.fetch_committers("gh")


if __name__ == "__main__":
    unittest.main()
