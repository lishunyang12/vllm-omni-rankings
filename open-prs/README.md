# vLLM-Omni open PR leaderboard

Public page: https://lishunyang12.github.io/vllm-omni-rankings/open-prs/

Ranks every author with at least one open PR in `vllm-project/vllm-omni`.
Drafts and bots are included by default. PR count is not a quality score.
The page includes tied ranks, author search, Draft filters, recent activity
sorting, per-author PR title/number search, GitHub links, CSV and JSON export.
Only PR metadata is collected: no CI results, reviews, comments or code diffs.
Both pages use English UI text and the unmodified official vLLM-Omni logo.

## Views and filters

- The contributor leaderboard includes a **New · 14d** column and sort.
  This counts all PRs created in a fixed rolling 14-day window, including merged
  and closed PRs. It is independent of the open-PR Draft filter. Only authors
  with current open PRs appear in this leaderboard.
- The [Topics subpage](https://lishunyang12.github.io/vllm-omni-rankings/open-prs/topics/)
  classifies PRs by model, hardware, and technical area, with title/label
  evidence, AND/OR combinations, search, CSV export, and shareable filter URLs.
  Topic counts overlap; result counts deduplicate PRs. Unmatched PRs remain
  accessible as **Unclassified**.
- **Exclude committers** on both views uses only Active Committers and Lead
  Maintainers listed in the public upstream governance document. The roster
  refreshes with every snapshot. It does not infer roles from CODEOWNERS,
  organization membership, or private collaborator permission data.

Topic rules live in topics/rules.js. They identify mentions rather than
verified capabilities. They do not inspect PR bodies or code changes.
The **Tests / build** topic describes PR content and does not collect CI status.

The 14-day counter separately paginates PRs across all states in descending
creation order until the cutoff. Its since and until timestamps define the
exact inclusive window; it is not limited to 1,000 results. A failed recent
count or governance fetch also preserves the previous complete snapshot.

## Official visual assets

The unchanged logo and favicon in assets/ are sourced from the upstream
[docs/source/logos](https://github.com/vllm-project/vllm-omni/tree/main/docs/source/logos)
directory: vllm-omni-logo.png and vllm-logo-only-light.ico.

## Refresh

Run `python open-prs/update.py` with authenticated GitHub CLI (`gh`) or
`GH_TOKEN`. The updater cursor-paginates `repository.pullRequests(states:OPEN)`
and is not subject to the search API's 1,000-result cap. It checks unique PRs
and total counts before atomically replacing `data.json`. A changing count
causes a new traversal; failures leave the existing complete snapshot intact.
GitHub has no atomic multi-page snapshot: `startedAt` / `updatedAt` bound capture.

The hosting server runs `bash open-prs/publish.sh` hourly at minute 17 via cron.
The script locks concurrent runs, refreshes only this page's JSON, rebases over
other repository changes and pushes with the host's authenticated GitHub CLI.
The push triggers the existing GitHub Pages build. No credential reaches the
browser. Host uptime, GitHub availability and deployments can delay updates.
The refresh button reads the latest published snapshot; a warning appears if
it is over three hours old. Commit history provides the public refresh log.

To move the updater to another host, clone this repository, authenticate `gh`,
and schedule `bash /path/to/checkout/open-prs/publish.sh`. It needs Git,
Python 3 and `flock` and refuses to overwrite tracked local changes.

## Preview

Run `python -m http.server 18445` at repository root and visit
http://localhost:18445/open-prs/ . The static page requires no frontend build.
