# vLLM-Omni open PR leaderboard

Public page: https://lishunyang12.github.io/vllm-omni-rankings/open-prs/

Ranks every author with at least one open PR in `vllm-project/vllm-omni`.
Drafts and bots are included by default. PR count is not a quality score.
The page includes tied ranks, author search, Draft filters, recent activity
sorting, per-author PR title/number search, GitHub links, CSV and JSON export.
Only basic PR metadata is collected: no CI, reviews, comments or code diffs.

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
