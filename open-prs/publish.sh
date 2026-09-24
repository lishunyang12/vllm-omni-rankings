#!/usr/bin/env bash
# Run from an authenticated host with gh, git, Python 3 and flock on PATH.
set -euo pipefail
cd "$(dirname "$0")/.."
exec 9>"$(git rev-parse --git-dir)/open-pr-refresh.lock"
flock -n 9 || exit 0
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo 'Tracked files have local changes; leaving them intact.' >&2
  exit 1
fi
git pull --rebase origin main
python3 open-prs/update.py
git add open-prs/data.json
if ! git diff --cached --quiet; then
  git -c user.name=lishunyang12 -c user.email=lishunyang12@users.noreply.github.com \
    commit -m 'chore: refresh open PR leaderboard'
fi
git pull --rebase origin main
git -c credential.helper= -c 'credential.helper=!gh auth git-credential' push origin HEAD:main
# An ordinary authenticated push triggers this repository's legacy Pages build.
