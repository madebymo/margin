#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
frontend_dir="$repo_root/frontend"
dist_path="backend/src/tutor/api/static/dist"

npm --prefix "$frontend_dir" ci
npm --prefix "$frontend_dir" run build

if ! git -C "$repo_root" diff --exit-code -- "$dist_path"; then
    printf '%s\n' "Frontend distribution is stale; commit the rebuilt tracked files."
    exit 1
fi

untracked="$(
    git -C "$repo_root" status --porcelain --untracked-files=all -- "$dist_path" \
        | sed -n 's/^?? //p'
)"
if [[ -n "$untracked" ]]; then
    printf '%s\n' "Frontend distribution has untracked build files:"
    printf '%s\n' "$untracked"
    exit 1
fi
