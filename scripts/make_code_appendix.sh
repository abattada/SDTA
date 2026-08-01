#!/usr/bin/env bash
#
# Build the anonymous code appendix archive, then verify that it is actually
# anonymous and actually clean.
#
# The repository directory itself must never be zipped for submission: it holds
# a live `.git` (real remote URL, real author name and email in every commit)
# plus the gitignored `data/` and `outputs/` trees, which are hundreds of MB.
# This script instead copies exactly the files git tracks or would offer to
# track, taken from the working tree, so the archive matches what is on disk
# (including uncommitted fixes) while excluding .git, data/, outputs/ and
# __pycache__.
#
# Usage:
#   scripts/make_code_appendix.sh [OUTPUT.zip]
#
#   OUTPUT.zip   archive path (default: <repo root>/code_appendix.zip)
#
# Exits non-zero if any verification check fails; the archive is then deleted,
# so a failing run can never leave a publishable-looking file behind.

set -euo pipefail

REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
OUTPUT="${1:-${REPO_ROOT}/code_appendix.zip}"
# absolute, so that the verification step can still find it after chdir
OUTPUT="$(cd "$(dirname "${OUTPUT}")" && pwd)/$(basename "${OUTPUT}")"
PREFIX="code_appendix/"

# Strings that must not appear anywhere in the archive.
LEAK_PATTERN='github\.com|abat|@gmail|/home/|/local/'

# Third-party URLs that are legitimate attributions, not deanonymizing links.
# Every one of these is an upstream baseline or dataset repository cited in
# THIRD_PARTY.md / docs/datasets.md. Anything else matching LEAK_PATTERN fails.
ALLOWED_PATTERN='github\.com/(yuqinie98/PatchTST|thuml/(TimeSiam|SimMTM)|ustc-time-series/(TimeDART|TimeMAE)|xingyu617/SimTS_Representation_Learning|zhouhaoyi/ETDataset|ts-kim/RevIN)'

# This script never enters the archive: it is packaging tooling, not paper
# implementation, and it has to spell the forbidden strings out in order to
# look for them, so shipping it would smuggle those very strings past review.
SELF="${PREFIX}scripts/$(basename "${BASH_SOURCE[0]}")"

fail() {
    echo "FAIL: $*" >&2
    rm -f "${OUTPUT}"
    exit 1
}

# --- build ------------------------------------------------------------------

# The appendix must match the files on disk, not a commit: fixes made in the
# working tree and configs that have not been `git add`ed still belong in the
# submission. `git ls-files --cached --others --exclude-standard` is exactly the
# set git tracks or would offer to track, so it picks up uncommitted edits and
# new files while still honouring .gitignore (data/, outputs/, __pycache__) and
# never reaching into .git itself.
echo "Staging the working tree -> ${OUTPUT}"
STAGE="$(mktemp -d)"
mkdir -p "${STAGE}/${PREFIX}"
( cd "${REPO_ROOT}" \
  && git ls-files -z --cached --others --exclude-standard \
  | xargs -0 -r -I{} cp --parents -- "{}" "${STAGE}/${PREFIX}" )
rm -f "${STAGE}/${SELF}"

rm -f "${OUTPUT}"
# -X strips the Unix extra fields (numeric UID/GID), a weak identity signal.
( cd "${STAGE}" && zip -qrX "${OUTPUT}" "${PREFIX%/}" )
rm -rf "${STAGE}"

# --- verify -----------------------------------------------------------------

WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
unzip -q "${OUTPUT}" -d "${WORK}"

# grep from inside the temp dir so that the temp path itself (which may live
# under /home or /tmp) can never be mistaken for a leak in the content.
cd "${WORK}"

[ -d "${PREFIX}" ] || fail "archive does not contain a single ${PREFIX} root"

echo "Checking: no repository metadata, no build artefacts"
found="$(find . -name '.git' -print)"
[ -z "${found}" ] || fail "archive contains .git:"$'\n'"${found}"
found="$(find . -name '__pycache__' -print)"
[ -z "${found}" ] || fail "archive contains __pycache__:"$'\n'"${found}"
found="$(find . -name '*.pyc' -print)"
[ -z "${found}" ] || fail "archive contains .pyc files:"$'\n'"${found}"

echo "Checking: no dataset or result trees"
for heavy in data outputs; do
    if [ -e "${PREFIX}${heavy}" ]; then
        fail "archive contains ${heavy}/ (must be rebuilt by the user)"
    fi
done

echo "Checking: no leaks in file or directory NAMES"
# Contents are scanned below, but a path is just as revealing: docs/abat_notes.md
# or docs/<CJK>.md would otherwise sail through. Paths are checked first because
# a bad name is the cheapest mistake to make and the easiest to miss in review.
badnames="$(
    { find . -mindepth 1 -iregex '.*\(github\|gitlab\|abat\|gmail\).*' -print
      find . -mindepth 1 -print | LC_ALL=C grep -P '[^\x00-\x7F]' || true
    } | sort -u )"
[ -z "${badnames}" ] || fail "deanonymizing or non-ASCII path names in archive:"$'\n'"${badnames}"

echo "Checking: no CJK or other locale-revealing text"
# Under double-blind review a Traditional/Simplified Chinese, Japanese or Korean
# string is a regional signal even though it names nobody. Typographic Unicode
# (arrows, en dashes, the s-caron of "Demsar") is fine and is not matched here.
cjk="$(grep -ralP '[\x{2E80}-\x{9FFF}\x{AC00}-\x{D7AF}\x{F900}-\x{FAFF}\x{FF00}-\x{FFEF}]' . || true)"
[ -z "${cjk}" ] || fail "archive contains CJK text (locale leak):"$'\n'"${cjk}"

echo "Checking: no deanonymizing strings"
# Two passes: find candidate lines, blank out the allowlisted third-party URLs,
# then re-test. A line carrying both a legitimate URL and a leak still fails.
# `tr -d '\0'` matters: a match inside a binary file carries NUL bytes, which
# make the downstream sed/grep treat their own input as binary and drop the
# line, so the leak would be found and then silently discarded.
leaks="$(grep -raniE "${LEAK_PATTERN}" . \
    | tr -d '\0' \
    | sed -E "s#${ALLOWED_PATTERN}##g" \
    | grep -aiE "${LEAK_PATTERN}" || true)"
[ -z "${leaks}" ] || fail "deanonymizing strings in archive:"$'\n'"${leaks}"

# --- report -----------------------------------------------------------------

files="$(find "${PREFIX}" -type f | wc -l)"
dirs="$(find "${PREFIX}" -mindepth 1 -type d | wc -l)"
unpacked="$(du -sh --apparent-size "${PREFIX}" | cut -f1)"
packed="$(du -h --apparent-size "${OUTPUT}" | cut -f1)"

echo
echo "OK: ${OUTPUT}"
echo "    source     working tree at $(git -C "${REPO_ROOT}" rev-parse --short HEAD)$(
    [ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ] && echo " plus uncommitted changes")"
echo "    files      ${files} in ${dirs} directories"
echo "    size       ${packed} zipped, ${unpacked} unpacked"
