#!/usr/bin/env bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Check if a GitHub user is a member of any specified team.
#
# Usage: check_github_team_membership.sh <gh_token> <username> <team1> [team2...]
#
# Prints "true" if the user is in any listed team, "false" otherwise.

set -euo pipefail

GH_TOKEN="$1"
USERNAME="$2"
ORG="qcom-ai-hub"

export GH_TOKEN

RESULT="false"
for team in "${@:3}"; do
  if gh api "orgs/${ORG}/teams/${team}/memberships/${USERNAME}" --jq '.state' 2>/dev/null; then
    echo "Author '${USERNAME}' is in ${ORG}/${team}." >&2
    RESULT="true"
    break
  fi
done

echo "$RESULT"
