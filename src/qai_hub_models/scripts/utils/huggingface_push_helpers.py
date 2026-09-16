# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from tempfile import TemporaryDirectory

import git  # noqa: TID251  We allow direct import of Git in scripts, since most users won't interact with them.
from git.exc import (  # noqa: TID251  We allow direct import of Git in scripts, since most users won't interact with them.
    GitCommandError,
)
from huggingface_hub import create_repo, create_tag, repo_exists, upload_folder

from qai_hub_models.cli.hf_common import is_hf_api_timeout_error, timeout_retry

HUGGINGFACE_ORG_NAME = "qualcomm"


def is_git_timeout_error(e: Exception) -> bool:
    return is_hf_api_timeout_error(e) or (
        isinstance(e, GitCommandError)
        and "The requested URL returned error: 429" in e.stderr
    )


def commit_and_push_to_hf(
    release_root_path: str | os.PathLike,
    hf_model_name: str,
    version: str,
    commit_description: str,
    hf_token: str,
    max_retries: int = 5,
    org_name: str = HUGGINGFACE_ORG_NAME,
) -> None:
    """
    Upload a folder to a HuggingFace repository and create a version tag.

    If the version tag already exists, the previous tag and associated commit
    are deleted before uploading the new content.

    Parameters
    ----------
    release_root_path
        Path to the folder containing files to upload.
    hf_model_name
        Name of the model repository on HuggingFace (under the qualcomm org).
    version
        Version string (e.g. '1.2.3r1') used for the git tag.
    commit_description
        Commit description for the upload. If '$QAIHM_TAG' is in this string,
        it will be replaced by the version tag.
    hf_token
        HuggingFace token with write access.
    max_retries
        Maximum number of retries for timeout errors.
    org_name
        HuggingFace organization name.
    """
    # Upload to hugging face. Retry if we hit timeouts.
    repo_id = f"{org_name}/{hf_model_name}"
    version_tag = f"v{version}"

    # If this tag exists already, delete previous tag and associated commit.
    if timeout_retry(lambda: repo_exists(repo_id, token=hf_token), max_retries):
        with TemporaryDirectory() as tmpdir:
            # Bare clone the repo (include only the history and no actual files)
            # so we can manipulate the repo git history cheaply.
            repo = timeout_retry(
                lambda: git.Repo.clone_from(
                    f"https://oauth2:{hf_token}@huggingface.co/{repo_id}",
                    tmpdir,
                    depth=2,
                    bare=True,
                ),
                max_retries,
                is_git_timeout_error,
            )

            # Only modify the old repo if the given tag exists already.
            if version_tag in repo.tags:
                tag = repo.tags[version_tag]
                main_branch = repo.heads.main
                remote = repo.remote("origin")
                if (
                    tag.commit == main_branch.commit
                    and len(main_branch.commit.parents) > 0
                ):
                    # If the tag maps to the last commit in the main branch,
                    # remove that commit from history so we can replace it.
                    previous_commit = main_branch.commit.parents[0]
                    repo.git.update_ref(main_branch.path, previous_commit.hexsha)
                    timeout_retry(
                        lambda: remote.push(main_branch, force=True),
                        max_retries,
                        is_git_timeout_error,
                    )
                # Delete the old tag.
                repo.delete_tag(tag)
                timeout_retry(
                    lambda: remote.push(
                        refspec=f"refs/tags/{version_tag}", delete=True
                    ),
                    max_retries,
                    is_git_timeout_error,
                )

    # Upload new commit and tag.
    timeout_retry(
        lambda: create_repo(repo_id=repo_id, exist_ok=True, token=hf_token),
        max_retries,
    )
    timeout_retry(
        lambda: upload_folder(
            folder_path=str(release_root_path),
            repo_id=repo_id,
            delete_patterns="*",  # Delete all previous files
            commit_message=version_tag,
            commit_description=commit_description.replace("$QAIHM_TAG", version_tag),
            token=hf_token,
        ),
        max_retries,
    )
    timeout_retry(
        lambda: create_tag(repo_id=repo_id, tag=version_tag, token=hf_token),
        max_retries,
    )
