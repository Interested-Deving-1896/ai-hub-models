# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Shared base code for AI Hub Models.

Each subfolder is a node in an acyclic dependency graph (DAG). Folders may
depend on other shared folders but must not form cycles.

Folders with dependencies contain a manifest.yaml listing their direct shared
folder dependencies (templates: key). Folders without dependencies have no
manifest.

Each folder contains a requirements.txt listing its direct pip dependencies
(not inherited from parent folders or transitive deps). Dependencies are
installed leaf-first during setup.
"""
