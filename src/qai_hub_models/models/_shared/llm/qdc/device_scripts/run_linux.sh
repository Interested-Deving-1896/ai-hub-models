#!/bin/bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

set -e
# Keep pipeline exit status tied to the real command rather than `tee`, which
# always succeeds; otherwise the tees below would mask genie failures.
set -o pipefail

# Drop per-job state on exit (dedicated-pool devices are reused).
cleanup_device() {
    rm -rf /data/local/tmp/TestContent/genie_bundle 2>/dev/null || true
    rm -rf /data/local/tmp/qairt 2>/dev/null || true
    rm -f /data/local/tmp/qairt.zip 2>/dev/null || true
}
trap cleanup_device EXIT

# Drop stale logs from a prior job on this shared device.
rm -rf /data/local/tmp/QDC_logs
mkdir -p /data/local/tmp/QDC_logs
exec > >(tee /data/local/tmp/QDC_logs/script.log) 2>&1

mount -o rw,remount /

cd /data/local/tmp/TestContent/genie_bundle

# Always re-download: dedicated-pool devices are reused, so a partial extract
# from a previous job would otherwise silently corrupt this run.
rm -rf /data/local/tmp/qairt
echo "=== Pre-download connectivity check ==="
echo "Pinging google.com before QAIRT SDK download..."
ping -c 1 google.com && echo "Pre-download ping: SUCCESS" || echo "Pre-download ping: FAILED"

curl -L -J --output /data/local/tmp/qairt.zip \
  https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/{QAIRT_VERSION}/v{QAIRT_VERSION}.zip

echo "=== Post-download connectivity check ==="
echo "Pinging google.com after QAIRT SDK download..."
ping -c 1 google.com && echo "Post-download ping: SUCCESS" || echo "Post-download ping: FAILED"

unzip -q /data/local/tmp/qairt.zip -d /data/local/tmp || {
    echo "unzip failed, retrying once" >&2
    rm -rf /data/local/tmp/qairt
    unzip -q /data/local/tmp/qairt.zip -d /data/local/tmp
}

export QAIRT_HOME=/data/local/tmp/qairt/{QAIRT_VERSION}
export PATH=$QAIRT_HOME/bin/aarch64-oe-linux-gcc11.2:$PATH
export LD_LIBRARY_PATH=$QAIRT_HOME/lib/aarch64-oe-linux-gcc11.2
export ADSP_LIBRARY_PATH=$QAIRT_HOME/lib/hexagon-{HEXAGON_VERSION}/unsigned

# genie-t2t-run fails randomly on QDC devices; give each invocation one retry
# before letting the failure propagate.
genie_retry() {
    "$@" || {
        echo "genie_retry: command failed, retrying once: $*" >&2
        "$@"
    }
}

# Run genie (capture initial output, including stderr)
genie_retry genie-t2t-run -c genie_config.json --prompt_file sample_prompt.txt 2>&1 | tee /data/local/tmp/QDC_logs/genie.log

# Run profiling iterations
for i in $(seq 1 {NUM_TRIALS}); do
    sed -i "s/\"seed\": [0-9]*/\"seed\": $i/" genie_config.json
    genie_retry genie-t2t-run -c genie_config.json --prompt_file sample_prompt.txt \
      --profile /data/local/tmp/QDC_logs/profile${i}.json
done

# Run evaluation over all prompt files
PROMPT_DIR=/data/local/tmp/TestContent/genie_bundle/prompts
EVAL_OUTPUT_FILE=/data/local/tmp/QDC_logs/eval_outputs.txt

if [ -d "$PROMPT_DIR" ]; then
    # Switch to power_saver perf_profile: sustained burst thermal-throttles and kills the eval loop on QDC.
    sed -i 's/"perf_profile": "[^"]*"/"perf_profile": "power_saver"/' htp_backend_ext_config.json
    true > "$EVAL_OUTPUT_FILE"
    for prompt_file in "$PROMPT_DIR"/prompt_*.txt; do
        idx=$(basename "$prompt_file" | sed 's/prompt_\([0-9]*\)\.txt/\1/')
        echo "===EVAL_IDX_${idx}===" | tee -a "$EVAL_OUTPUT_FILE"
        genie_retry genie-t2t-run -c genie_config.json --prompt_file "$prompt_file" 2>&1 | tee -a "$EVAL_OUTPUT_FILE"
        # Short inter-prompt cooldown to keep the HTP from thermal-throttling.
        sleep 3
    done
fi

mount -o rw,remount /
