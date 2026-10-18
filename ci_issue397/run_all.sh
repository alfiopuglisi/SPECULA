#!/bin/bash
# Run every repro mode in its own process and summarize: a crash shows as
# rc=134 (SIGABRT) with NSInvalidArgumentException in the output.
cd "$(dirname "$0")"
export MPLBACKEND=${MPLBACKEND:-macosx}
summary=""
for mode in "$@"; do
    echo "::group::$mode"
    python repro.py $mode > out_$mode.log 2>&1
    rc=$?
    tail -60 out_$mode.log
    echo "::endgroup::"
    crash=$(grep -c NSInvalidArgumentException out_$mode.log)
    summary="$summary\n$mode: rc=$rc NSInvalidArgumentException=$crash"
done
echo -e "===== SUMMARY$summary"
