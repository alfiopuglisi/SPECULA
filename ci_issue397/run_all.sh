#!/bin/bash
# Run every repro mode in its own process and summarize.
# Usage: run_all.sh main|fix mode...
# With "fix", all specula* modes must succeed and produce the diagram;
# with "main", specula_nodisp_then_plot is expected to crash.
cd "$(dirname "$0")"
export MPLBACKEND=${MPLBACKEND:-macosx}
code=$1; shift
summary=""
failed=0
for mode in "$@"; do
    rm -f diagram.png
    echo "::group::$mode"
    python repro.py $mode > out_$mode.log 2>&1
    rc=$?
    grep -vE "LoopControl|output_name" out_$mode.log | tail -60
    echo "::endgroup::"
    ok="(info)"
    case "$code:$mode" in
        fix:specula*)
            if [ $rc -eq 0 ] && grep -q "diagram size" out_$mode.log; then ok=PASS; else ok=FAIL; failed=1; fi ;;
        main:specula_nodisp_then_plot)
            if [ $rc -ne 0 ]; then ok="PASS (crash reproduced)"; else ok="FAIL (no crash)"; failed=1; fi ;;
    esac
    summary="$summary\n$mode: rc=$rc $ok"
done
echo -e "===== SUMMARY ($code)$summary"
exit $failed
