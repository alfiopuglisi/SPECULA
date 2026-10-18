'''
Cross-platform checks for issue #81 (multiprocessing on macOS and Windows).

Usage: python check.py fix|main

With "main" (unfixed code) the bug is expected to be reproduced on platforms
using the 'spawn' start method / without os.fork(); with "fix" it must be gone.
Exit code is non-zero if any expectation is not met.
'''
import os
import sys
import time
import shutil
import subprocess
import multiprocessing as mp
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = sys.argv[1] == 'fix'
SPAWN = mp.get_start_method() != 'fork'
HAS_FORK = hasattr(os, 'fork')
PORT = 18765
os.environ['SPECULA_PORT'] = str(PORT)
NEW_MSG = 'A SPECULA simulation was started while a child process was being initialized'
OLD_MSG = 'current process has finished its bootstrapping phase'

failures = []


def check(name, ok, out=''):
    print(f'[{"PASS" if ok else "FAIL"}] {name}', flush=True)
    if not ok:
        failures.append(name)
        print('----- output -----\n' + out[-6000:] + '\n------------------', flush=True)


def run(cmd, timeout=900):
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True,
                           errors='replace', timeout=timeout)
        rc, out = r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired as e:
        rc, out = 'timeout', str(e.stdout) + str(e.stderr)
    print(f'$ {" ".join(cmd)}  -> rc={rc} in {time.time() - t0:.1f} s', flush=True)
    return rc, out


def http_ok(timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/', timeout=5) as r:
                return r.status == 200
        except Exception:
            time.sleep(1)
    return False


print(f'platform={sys.platform} start_method={mp.get_start_method()} has_fork={HAS_FORK} code={"fix" if FIX else "main"}')

# 1. Guarded script: display server must start and still be alive at the end
rc, out = run([sys.executable, 'guarded.py', 'p.yml'])
check('guarded script: simulation ok and display server alive',
      rc == 0 and 'Traceback' not in out and 'Server process terminated' in out
      and 'SIMULATION DONE' in out, out)

# 2. specula CLI (guarded entry point)
rc, out = run([shutil.which('specula'), 'p.yml', '--cpu'])
check('specula CLI: simulation ok and display server alive',
      rc == 0 and 'Traceback' not in out and 'Server process terminated' in out, out)

# 3. Unguarded script (the case reported in issue #81)
rc, out = run([sys.executable, 'unguarded.py', 'p.yml'])
nread = out.count('Reading parameters')
if not SPAWN:
    check('unguarded script (fork): works as before',
          rc == 0 and 'Traceback' not in out and 'Server process terminated' in out, out)
elif FIX:
    check('unguarded script (spawn, fix): clear early error, simulation not rebuilt in child',
          NEW_MSG in out and OLD_MSG not in out and nread == 1 and 'SIMULATION DONE' in out, out)
else:
    check('unguarded script (spawn, main): bug reproduced (bootstrapping error, simulation rebuilt in child)',
          OLD_MSG in out and nread == 2, out)

# 4. Web frontend start/stop
start, stop = shutil.which('specula_frontend_start'), shutil.which('specula_frontend_stop')
if HAS_FORK:
    rc, out = run([start])
    ok = rc == 0 and http_ok()
    rc2, out2 = run([stop])
    time.sleep(2)
    check('web frontend (fork): daemonized, serves HTTP, stops', ok and rc2 == 0 and not http_ok(timeout=3),
          out + out2)
else:
    p = subprocess.Popen([start], cwd=HERE, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, errors='replace')
    ok = http_ok()
    p.terminate()
    out = p.communicate(timeout=60)[0]
    rc2, out2 = run([stop])
    if FIX:
        check('web frontend (no fork, fix): runs in foreground and serves HTTP; stop explains Ctrl-C',
              ok and 'running in foreground' in out and rc2 == 0 and 'Ctrl-C' in out2, out + out2)
    else:
        check("web frontend (no fork, main): bug reproduced (os has no attribute 'fork')",
              not ok and "has no attribute 'fork'" in out, out + out2)

print(f'\n{len(failures)} failed: {failures}' if failures else '\nAll checks passed')
sys.exit(1 if failures else 0)
