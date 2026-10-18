'''Diagnose why the daemonized web frontend does not serve on macOS.'''
import os
import sys
import time
import subprocess
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = 18766


def http_ok(timeout=40):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/', timeout=5) as r:
                return r.status == 200
        except Exception as e:
            last = e
            time.sleep(1)
    print('last error:', repr(last))
    return False


FOREGROUND = '''
from specula.scripts import web_frontend as w
w.daemonize = lambda: None
w.start()
'''

FORK_LOGGED = '''
import os, sys
from specula.scripts import web_frontend as w
def daemonize():
    if os.fork() > 0:
        sys.exit(0)
    os.setsid()
    if os.fork() > 0:
        sys.exit(0)
    f = open(%r, 'w')
    os.dup2(f.fileno(), 1)
    os.dup2(f.fileno(), 2)
w.daemonize = daemonize
w.start()
'''

REAL = '''
from specula.scripts import web_frontend as w
w.start()
'''


def kill_port():
    subprocess.run(f'lsof -ti tcp:{PORT} | xargs kill -9', shell=True)
    time.sleep(1)


def variant(name, code, env_extra={}, background=False):
    print(f'\n===== {name}', flush=True)
    env = dict(os.environ, SPECULA_PORT=str(PORT), **env_extra)
    p = subprocess.Popen([sys.executable, '-c', code], cwd=HERE, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    ok = http_ok()
    print(f'{name}: HTTP ok = {ok}', flush=True)
    if p.poll() is None:
        p.terminate()
    out = p.communicate(timeout=30)[0]
    print(f'rc={p.returncode}\n--- output ---\n{out[-4000:]}', flush=True)
    kill_port()
    return ok


log = os.path.join(HERE, 'daemon.log')
variant('foreground (no fork)', FOREGROUND)
variant('forked, logging to file', FORK_LOGGED % log)
time.sleep(2)
print('--- daemon.log ---')
print(open(log).read()[-6000:] if os.path.exists(log) else '(missing)')
variant('real daemonize, OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES', REAL,
        {'OBJC_DISABLE_INITIALIZE_FORK_SAFETY': 'YES'})
print('--- recent crash reports ---')
subprocess.run('ls -t ~/Library/Logs/DiagnosticReports/ 2>/dev/null | head -5', shell=True)
