#!/bin/bash
# Is the problem the macOS runner network, Flask, or the frontend itself?
cd "$(dirname "$0")"

probe() {   # name, port, command...
    name=$1; port=$2; shift 2
    echo "===== $name"
    "$@" > srv.log 2>&1 &
    pid=$!
    sleep 8
    lsof -nP -iTCP:$port -sTCP:LISTEN
    curl -sv -m 10 -o /dev/null -w "HTTP code: %{http_code}\n" http://127.0.0.1:$port/ 2>&1 | grep -E "Connected|HTTP code|timed out|refused|^< HTTP"
    kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null
    echo "--- server output"; tail -15 srv.log
}

probe "stdlib http.server" 18770 python -m http.server 18770 --bind 0.0.0.0
probe "plain flask" 18771 python -c "
from flask import Flask
app = Flask('x')
app.route('/')(lambda: 'ok')
app.run(host='0.0.0.0', port=18771)"
probe "frontend 0.0.0.0" 18772 env SPECULA_PORT=18772 python -c "
from specula.scripts import web_frontend as w
w.daemonize = lambda: None
w.start()"
probe "frontend 127.0.0.1" 18773 env SPECULA_PORT=18773 python -c "
from specula.scripts import web_frontend as w
w.daemonize = lambda: None
run = w.socketio.run
w.socketio.run = lambda app, **kw: run(app, **dict(kw, host='127.0.0.1'))
w.start()"
probe "frontend, flask-socketio debug log" 18774 env SPECULA_PORT=18774 python -c "
import logging; logging.basicConfig(level=logging.DEBUG)
from specula.scripts import web_frontend as w
w.daemonize = lambda: None
w.start()"
pip list 2>/dev/null | grep -iE "flask|werkzeug|socketio|engineio|eventlet|gevent|simple-websocket"
/usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate
