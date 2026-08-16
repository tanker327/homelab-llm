#!/bin/bash
# Stable systemd entry point: execs whichever launcher is named in
# production-engine.conf, so switching engines (e.g. NVFP4 <-> FP8 for
# vision) is a one-line conf edit + `sudo systemctl restart llama-server`.
set -eu

DIR="$(cd "$(dirname "$0")" && pwd)"
CONF="$DIR/production-engine.conf"

LAUNCHER="$(grep -v '^\s*#' "$CONF" | grep -v '^\s*$' | head -1 | tr -d '[:space:]')"

if [ -z "$LAUNCHER" ] || [ ! -x "$DIR/$LAUNCHER" ]; then
  echo "start-production.sh: invalid launcher '$LAUNCHER' in $CONF" >&2
  exit 1
fi

echo "start-production.sh: launching $LAUNCHER"
exec "$DIR/$LAUNCHER"
