#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

API_PORT="${API_PORT:-8000}"
HTML_PORT="${HTML_PORT:-8765}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: python3 no esta disponible en el sistema."
  exit 1
fi

if [[ ! -d ".venv" ]]; then
  echo "[1/5] Creando entorno virtual en .venv ..."
  python3 -m venv .venv
fi

echo "[2/5] Activando entorno virtual ..."
source .venv/bin/activate

echo "[3/5] Instalando dependencias ..."
python3 -m pip install --upgrade pip >/dev/null
python3 -m pip install -r requirements.txt

echo "[4/5] Generando HTML ..."
python3 frontend/create_bundle_html.py --output outputs/bundle_csv_creator.html

echo "[5/5] Levantando servidor estatico (puerto ${HTML_PORT}) ..."
python3 -m http.server "${HTML_PORT}" --directory "$PROJECT_ROOT" >/tmp/hack2026_html_server.log 2>&1 &
HTML_PID=$!

cleanup() {
  if kill -0 "$HTML_PID" >/dev/null 2>&1; then
    kill "$HTML_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

echo ""
echo "Listo. Abre esta URL en tu navegador:"
echo "http://127.0.0.1:${HTML_PORT}/outputs/bundle_csv_creator.html"
echo ""
echo "Backend de inferencia arrancando en http://127.0.0.1:${API_PORT} ..."
echo "Para detener todo, presiona Ctrl+C."
echo ""

python3 frontend/app.py
