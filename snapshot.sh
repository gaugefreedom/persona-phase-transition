#!/usr/bin/env bash
# Usage:
#   bash snapshot.sh                 # full -> writes snapshot.txt
#   bash snapshot.sh quick           # quick -> writes snapshot-quick.txt
#   bash snapshot.sh full out.txt    # full  -> writes out.txt
#   bash snapshot.sh quick out.txt   # quick -> writes out.txt

set -euo pipefail

MODE="${1:-full}"                         # "full" or "quick"
DEFAULT_OUT_FULL="snapshot.txt"
DEFAULT_OUT_QUICK="snapshot-quick.txt"
OUTFILE="${2:-$([ "$MODE" = "quick" ] && echo "$DEFAULT_OUT_QUICK" || echo "$DEFAULT_OUT_FULL")}"

ROOT="."
MAX_SIZE=$((5 * 1024 * 1024))            # 5 MB per file cap

# From here on, write directly to $OUTFILE (overwrite)
exec >"$OUTFILE"

echo "===== PROJECT STRUCTURE ====="

# Exclusion pattern for tree (includes the dynamic $OUTFILE)
TREE_IGNORE=".*|node_modules|dist|target|$OUTFILE|CONTRIBUTING.md|README.md|LICENSE"

if command -v tree >/dev/null 2>&1; then
  tree -a -I "$TREE_IGNORE" "$ROOT" || true
else
  # Fallback listing if tree isn't installed
  find "$ROOT" -type d \
    -not -path "*/.*" \
    -not -path "*/node_modules/*" \
    -not -path "*/dist/*" \
    -not -path "*/target/*" \
    -print
fi

# QUICK MODE: only dump core files and exit
if [ "$MODE" = "quick" ]; then
  echo
  echo "===== CORE FILES ====="
  for f in README.md app/src/main.tsx app/src/App.tsx \
           src-tauri/src/main.rs src-tauri/src/api.rs \
           src-tauri/Cargo.toml src-tauri/tauri.conf.json; do
    [ -f "$f" ] || continue
    echo
    echo "----- FILE: $f -----"
    sed -e 's/\r$//' "$f"
  done
  exit 0
fi

echo
echo "===== FILE CONTENTS ====="

# Full mode: dump all text-like files, with exclusions
find "$ROOT" -type f \
  -not -path "*/.*" \
  -not -path "*/node_modules/*" \
  -not -path "*/dist/*" \
  -not -path "*/target/*" \
  -not -name "CONTRIBUTING.md" \
  -not -name "README.md" \
  -not -name "LICENSE" \
  -not -name "$OUTFILE" \
  -print0 \
| while IFS= read -r -d '' file; do
    # Size check (Linux stat -c, macOS/BSD stat -f)
    size=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file")
    if [ "$size" -gt "$MAX_SIZE" ]; then
      echo
      echo "----- FILE: $file (skipped: > ${MAX_SIZE} bytes) -----"
      continue
    fi

    # Only dump text-like files
    mime=$(file --mime-type -b "$file" 2>/dev/null || echo text/plain)
    case "$mime" in
      text/*|application/json|application/javascript|application/xml)
        echo
        echo "----- FILE: $file -----"
        sed -e 's/\r$//' "$file"
        ;;
      *)
        echo
        echo "----- FILE: $file (skipped: $mime) -----"
        ;;
    esac
  done
