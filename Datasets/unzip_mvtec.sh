bash -lc 'set -euo pipefail
DEST="../Data/Industrial_Dataset/MVTecAD"
RAW="$DEST/raw"
mkdir -p "$DEST"

echo "== Sample contents of bottle.tar.xz =="
if [ -f "$RAW/bottle.tar.xz" ]; then
  tar -tf "$RAW/bottle.tar.xz" | head -n 25 || true
else
  echo "bottle.tar.xz not found in $RAW"; ls -al "$RAW" || true
fi

echo "== Extracting all .tar.xz archives to $DEST =="
shopt -s nullglob
for f in "$RAW"/*.tar.xz; do
  echo "Extracting $(basename "$f")"
  tar -xJf "$f" -C "$DEST"
done

echo "== Top-level classes under $DEST =="
ls -1 "$DEST" | sort || true

echo "== Verify structure for a few classes =="
for c in bottle cable carpet; do
  if [ -d "$DEST/$c" ]; then
    echo "-- $c --"
    find "$DEST/$c" -maxdepth 2 -type d | sed "s|$DEST/||" | sort
  fi
done
'