#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ -n "${PYTHON:-}" ]]; then
  exec "$PYTHON" -m modiff.install "$@"
fi
if command -v python3 >/dev/null 2>&1; then
  exec python3 -m modiff.install "$@"
fi

os="$(uname -s)"
arch="$(uname -m)"
case "$os/$arch" in
  Linux/x86_64)
    asset="uv-x86_64-unknown-linux-gnu.tar.gz"
    expected="6426a73c3837e6e2483ee344cbc00f36394d179afcba6183cb77437e67db4af0"
    ;;
  Darwin/arm64)
    asset="uv-aarch64-apple-darwin.tar.gz"
    expected="8f7fbf1708399b921857bce71e1d60f0d3ccf52a30caebc1c1a2f175dce13ab6"
    ;;
  *)
    echo "No verified bootstrap toolchain is available for $os/$arch. Install Python 3.12 and rerun." >&2
    exit 2
    ;;
esac

bootstrap=".modiff/bootstrap"
mkdir -p "$bootstrap"
archive="$bootstrap/$asset"
url="https://github.com/astral-sh/uv/releases/download/0.11.26/$asset"
if [[ ! -f "$archive" ]]; then
  echo "Downloading the verified MoDiff bootstrap tool..."
  if command -v curl >/dev/null 2>&1; then
    curl -fL "$url" -o "$archive.part"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$archive.part" "$url"
  else
    echo "curl or wget is required for first-time bootstrap." >&2
    exit 2
  fi
  mv "$archive.part" "$archive"
fi
if command -v sha256sum >/dev/null 2>&1; then
  actual="$(sha256sum "$archive" | cut -d' ' -f1)"
else
  actual="$(shasum -a 256 "$archive" | cut -d' ' -f1)"
fi
if [[ "$actual" != "$expected" ]]; then
  echo "Bootstrap hash verification failed; remove $archive and retry." >&2
  exit 2
fi
rm -rf "$bootstrap/uv"
mkdir -p "$bootstrap/uv"
tar -xf "$archive" -C "$bootstrap/uv"
uv_bin="$(find "$bootstrap/uv" -type f -name uv -perm -u+x | head -n 1)"
export UV_PYTHON_INSTALL_DIR="$PWD/.modiff/tools/python"
"$uv_bin" python install 3.12
python_bin="$(find "$UV_PYTHON_INSTALL_DIR" -type f -path '*/bin/python3.12' | head -n 1)"
exec "$python_bin" -m modiff.install "$@"
