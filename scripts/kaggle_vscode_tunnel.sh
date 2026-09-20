#!/usr/bin/env bash
set -euo pipefail

# Expose this Kaggle container to VS Code via Microsoft's built-in Remote
# Tunnels feature. No SSH keys, no ngrok, no card verification: just a
# GitHub (or Microsoft) account login via device code.
#
# Run inside a Kaggle notebook cell with Internet enabled:
#   !bash scripts/kaggle_vscode_tunnel.sh

INSTALL_DIR="/kaggle/working/.vscode-cli"
mkdir -p "$INSTALL_DIR"

if [ ! -x "$INSTALL_DIR/code" ]; then
  curl -Lk "https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64" \
    -o "$INSTALL_DIR/vscode_cli.tar.gz"
  tar -xzf "$INSTALL_DIR/vscode_cli.tar.gz" -C "$INSTALL_DIR"
  rm "$INSTALL_DIR/vscode_cli.tar.gz"
fi

echo "Starting VS Code tunnel. Open the printed https://github.com/login/device"
echo "URL and enter the code to authenticate, then keep this cell running."
echo

exec "$INSTALL_DIR/code" tunnel --accept-server-license-terms --name kaggle-fol
