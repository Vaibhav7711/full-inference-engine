# Run this project on Kaggle's T4, from VS Code

Kaggle notebooks don't expose SSH by default, and third-party tunnel services
(ngrok, etc.) generally require card verification for the features this would
need. Instead, this uses VS Code's own **Remote Tunnels** feature: it runs the
`code` CLI inside the Kaggle container, tunnels it out through Microsoft's
relay, and your local VS Code connects to it after a GitHub login. No SSH
keys, no card, no third-party account.

## One-time local setup

In VS Code, install the **Remote - Tunnels** extension
(`ms-vscode.remote-server`) if you don't have it.

## Every session

1. Open (or upload) `scripts/kaggle_notebook.ipynb` as a Kaggle notebook.
2. In the right-hand **Settings** panel: **Accelerator: GPU T4 x2**, **Internet: On**.
3. Run all cells in order. The last cell (`scripts/kaggle_vscode_tunnel.sh`)
   prints something like:

   ```
   To grant access to the server, please log into https://github.com/login/device
   and use code XXXX-XXXX
   ```

4. Open that URL in a browser, enter the code, and authorize with your GitHub
   account (the same account you'll sign into VS Code with). Once it
   authenticates, it prints a line confirming the tunnel is open under the
   name `kaggle-fol`. Leave that cell running — stopping it kills the tunnel
   and the session.

5. In VS Code: `Cmd+Shift+P` → **Remote Tunnels: Connect to Tunnel...** → sign
   in with the same GitHub account if prompted → select `kaggle-fol`.

6. Once connected: **File → Open Folder** → `/kaggle/working/fol`.

7. You're now editing and running on the actual Kaggle T4 from your local VS
   Code window — open a terminal in that window and run:

   ```bash
   python -m pytest -v
   python -m engine.cli --prompt "Explain KV caching in one sentence." --max-new-tokens 32
   ```

## Things that will bite you

- **Kaggle sessions are ephemeral.** `/kaggle/working` (and anything installed)
  is wiped when the session ends or restarts. Commit and `git push` from the
  Remote Tunnels terminal before you stop the notebook, or download `results/`
  output files you care about.
- **Idle/time limits.** Kaggle GPU sessions have a ~9h continuous limit and a
  weekly GPU quota (~30h). Keeping the tunnel cell "running" counts as active,
  but idle disconnects from the notebook UI can still kill it — check back
  periodically.
- **Re-authenticate each session.** The tunnel is tied to that container
  instance, so you repeat the device-code login every time you restart the
  notebook (the tunnel *name* `kaggle-fol` stays the same, so VS Code's history
  entry for it still works — just reconnect after it comes back up).
- **Don't reinstall torch.** `scripts/setup_kaggle.sh` intentionally installs
  from `requirements-colab.txt`, which excludes torch — Kaggle's preinstalled
  build is already matched to its CUDA driver, same constraint as Colab.
