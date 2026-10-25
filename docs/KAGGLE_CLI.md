# Kaggle CLI workflow

The project uses the official `kaggle==2.2.5` package in `.venv-kaggle/`. Run `./scripts/kaggle` from the project root; it resolves the isolated environment without activation. `training/kaggle_cli_requirements.txt` records the installed dependencies. No model inference or GPU training runs locally.

Authentication uses the user's token in `~/.kaggle/access_token`, created interactively with `python3 scripts/configure_kaggle_auth.py`. The script hides input, writes owner-only permissions, refuses overwrite and never prints the token. Credentials and the virtual environment are ignored. Do not put credentials into notebooks, metadata, command arguments, logs, or chat.

Authenticated access was verified on 23 October 2026 by reading the private pilot's COMPLETE status. Source and metadata for saved version 2 were downloaded to `training/kaggle_cli/pilot_source/`. Step-50 and step-100 inference adapters and manifests were downloaded separately. Their completion-marker hashes, saved pilot identity and downloaded file hashes match; see `training/kaggle_cli/checkpoint_download_verification.json`. This subset excludes optimizer/RNG state and is not a full recovery-checkpoint validation.

```bash
./scripts/kaggle --version
./scripts/kaggle kernels status thanvirdiouf/trotsky-qlora-500-example-pilot
./scripts/kaggle kernels files thanvirdiouf/trotsky-qlora-500-example-pilot/2 --page-size 200
./scripts/kaggle kernels pull thanvirdiouf/trotsky-qlora-500-example-pilot/2 -p training/kaggle_cli/pilot_source -m
./scripts/kaggle quota
```

For new inference notebooks, prepare a separate folder containing `kernel-metadata.json` and the notebook. Preserve `is_private: true`, use a version-pinned pilot output input, then submit with `kernels push -p FOLDER --accelerator NvidiaTeslaT4`. Monitor with `kernels status`, `kernels logs`, and download outputs with `kernels output`. CLI pushes run notebooks unless `--no-run` is given. Never push the training pilot to perform an inference comparison.

Official documentation: [Kaggle CLI](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md), [notebook metadata](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md).
