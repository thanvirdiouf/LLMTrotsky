#!/usr/bin/env python3
"""Save a user-entered Kaggle personal token without printing or logging it."""
from getpass import getpass
import os
from pathlib import Path


def main():
    if not os.isatty(0):
        raise SystemExit('Run this script in your own interactive terminal.')
    token = getpass('Paste Kaggle API token (hidden): ').strip()
    if not token or any(c.isspace() for c in token):
        raise SystemExit('Expected one nonempty token without whitespace; no file written.')
    directory = Path.home() / '.kaggle'
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = directory / 'access_token'
    if path.exists():
        raise SystemExit(f'{path} already exists; no credential overwritten.')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as stream:
        stream.write(token + '\n')
    print(f'Saved token to {path} with owner-only permissions. Token was not displayed.')


if __name__ == '__main__':
    main()
