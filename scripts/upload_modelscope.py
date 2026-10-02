"""Publish the built package to ModelScope.  Run: python scripts/upload_modelscope.py [--repo owner/name] [--dry-run]

Two things gate an upload, and neither is ceremonial:

* **the package must pass its own self-check first.**  `examples/selfcheck.py` replays every task in
  the built directory through the full live path and compares it to the record the builder wrote.  It
  needs the base checkpoint, so it is the one test that could not run at build time on a machine
  without one.  Uploading without it publishes weights nobody read out of.
* **the credential comes from the environment, never from this file or a command line.**  A token in
  argv lands in shell history and in any log of the run; `--token` is deliberately absent.  Set
  MODELSCOPE_SDK_TOKEN, or put it in `.env` (already gitignored) and source it.

The repo is created if it does not exist, and afterwards the script lists the remote files and
compares them to what was uploaded: a partial upload that still shows a green "success" line is the
failure mode worth spending a request to rule out.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "dist" / "modelscope"
DEFAULT_REPO = "陆白tuber/rwkv7-system1"
BASE_FILE = "rwkv7-g1d-0.1b-20260129-ctx8192.pth"


def token() -> str:
    t = os.environ.get("MODELSCOPE_SDK_TOKEN") or os.environ.get("MODELSCOPE_API_TOKEN")
    if t:
        return t.strip()
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8", errors="replace").splitlines():
            k, _, v = line.partition("=")
            if k.strip() in ("MODELSCOPE_SDK_TOKEN", "MODELSCOPE_API_TOKEN") and v.strip():
                return v.strip()
    # Third source: a token the SDK was configured with (`modelscope login --token ...`).  Note what
    # is deliberately NOT tried: this machine's `~/.modelscope/credentials/session` is a 32-hex login
    # *session id*, not an SDK token, and `HubApi().token` is empty -- substituting one credential
    # for another until a request happens to succeed is not how a write to someone's account should
    # be authorised.
    try:
        from modelscope.hub.api import HubApi

        persisted = getattr(HubApi(), "token", None)
        if isinstance(persisted, str) and persisted.strip():
            return persisted.strip()
    except Exception:
        pass
    raise SystemExit(
        "no ModelScope SDK token in the environment.\n"
        "  get one from https://www.modelscope.cn/my/myaccesstoken and either\n"
        "    export MODELSCOPE_SDK_TOKEN=...   (nothing pasted into a chat or an argv)\n"
        "    modelscope login --token ...      (this script will read it back)\n"
        "  or add it to .env, which .gitignore already excludes.")


def selfcheck() -> None:
    ckpt = ROOT / "weights" / BASE_FILE
    if not ckpt.exists():
        raise SystemExit(f"{ckpt} is missing, so the package cannot be verified before upload")
    print("--- verifying the package end to end (every task, full live path)")
    res = subprocess.run([sys.executable, "examples/selfcheck.py", str(ckpt)], cwd=PKG,
                         capture_output=True, text=True)
    tail = res.stdout.strip().splitlines()[-4:]
    print("\n".join(tail))
    if res.returncode != 0:
        raise SystemExit("self-check failed; nothing was uploaded")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--dry-run", action="store_true",
                    help="verify and list what would go up; contact no server")
    args = ap.parse_args(argv)

    if not PKG.exists():
        raise SystemExit(f"{PKG} does not exist; run scripts/build_modelscope_pkg.py first")
    local = sorted(p.relative_to(PKG).as_posix() for p in PKG.rglob("*") if p.is_file())
    size = sum((PKG / f).stat().st_size for f in local) / 1e6
    print(f"{PKG}: {len(local)} files, {size:.1f} MB")
    for f in ("modeling/heads.json", "modeling/service.py", "README.md", "examples/selfcheck.py"):
        if f not in local:
            raise SystemExit(f"{f} is missing from the package -- refusing to upload")
    selfcheck()
    if args.dry_run:
        print("\n--dry-run: would create/read the repo and upload:")
        for f in local:
            print("   ", f)
        return 0

    secret = token()
    from modelscope.hub.api import HubApi

    api = HubApi()
    who, _ = api.login(secret)
    print(f"--- logged in as {who}")

    # The repo is created only if reading it back fails, and the result is verified by reading it
    # again rather than trusting a return value -- `create_model` takes version-dependent kwargs and
    # an id that already exists is the normal case, not an error worth failing on.
    try:
        api.get_model(args.repo)
        print(f"--- {args.repo} already exists")
    except Exception as exc:
        print(f"--- {args.repo} not readable ({type(exc).__name__}); creating")
        try:
            api.create_model(args.repo)
        except Exception as exc2:
            print(f"  create_model said: {type(exc2).__name__}: {exc2}")
            raise

    print("--- uploading")
    from modelscope import push_to_hub

    push_to_hub(repo_name=args.repo, output_dir=str(PKG), token=secret, private=False,
                commit_message=f"rwkv7 decision heads: {len(local)} files, {size:.1f} MB")

    print("--- reading the remote file list back")
    remote = sorted((f["Path"] if isinstance(f, dict) else getattr(f, "Path", str(f)))
                    for f in api.get_model_files(args.repo, recursive=True))
    missing = [f for f in local if f not in remote]
    print(f"remote has {len(remote)} files; local files not visible remotely: {missing or 'none'}")
    if missing:
        print("  upload is NOT confirmed -- the list above is what is missing")
        return 1
    print(f"\nhttps://www.modelscope.cn/models/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
