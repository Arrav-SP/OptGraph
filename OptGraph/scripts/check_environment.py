#!/usr/bin/env python3
"""Report the toolchain, verify C -> LLVM IR -> opt works, write environment.json.

Exit code 0 when the environment is usable, 1 otherwise.
"""
import argparse
import sys

import _bootstrap  # noqa: F401
from src import llvm_pipeline as lp
from src.utils import ROOT, load_config, load_passes, write_json


def check(write: bool = True, quiet: bool = False) -> tuple[bool, dict]:
    say = (lambda *a: None) if quiet else print
    tc = lp.Toolchain()
    env = lp.collect_environment(tc)
    cfg = load_config()
    env["seed"] = cfg["settings"].get("seed")
    problems = []

    say("Operating system : ", env["os"])
    say("WSL              : ", f"yes ({env['wsl_distro']})" if env["is_wsl"] else "no / not detected")
    say("Distribution     : ", env["distribution"])
    say("Ubuntu version   : ", env["ubuntu_version"])
    say("Python           : ", env["python_version"])
    for label, key, tool in (("Clang", "clang_version", "clang"), ("LLVM", "llvm_version", "llvm-config"),
                             ("opt", "opt_version", "opt"), ("llvm-config", "llvm_config_version", "llvm-config"),
                             ("llc", "llc_version", "llc")):
        say(f"{label:<17}: ", env[key] or "NOT FOUND")
        if not env[key] and tool in ("clang", "opt"):
            problems.append(f"{tool} is missing. Install with: {lp.INSTALL_HINT}")
        elif not env[key] and label != "LLVM":
            say(f"    note: {tool} is optional for OptGraph but was not found")

    if env["system"] != "Linux":
        say("    note: OptGraph is developed for Linux/WSL; run it inside WSL (`wsl`) on Windows")

    smoke = {"compiled": False, "pass_ran": False, "ir_changed": False, "error": "tools missing"}
    bad_passes = {}
    if not tc.missing():
        smoke = lp.smoke_test(tc)
        say("C -> LLVM IR     : ", "ok" if smoke["compiled"] else "FAILED")
        say("opt pass run     : ", "ok" if smoke["pass_ran"] else "FAILED")
        say("IR changed       : ", "yes" if smoke["ir_changed"] else "NO")
        if smoke["error"]:
            problems.append(smoke["error"])
        passes = load_passes(cfg)
        bad_passes = lp.validate_passes(tc, passes)
        say("Configured passes: ", f"{len(passes) - len(bad_passes)}/{len(passes)} accepted by opt")
        for name, err in bad_passes.items():
            problems.append(f"pass '{name}' rejected by this opt: {err}")
    env["smoke_test"] = smoke
    env["rejected_passes"] = bad_passes
    env["pass_configuration"] = cfg["passes"]
    env["settings"] = cfg["settings"]

    ok = not problems
    env["environment_ok"] = ok
    if write:
        write_json(ROOT / "environment.json", env)
        say(f"\nWrote {ROOT / 'environment.json'}")
    if problems:
        print("\nENVIRONMENT PROBLEMS:", file=sys.stderr)
        for p in problems:
            print("  - " + p, file=sys.stderr)
    else:
        say("Environment OK.")
    return ok, env


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-write", action="store_true", help="do not write environment.json")
    args = ap.parse_args()
    sys.exit(0 if check(write=not args.no_write)[0] else 1)
