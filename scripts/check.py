#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
import re
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def check_cubin(path, sha, errors):
    if not path.is_file():
        errors.append(f"missing cubin: {path}")
    elif hashlib.sha256(path.read_bytes()).hexdigest() != sha:
        errors.append(f"SHA-256 mismatch: {path}")


def main():
    kernels = tomllib.loads((ROOT / "kernels.toml").read_text())["kernels"]
    errors = []
    for path in (ROOT / "manifests").glob("*.json"):
        manifest = json.loads(path.read_text())
        modules = manifest["modules"]
        for name, module in modules.items():
            kernel = kernels[name]
            sha = module["sha256"]
            assert re.fullmatch(r"[0-9a-f]{64}", sha), name
            assert sha == kernel["sha256"], name
            assert module["source"] in (f"hf:Pegainfer/kern-kernels/blobs/{sha}", f"{name}.cubin"), name
            for field in ("source", "build"):
                if field in kernel:
                    assert (ROOT / kernel[field]).is_file(), (name, field)
            assert "source" in kernel or "upstream" in kernel, name
            if "prebuilt" in kernel:
                check_cubin(ROOT / kernel["prebuilt"], sha, errors)
            if len(sys.argv) > 1:
                cubin = Path(sys.argv[1]) / f"{name}.cubin"
                check_cubin(cubin, sha, errors)
        for op in manifest["ops"].values():
            for launch in op["impl"].get("launches", []):
                if "module" in launch:
                    assert launch["module"] in modules, launch["module"]
        for program in manifest["programs"].values():
            for call in program["calls"]:
                assert call["op"] in manifest["ops"], call["op"]
        if not errors:
            print(f"{path.name}: {len(modules)} modules, source links, references and cubin checks OK")
    for table in sorted((ROOT / "decode").glob("*/kernels.toml")):
        check_capture(table, errors)
    if errors:
        raise SystemExit("\n".join(errors))


def check_capture(table, errors):
    """A captured set: every prebuilt cubin matches its hash, every module is either here or says why not."""
    here = table.parent
    kernels = tomllib.loads(table.read_text())["kernels"]
    pinned = set()
    for name, kernel in kernels.items():
        assert re.fullmatch(r"[0-9a-f]{64}", kernel["sha256"]), name
        assert ("prebuilt" in kernel) != ("not_included" in kernel), name
        assert kernel["symbols"], name
        if "prebuilt" in kernel:
            check_cubin(here / kernel["prebuilt"], kernel["sha256"], errors)
            pinned.add(kernel["prebuilt"])
    stray = {f"prebuilt/{p.name}" for p in (here / "prebuilt").glob("*.cubin")} - pinned
    errors += [f"{here.name}: {p} is in no kernels.toml entry" for p in sorted(stray)]
    print(f"{here.name}: {len(kernels)} modules, {len(pinned)} prebuilt, hashes OK")


if __name__ == "__main__":
    main()
