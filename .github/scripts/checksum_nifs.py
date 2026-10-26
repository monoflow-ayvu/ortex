#!/usr/bin/env python3
"""Write checksum-Elixir.Ortex.Native.exs for the NIF artifacts in a directory.

Shared by the release workflow and its test gate so the file the tests validate is
byte-for-byte the one the release publishes.

Usage: checksum_nifs.py <artifacts-dir> <version>
"""

import glob
import hashlib
import os
import sys


def main() -> None:
    artifacts_dir, version = sys.argv[1], sys.argv[2]
    paths = sorted(glob.glob(os.path.join(artifacts_dir, f"*-v{version}-*.so.tar.gz")))
    if not paths:
        raise SystemExit(f"no artifacts matching {artifacts_dir}/*-v{version}-*.so.tar.gz")

    with open("checksum-Elixir.Ortex.Native.exs", "w") as checksum_file:
        checksum_file.write("%{\n")
        for path in paths:
            with open(path, "rb") as artifact:
                digest = hashlib.sha256(artifact.read()).hexdigest()
            checksum_file.write(f'  "{os.path.basename(path)}" => "sha256:{digest}",\n')
        checksum_file.write("}\n")


if __name__ == "__main__":
    main()
