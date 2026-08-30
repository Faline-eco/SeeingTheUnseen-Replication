"""Entry point for running the pipeline from an IDE (e.g. PyCharm).

Running `src/georef/cli.py` directly fails with "attempted relative import with
no known parent package", because that executes the file as a loose script
rather than as part of the `georef` package. This launcher uses an absolute
import instead, so PyCharm's Run button works on it directly.

HOW TO USE: edit the ARGS list below, then just press Run. No run-configuration
parameters are needed. (Any parameters you *do* pass on the command line still
take precedence over ARGS.)
"""
import sys

from georef.cli import main

# ─────────────────────────── EDIT YOUR ARGS HERE ────────────────────────────
# Examples:
#   ["--flight-ids", "104", "--dry-run"]      # preview one flight, render nothing
#   ["--flight-ids", "1,3,104"]               # process specific flights
#   ["--flight-ids", "all"]                   # process every flight (~hours)
#   ["--flight-ids", "104", "--modalities", "thermal", "--overwrite"]
ARGS = ["--flight-ids", "all"]
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Command-line arguments (if any) override the ARGS defined above.
    argv = sys.argv[1:] if len(sys.argv) > 1 else ARGS
    sys.exit(main(argv))
