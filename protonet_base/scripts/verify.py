#! /usr/bin/env python

"""Script that runs all verification steps.
"""

import argparse

import os
import shutil
from subprocess import run
from subprocess import CalledProcessError
import sys

def main(checks):
    try:
        print("Verifying with " + str(checks))
        if "pytest" in checks:
            print("Tests (pytest):")
            run("pytest -v --color=yes", shell=True, check=True)

        if "pylint" in checks:
            print("Linter (pylint):")
            run("pylint -d locally-disabled,locally-enabled -f colorized allennlp tests", shell=True, check=True)
            print("pylint checks passed")

        if "mypy" in checks:
            print("Typechecker (mypy):")
            run("mypy allennlp --ignore-missing-imports", shell=True, check=True)
            print("mypy checks passed")

        if "build-docs" in checks:
            print("Documentation (build):")
            run("cd doc; make html-strict", shell=True, check=True)

        if "check-docs" in checks:
            print("Documentation (check):")
            run("./scripts/check_docs.py", shell=True, check=True)
            print("check docs passed")

    except CalledProcessError:
        # squelch the exception stacktrace
        sys.exit(1)

if __name__ == "__main__":
    checks = ['pytest', 'pylint', 'mypy', 'build-docs', 'check-docs']

    parser = argparse.ArgumentParser()
    parser.add_argument('--checks', type=str, required=False, nargs='+', choices=checks)

    args = parser.parse_args()

    if args.checks:
        run_checks = args.checks
    else:
        run_checks = checks

    main(run_checks)
