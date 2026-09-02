"""Compatibility entry point for the Stage Y inventory/capability audit.

The full Stage Y script runs this audit before any permitted policy inference.
"""

from stageY_N32_N64_policy_extrapolation import main


if __name__ == "__main__":
    main()
