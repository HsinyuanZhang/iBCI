"""P-FIX / P-CA wrappers. Import sealed estimators; do not patch the old package."""

from .factory import FORBIDDEN_OLD_FACTORY, FreshPArm, build_fresh_arm

__all__ = ["FORBIDDEN_OLD_FACTORY", "FreshPArm", "build_fresh_arm"]
