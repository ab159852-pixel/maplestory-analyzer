"""Application version and public release channel."""
from __future__ import annotations

# Keep the package/executable identifier stable so existing auto-update
# installs and LOCALAPPDATA settings continue to work.  The user-facing
# product name is intentionally shorter and more polished.
APP_NAME = "MapleStoryAnalyzer"
APP_DISPLAY_NAME = "Maple Insight"
APP_VERSION = "1.0.16"
GITHUB_REPOSITORY = "ab159852-pixel/maplestory-analyzer"
RELEASES_API_URL = f"https://api.github.com/repos/{GITHUB_REPOSITORY}/releases/latest"
