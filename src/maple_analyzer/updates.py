"""Background update discovery and safe hand-off to a Windows updater.

The application is distributed as a PyInstaller one-folder build.  Windows
keeps the running executable locked, so this module never replaces files in
the current process.  It downloads and verifies a GitHub Release zip, then
starts a short-lived PowerShell helper which waits for this process to exit,
swaps the application folder, and starts the new executable.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Mapping
from urllib.request import Request, urlopen
import zipfile

from .version import APP_NAME, APP_VERSION, RELEASES_API_URL


_USER_AGENT = f"{APP_NAME}/{APP_VERSION} update-check"
_ZIP_NAME_RE = re.compile(r"\.zip$", re.IGNORECASE)
_VERSION_RE = re.compile(r"^v?(\d+)(?:\.(\d+))?(?:\.(\d+))?(?:[-+].*)?$", re.IGNORECASE)


class UpdateError(RuntimeError):
    """Raised when an update cannot be downloaded or safely staged."""


@dataclass(frozen=True)
class UpdateInfo:
    version: str
    tag_name: str
    release_url: str
    download_url: str
    asset_name: str
    sha256: str | None
    notes: str


def _version_key(value: object) -> tuple[int, int, int]:
    match = _VERSION_RE.match(str(value or "").strip())
    if not match:
        return (0, 0, 0)
    return tuple(int(part or 0) for part in match.groups())


def _asset_sha256(asset: Mapping[str, Any]) -> str | None:
    digest = str(asset.get("digest") or "").strip().lower()
    if digest.startswith("sha256:"):
        digest = digest.removeprefix("sha256:")
    return digest if re.fullmatch(r"[0-9a-f]{64}", digest) else None


def parse_latest_release(
    payload: Mapping[str, Any], *, current_version: str = APP_VERSION
) -> UpdateInfo | None:
    """Parse GitHub's latest-release response and ignore stale releases."""
    if payload.get("draft") or payload.get("prerelease"):
        return None
    tag_name = str(payload.get("tag_name") or payload.get("name") or "").strip()
    version_match = _VERSION_RE.match(tag_name)
    if version_match is None or _version_key(tag_name) <= _version_key(current_version):
        return None

    assets = payload.get("assets")
    if not isinstance(assets, list):
        raise UpdateError("latest release has no downloadable assets")
    candidates = [
        asset for asset in assets
        if isinstance(asset, Mapping)
        and _ZIP_NAME_RE.search(str(asset.get("name") or ""))
        and str(asset.get("browser_download_url") or "").startswith("https://")
    ]
    if not candidates:
        raise UpdateError("latest release has no Windows zip asset")
    asset = next(
        (
            item for item in candidates
            if "win64" in str(item.get("name") or "").casefold()
            or "windows" in str(item.get("name") or "").casefold()
        ),
        candidates[0],
    )
    notes = str(payload.get("body") or "").strip()
    return UpdateInfo(
        version=tag_name.removeprefix("v"),
        tag_name=tag_name,
        release_url=str(payload.get("html_url") or ""),
        download_url=str(asset["browser_download_url"]),
        asset_name=str(asset.get("name") or "update.zip"),
        sha256=_asset_sha256(asset),
        notes=notes,
    )


def check_for_update(*, timeout: float = 8.0) -> UpdateInfo | None:
    """Fetch the public latest release without changing local state."""
    request = Request(
        RELEASES_API_URL,
        headers={"User-Agent": _USER_AGENT, "Accept": "application/vnd.github+json"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise UpdateError("could not check GitHub Releases") from exc
    if not isinstance(payload, Mapping):
        raise UpdateError("GitHub returned an invalid release response")
    return parse_latest_release(payload)


def _safe_archive_members(archive: zipfile.ZipFile) -> bool:
    """Reject zip-slip paths before the PowerShell extractor sees the file."""
    names = [member.filename for member in archive.infolist() if not member.is_dir()]
    if not names:
        return False
    for name in names:
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            return False
    return any(Path(name).name.casefold() == f"{APP_NAME.lower()}.exe" for name in names)


def download_update(info: UpdateInfo, *, timeout: float = 60.0) -> Path:
    """Download and verify one release zip into the user's temporary folder."""
    safe_version = re.sub(r"[^0-9A-Za-z._-]+", "_", info.version)
    temporary_path = Path(tempfile.gettempdir()) / f"{APP_NAME}-update-{safe_version}.zip"
    request = Request(
        info.download_url,
        headers={"User-Agent": _USER_AGENT, "Accept": "application/octet-stream"},
    )
    try:
        with urlopen(request, timeout=timeout) as response, temporary_path.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
    except Exception as exc:
        temporary_path.unlink(missing_ok=True)
        raise UpdateError("update download failed") from exc

    try:
        digest = hashlib.sha256(temporary_path.read_bytes()).hexdigest()
        if info.sha256 and digest.casefold() != info.sha256.casefold():
            raise UpdateError("update checksum does not match the release digest")
        with zipfile.ZipFile(temporary_path) as archive:
            if not _safe_archive_members(archive) or archive.testzip() is not None:
                raise UpdateError("update archive is invalid")
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


_POWERSHELL_UPDATER = r'''param(
    [Parameter(Mandatory = $true)][string] $ZipPath,
    [Parameter(Mandatory = $true)][string] $InstallDir,
    [Parameter(Mandatory = $true)][string] $ExeName,
    [Parameter(Mandatory = $true)][int] $ProcessId,
    [Parameter(Mandatory = $true)][string] $ScriptPath
)
$ErrorActionPreference = "Stop"
$stage = Join-Path ([System.IO.Path]::GetTempPath()) ("MapleStoryAnalyzer-update-" + [guid]::NewGuid().ToString("N"))
$backup = "$InstallDir.previous"
try {
    New-Item -ItemType Directory -Path $stage -Force | Out-Null
    Expand-Archive -LiteralPath $ZipPath -DestinationPath $stage -Force
    $package = Join-Path $stage "MapleStoryAnalyzer"
    if (-not (Test-Path (Join-Path $package $ExeName))) {
        $package = $stage
    }
    if (-not (Test-Path (Join-Path $package $ExeName))) {
        throw "updated package does not contain $ExeName"
    }

    for ($attempt = 0; $attempt -lt 60; $attempt++) {
        if ($null -eq (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)) { break }
        Start-Sleep -Milliseconds 500
    }
    if ($null -ne (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)) {
        throw "the old application did not exit"
    }

    if (Test-Path $backup) { Remove-Item -LiteralPath $backup -Recurse -Force }
    Move-Item -LiteralPath $InstallDir -Destination $backup
    try {
        Move-Item -LiteralPath $package -Destination $InstallDir
    } catch {
        Move-Item -LiteralPath $backup -Destination $InstallDir
        throw
    }
    Start-Process -FilePath (Join-Path $InstallDir $ExeName)
    Remove-Item -LiteralPath $backup -Recurse -Force -ErrorAction SilentlyContinue
} catch {
    $log = Join-Path ([System.IO.Path]::GetTempPath()) "MapleStoryAnalyzer-update-error.txt"
    (Get-Date -Format o) + " " + $_.Exception.Message | Set-Content -LiteralPath $log -Encoding UTF8
    if ((Test-Path $backup) -and -not (Test-Path $InstallDir)) {
        Move-Item -LiteralPath $backup -Destination $InstallDir -ErrorAction SilentlyContinue
    }
} finally {
    Remove-Item -LiteralPath $stage -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $ZipPath -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $ScriptPath -Force -ErrorAction SilentlyContinue
}
'''


def schedule_update(zip_path: Path) -> None:
    """Start the detached updater and return; caller should then exit."""
    if sys.platform != "win32" or not getattr(sys, "frozen", False):
        raise UpdateError("automatic installation is available only in the packaged Windows app")
    executable = Path(sys.executable).resolve()
    install_dir = executable.parent
    script_fd, script_name = tempfile.mkstemp(prefix="maplestory-updater-", suffix=".ps1")
    os.close(script_fd)
    script_path = Path(script_name)
    script_path.write_text(_POWERSHELL_UPDATER, encoding="utf-8")
    command = [
        "powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
        "-File", str(script_path),
        "-ZipPath", str(zip_path),
        "-InstallDir", str(install_dir),
        "-ExeName", executable.name,
        "-ProcessId", str(os.getpid()),
        "-ScriptPath", str(script_path),
    ]
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
    try:
        subprocess.Popen(command, creationflags=creation_flags, close_fds=True)
    except Exception as exc:
        script_path.unlink(missing_ok=True)
        raise UpdateError("could not start the update helper") from exc
