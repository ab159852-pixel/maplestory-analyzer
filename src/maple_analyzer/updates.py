"""Background update discovery and safe hand-off to a Windows updater.

The application is distributed as a PyInstaller one-folder build.  Windows
keeps the running executable locked, so this module never replaces files in
the current process.  It downloads and verifies a GitHub Release zip, then
starts a short-lived PowerShell helper which waits for this process to exit,
swaps the application folder, and starts the new executable.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
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


def _archive_release_version(archive: zipfile.ZipFile) -> str | None:
    """Read the release marker placed beside the packaged executable."""
    candidates = [
        member
        for member in archive.infolist()
        if not member.is_dir()
        and Path(member.filename).name.casefold() == "release-version.txt"
    ]
    if len(candidates) != 1:
        return None
    try:
        value = archive.read(candidates[0]).decode("ascii").strip()
    except (UnicodeDecodeError, OSError, RuntimeError):
        return None
    return value if _VERSION_RE.fullmatch(value) else None


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
            package_version = _archive_release_version(archive)
            if package_version != info.version:
                raise UpdateError(
                    "update package version does not match the selected release"
                )
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


_POWERSHELL_UPDATER = r'''param(
    [Parameter(Mandatory = $true)][string] $ZipPath,
    [Parameter(Mandatory = $true)][string] $InstallDir,
    [Parameter(Mandatory = $true)][string] $ExeName,
    [Parameter(Mandatory = $true)][string] $PackageExeName,
    [Parameter(Mandatory = $true)][string] $ExpectedVersion,
    [Parameter(Mandatory = $true)][int] $ProcessId,
    [Parameter(Mandatory = $true)][string] $ScriptPath,
    [Parameter(Mandatory = $true)][string] $StatusPath
)
$ErrorActionPreference = "Stop"
$success = $false
$stage = Join-Path ([System.IO.Path]::GetTempPath()) ("MapleStoryAnalyzer-update-" + [guid]::NewGuid().ToString("N"))
$backup = "$InstallDir.previous-" + [guid]::NewGuid().ToString("N")
$newProcess = $null
function Write-UpdateStatus([string] $Message) {
    try {
        (Get-Date -Format o) + " " + $Message | Add-Content -LiteralPath $StatusPath -Encoding UTF8
    } catch {
        # A status file must never prevent the actual update transaction.
    }
}
try {
    # The app is normally launched with its own install directory as the
    # process working directory. Windows refuses to move a directory that a
    # PowerShell process currently has as its working directory, so release
    # that handle before swapping the one-folder installation.
    $helperWorkingDir = [System.IO.Path]::GetTempPath()
    [System.IO.Directory]::SetCurrentDirectory($helperWorkingDir)
    Set-Location -LiteralPath $helperWorkingDir
    Write-UpdateStatus ("helper-start install=" + $InstallDir + " exe=" + $ExeName + " expected=" + $ExpectedVersion)
    if (-not (Test-Path -LiteralPath $ZipPath)) {
        throw "update archive does not exist: $ZipPath"
    }
    New-Item -ItemType Directory -Path $stage -Force | Out-Null
    Expand-Archive -LiteralPath $ZipPath -DestinationPath $stage -Force
    $package = Join-Path $stage "MapleStoryAnalyzer"
    if (-not (Test-Path -LiteralPath $package)) {
        $package = $stage
    }
    $packageExe = Join-Path $package $PackageExeName
    if (-not (Test-Path -LiteralPath $packageExe)) {
        $packageExe = Get-ChildItem -LiteralPath $package -Filter "*.exe" -File |
            Select-Object -First 1 -ExpandProperty FullName
    }
    if (-not $packageExe) {
        throw "updated package does not contain a Windows executable"
    }
    $packageVersionFile = Join-Path $package "release-version.txt"
    if (-not (Test-Path -LiteralPath $packageVersionFile)) {
        throw "updated package is missing release-version.txt"
    }
    $packageVersion = (Get-Content -LiteralPath $packageVersionFile -Raw).Trim()
    if (-not [string]::Equals($packageVersion, $ExpectedVersion, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "updated package version $packageVersion does not match expected $ExpectedVersion"
    }
    Write-UpdateStatus ("package-version-verified version=" + $packageVersion)
    $targetExe = Join-Path $package $ExeName
    if (-not [string]::Equals($packageExe, $targetExe, [System.StringComparison]::OrdinalIgnoreCase)) {
        Move-Item -LiteralPath $packageExe -Destination $targetExe
    }
    if (-not (Test-Path -LiteralPath $targetExe)) {
        throw "updated package does not contain $ExeName"
    }
    Write-UpdateStatus ("package-ready path=" + $targetExe)

    for ($attempt = 0; $attempt -lt 60; $attempt++) {
        if ($null -eq (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)) { break }
        Start-Sleep -Milliseconds 500
    }
    if ($null -ne (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)) {
        throw "the old application did not exit"
    }
    Write-UpdateStatus "old-process-exited"

    Move-Item -LiteralPath $InstallDir -Destination $backup
    Write-UpdateStatus ("old-install-backed-up path=" + $backup)
    try {
        Move-Item -LiteralPath $package -Destination $InstallDir
    } catch {
        Move-Item -LiteralPath $backup -Destination $InstallDir
        throw
    }
    $targetExe = Join-Path $InstallDir $ExeName
    if (-not (Test-Path -LiteralPath $targetExe)) {
        throw "installed package does not contain $targetExe"
    }
    $installedVersionFile = Join-Path $InstallDir "release-version.txt"
    if (-not (Test-Path -LiteralPath $installedVersionFile)) {
        throw "installed package is missing release-version.txt"
    }
    $installedVersion = (Get-Content -LiteralPath $installedVersionFile -Raw).Trim()
    if (-not [string]::Equals($installedVersion, $ExpectedVersion, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "installed package version $installedVersion does not match expected $ExpectedVersion"
    }
    Write-UpdateStatus ("new-install-ready path=" + $targetExe + " version=" + $installedVersion)
    $newProcess = Start-Process -FilePath $targetExe -WorkingDirectory $InstallDir -PassThru
    # Starting a process is not proof that the one-folder package is usable.
    # Keep the old folder until the new process survives its import/theme/Tk
    # startup window; otherwise a missing asset would look like a successful
    # update and leave the user with no working application.
    for ($attempt = 0; $attempt -lt 40; $attempt++) {
        if ($newProcess.HasExited) {
            throw "updated application exited during startup (code $($newProcess.ExitCode))"
        }
        Start-Sleep -Milliseconds 250
    }
    Write-UpdateStatus ("new-process-started pid=" + $newProcess.Id + " version=" + $ExpectedVersion)
    $success = $true
    Remove-Item -LiteralPath $backup -Recurse -Force -ErrorAction SilentlyContinue
    Write-UpdateStatus "update-success"
} catch {
    $message = $_.Exception.Message
    Write-UpdateStatus ("update-error " + $message)
    $log = Join-Path ([System.IO.Path]::GetTempPath()) "MapleStoryAnalyzer-update-error.txt"
    (Get-Date -Format o) + " " + $message + " [InstallDir=$InstallDir]" |
        Set-Content -LiteralPath $log -Encoding UTF8
    if ($null -ne $newProcess -and -not $newProcess.HasExited) {
        Stop-Process -Id $newProcess.Id -Force -ErrorAction SilentlyContinue
    }
    if ((Test-Path $backup) -and (Test-Path $InstallDir)) {
        Remove-Item -LiteralPath $InstallDir -Recurse -Force -ErrorAction SilentlyContinue
    }
    if ((Test-Path $backup) -and -not (Test-Path $InstallDir)) {
        Move-Item -LiteralPath $backup -Destination $InstallDir -ErrorAction SilentlyContinue
    }
} finally {
    Remove-Item -LiteralPath $stage -Recurse -Force -ErrorAction SilentlyContinue
    if ($success) {
        Remove-Item -LiteralPath $ZipPath -Force -ErrorAction SilentlyContinue
    }
    Remove-Item -LiteralPath $ScriptPath -Force -ErrorAction SilentlyContinue
    if ($success) {
        Write-UpdateStatus "helper-finished"
    }
}
'''


def schedule_update(zip_path: Path, *, expected_version: str) -> None:
    """Start the detached updater and return; caller should then exit."""
    if sys.platform != "win32" or not getattr(sys, "frozen", False):
        raise UpdateError("automatic installation is available only in the packaged Windows app")
    executable = Path(sys.executable).resolve()
    install_dir = executable.parent
    script_fd, script_name = tempfile.mkstemp(prefix="maplestory-updater-", suffix=".ps1")
    os.close(script_fd)
    script_path = Path(script_name)
    script_path.write_text(_POWERSHELL_UPDATER, encoding="utf-8")
    status_path = Path(tempfile.gettempdir()) / f"{APP_NAME}-update-status.txt"
    status_path.unlink(missing_ok=True)
    powershell = Path(os.environ.get("WINDIR", r"C:\Windows")) / (
        "System32" / Path("WindowsPowerShell") / Path("v1.0") / Path("powershell.exe")
    )
    powershell_command = str(powershell) if powershell.is_file() else "powershell.exe"
    command = [
        powershell_command, "-NoLogo", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
        "-File", str(script_path),
        "-ZipPath", str(zip_path),
        "-InstallDir", str(install_dir),
        "-ExeName", executable.name,
        "-PackageExeName", f"{APP_NAME}.exe",
        "-ExpectedVersion", expected_version,
        "-ProcessId", str(os.getpid()),
        "-ScriptPath", str(script_path),
        "-StatusPath", str(status_path),
    ]
    # Do not combine DETACHED_PROCESS with PowerShell.  On this Windows
    # build that combination can return a child PID and exit immediately
    # without executing ``-File`` at all; the app then closes while the ZIP
    # and updater script remain untouched.  CREATE_NO_WINDOW is sufficient
    # to keep the helper invisible and, unlike DETACHED_PROCESS, actually
    # runs the script and survives the parent application's exit.  A new
    # process group also makes the helper independent of the app's console
    # lifetime without detaching its standard process creation semantics.
    creation_flags = (
        getattr(subprocess, "CREATE_NO_WINDOW", 0)
        | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    )
    try:
        # Do not inherit the app directory as the helper's current directory:
        # Windows then treats that directory as in use and Move-Item cannot
        # rename it after the old process exits. A temp working directory also
        # keeps the detached helper independent of the soon-to-close app.
        status_path.write_text(
            "launcher-start\n",
            encoding="utf-8",
        )
        helper = subprocess.Popen(
            command,
            cwd=tempfile.gettempdir(),
            creationflags=creation_flags,
            close_fds=True,
        )
        # Popen returning only proves that CreateProcess accepted the
        # command.  Wait briefly for the PowerShell helper to acknowledge
        # that it actually entered the script.  This prevents the app from
        # closing after a silent CreateProcess/PowerShell failure.  The
        # helper keeps writing the detailed transaction log after this
        # acknowledgement while the current process exits.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                status = status_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                status = ""
            if "helper-start" in status:
                return
            exit_code = helper.poll()
            if exit_code is not None:
                raise UpdateError(
                    f"update helper exited before installation started (code {exit_code})"
                )
            time.sleep(0.05)
        with contextlib.suppress(Exception):
            helper.terminate()
        raise UpdateError("update helper did not acknowledge startup")
    except Exception as exc:
        script_path.unlink(missing_ok=True)
        if isinstance(exc, UpdateError):
            raise
        raise UpdateError("could not start the update helper") from exc
