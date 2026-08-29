from __future__ import annotations

from io import BytesIO
import inspect
import os
from pathlib import Path
import subprocess
import zipfile

from maple_analyzer.updates import (
    _POWERSHELL_UPDATER,
    _archive_release_version,
    _safe_archive_members,
    parse_latest_release,
)


def _release_payload(*, tag: str = "v1.0.5") -> dict:
    return {
        "tag_name": tag,
        "html_url": f"https://github.com/ab159852-pixel/maplestory-analyzer/releases/tag/{tag}",
        "body": "Improve OCR and automatic resolution scaling.",
        "assets": [
            {
                "name": "MapleStoryAnalyzer-v1.0.5-source.zip",
                "browser_download_url": "https://example.invalid/source.zip",
                "digest": "sha256:" + "0" * 64,
            },
            {
                "name": "MapleStoryAnalyzer-v1.0.5-win64.zip",
                "browser_download_url": "https://example.invalid/win64.zip",
                "digest": "sha256:" + "a" * 64,
            },
        ],
    }


def test_latest_release_prefers_windows_asset_and_reads_digest():
    info = parse_latest_release(_release_payload(), current_version="1.0.4")

    assert info is not None
    assert info.version == "1.0.5"
    assert info.asset_name.endswith("win64.zip")
    assert info.download_url.endswith("win64.zip")
    assert info.sha256 == "a" * 64


def test_stale_or_prerelease_is_not_offered():
    assert parse_latest_release(_release_payload(tag="v1.0.4"), current_version="1.0.4") is None
    payload = _release_payload()
    payload["prerelease"] = True
    assert parse_latest_release(payload, current_version="1.0.4") is None


def test_newer_release_is_offered_for_updater_test():
    info = parse_latest_release(_release_payload(tag="v1.0.11"), current_version="1.0.10")

    assert info is not None
    assert info.version == "1.0.11"


def test_archive_members_reject_zip_slip_and_require_app_executable():
    safe_buffer = BytesIO()
    with zipfile.ZipFile(safe_buffer, "w") as archive:
        archive.writestr("MapleStoryAnalyzer/MapleStoryAnalyzer.exe", b"exe")
    with zipfile.ZipFile(BytesIO(safe_buffer.getvalue())) as archive:
        assert _safe_archive_members(archive)

    unsafe_buffer = BytesIO()
    with zipfile.ZipFile(unsafe_buffer, "w") as archive:
        archive.writestr("../MapleStoryAnalyzer.exe", b"exe")
    with zipfile.ZipFile(BytesIO(unsafe_buffer.getvalue())) as archive:
        assert not _safe_archive_members(archive)


def test_release_marker_must_be_read_from_a_single_archive_member():
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("MapleStoryAnalyzer/MapleStoryAnalyzer.exe", b"exe")
        archive.writestr("MapleStoryAnalyzer/release-version.txt", "1.0.26")
    with zipfile.ZipFile(BytesIO(buffer.getvalue())) as archive:
        assert _archive_release_version(archive) == "1.0.26"


def test_updater_handles_renamed_exe_and_records_transaction_progress():
    assert "$PackageExeName" in _POWERSHELL_UPDATER
    assert "$ExpectedVersion" in _POWERSHELL_UPDATER
    assert "package-version-verified" in _POWERSHELL_UPDATER
    assert "new-install-ready" in _POWERSHELL_UPDATER
    assert "SetCurrentDirectory($helperWorkingDir)" in _POWERSHELL_UPDATER
    assert "Start-Process -FilePath $targetExe -WorkingDirectory $InstallDir -PassThru" in _POWERSHELL_UPDATER
    assert "$StatusPath" in _POWERSHELL_UPDATER
    assert "update-success" in _POWERSHELL_UPDATER
    assert "Move-WithRetry" in _POWERSHELL_UPDATER
    assert "Copy-WithRobocopy" in _POWERSHELL_UPDATER
    assert "in-place-copy-complete" in _POWERSHELL_UPDATER
    assert "old-process-restarted" in _POWERSHELL_UPDATER
    assert "System.Windows.Forms.MessageBox" in _POWERSHELL_UPDATER
    assert "Get-InstallProcesses" in _POWERSHELL_UPDATER
    assert "[System.IO.Path]::GetFullPath($candidatePath)" in _POWERSHELL_UPDATER
    assert "force-stopping-install-process" in _POWERSHELL_UPDATER
    assert "parent-process-captured" in _POWERSHELL_UPDATER
    assert "$parentProcess.WaitForExit(6000)" in _POWERSHELL_UPDATER
    assert "$parentProcess.Kill()" in _POWERSHELL_UPDATER
    assert "force-stopping-parent-process" in _POWERSHELL_UPDATER
    assert "old-install-processes-cleared" in _POWERSHELL_UPDATER


def test_updater_does_not_use_detached_process_for_powershell():
    # DETACHED_PROCESS can accept CreateProcess but silently skip the
    # PowerShell -File script on Windows. CREATE_NO_WINDOW keeps it hidden
    # while allowing the helper to run after the app exits.
    from maple_analyzer import updates

    source = inspect.getsource(updates.schedule_update)
    assert 'getattr(subprocess, "CREATE_NO_WINDOW", 0)' in source
    assert 'getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)' in source
    assert ' | getattr(subprocess, "DETACHED_PROCESS", 0)' not in source
    assert '"launcher-start\\n"' in source


def test_updater_requires_helper_start_ack_before_app_exit():
    from maple_analyzer import updates

    source = inspect.getsource(updates.schedule_update)
    assert '"helper-start" in status' in source
    assert 'update helper did not acknowledge startup' in source
    assert 'helper.terminate()' in source


def test_embedded_powershell_updater_has_valid_syntax(tmp_path: Path):
    script = tmp_path / "updater.ps1"
    script.write_text(_POWERSHELL_UPDATER, encoding="utf-8")
    parser = (
        "$tokens=$null; $errors=$null; "
        "[System.Management.Automation.Language.Parser]::ParseFile("
        "$env:MAPLE_UPDATER_PARSE_PATH, [ref]$tokens, [ref]$errors) | Out-Null; "
        "if ($errors.Count -gt 0) { "
        "$errors | ForEach-Object { Write-Error $_.Message }; exit 1 }"
    )
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            parser,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "MAPLE_UPDATER_PARSE_PATH": str(script)},
        timeout=15,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
