"""UI string translation. English + Traditional Chinese, default Traditional
Chinese per user request. One mapping table (`_STRINGS`) is the single source
of truth for every user-facing string in overlay.py -- no literal UI text
should live inline in overlay.py itself, so both languages stay complete and
adding a string can't accidentally skip a translation.

Live game-data labels (HP/MP/EXP/LV) are deliberately left as-is in both
languages -- they're the exact abbreviations the game's own panel displays
(see regions.py/parser.py), not app UI chrome, so translating them would make
the overlay's labels mismatch what's on-screen in the actual game.
"""
from __future__ import annotations

from typing import Literal

Lang = Literal["en", "zh"]

_STRINGS: dict[str, dict[Lang, str]] = {
    "tab_live": {"en": "Live", "zh": "即時"},
    "tab_history": {"en": "History", "zh": "紀錄"},
    "tab_settings": {"en": "Settings", "zh": "設定"},

    "status_tracking": {"en": "Tracking", "zh": "追蹤中"},
    "status_loading": {"en": "Loading OCR…", "zh": "正在載入 OCR…"},
    "status_ocr_failed": {"en": "OCR unavailable", "zh": "OCR 無法使用"},
    "hud_button_enter": {"en": "FLOAT", "zh": "浮動"},
    "hud_button_exit": {"en": "FULL", "zh": "還原"},
    "status_idle": {"en": "Idle", "zh": "閒置"},
    "status_calibrating": {"en": "Calibrating…", "zh": "校準中…"},
    "status_potion_baseline": {"en": "Reading shortcut quantities…", "zh": "正在偵測快捷欄水量…"},
    "status_paused": {"en": "Paused", "zh": "已暫停"},
    "status_stopped": {"en": "Stopped", "zh": "已停止"},
    "timer_left": {"en": "{time} left", "zh": "剩餘 {time}"},

    # Recognized capture.py RuntimeError messages (see overlay.py's
    # _localize_error) -- the game closing or minimizing is a normal,
    # expected condition users hit constantly, so it gets a real translation
    # rather than showing capture.py's raw English exception text.
    "status_error_minimized": {"en": "Game window minimized", "zh": "遊戲視窗已最小化"},
    "status_error_not_found": {"en": "Game window not found", "zh": "找不到遊戲視窗"},
    "status_error_obscured": {
        "en": "Stat panel is covered; live capture is unavailable",
        "zh": "狀態列被其他視窗遮擋，且即時擷取不可用",
    },
    # Fallback for anything NOT recognized above -- an actual bug, not a
    # known game-window state, so {detail} (the raw exception text) stays in
    # English rather than pretending to translate arbitrary Python errors.
    "status_error_unknown": {"en": "Error: {detail}", "zh": "發生錯誤：{detail}"},

    "kv_start_exp": {"en": "Start EXP", "zh": "起始經驗值"},
    "kv_job": {"en": "Job", "zh": "職業"},
    "kv_map": {"en": "Map", "zh": "地圖"},
    "kv_proj_exp_interval": {"en": "EXP / interval", "zh": "每區間預估經驗"},
    "kv_mesos_projected": {"en": "Mesos / interval", "zh": "每區間預估楓幣"},
    "kv_potion_cost_projected": {"en": "Potion cost / interval", "zh": "每區間預估藥水成本"},
    "hud_proj_exp_interval": {"en": "Est. EXP / {minutes} min", "zh": "每{minutes}分鐘預估經驗"},
    "hud_mesos_interval": {"en": "Est. Mesos / {minutes} min", "zh": "每{minutes}分鐘預估楓幣"},
    "hud_potion_cost_interval": {"en": "Est. Cost / {minutes} min", "zh": "每{minutes}分鐘預估消耗"},
    "kv_exp_diff": {"en": "EXP diff", "zh": "經驗值變化"},
    "kv_exp_rate": {"en": "EXP / hour", "zh": "每小時經驗值"},
    "kv_mesos": {"en": "Mesos earned", "zh": "楓幣收入"},
    "kv_shortcut_inventory": {"en": "Shortcut quantities", "zh": "快捷欄水量"},
    "kv_hp_potions": {"en": "HP potions", "zh": "HP 藥水"},
    "kv_mp_potions": {"en": "MP potions", "zh": "MP 藥水"},
    "kv_shared_potions": {"en": "HP/MP potions", "zh": "HP/MP 共用藥水"},
    "kv_hp_recovery": {"en": "HP natural/skill recovery", "zh": "HP 自然／技能回復"},
    "kv_mp_recovery": {"en": "MP natural/skill recovery", "zh": "MP 自然／技能回復"},
    "kv_hp_recovery_savings": {"en": "HP recovery saved", "zh": "HP 回復節省"},
    "kv_mp_recovery_savings": {"en": "MP recovery saved", "zh": "MP 回復節省"},
    "recovery_compact": {
        "en": "natural/skill {natural} · potion {potion}",
        "zh": "自然／技能 {natural} · 藥水 {potion}",
    },
    "recovery_savings_compact": {
        "en": "saved {amount}",
        "zh": "節省約 {amount} 楓幣",
    },
    "kv_eta": {"en": "Level-up ETA", "zh": "升級預估時間"},
    "kv_proj_exp": {"en": "Est. session EXP", "zh": "預估本次經驗值"},
    "kv_hp_loss": {"en": "HP loss", "zh": "HP 損失"},
    "kv_mp_loss": {"en": "MP loss", "zh": "MP 損失"},
    "live_snapshot_header": {"en": "LIVE SNAPSHOT", "zh": "即時狀態"},
    "session_forecast_header": {"en": "SESSION FORECAST", "zh": "本次預估"},
    "economy_header": {"en": "ECONOMY & RECOVERY", "zh": "楓幣與回復"},

    "restart_button": {"en": "Restart Session", "zh": "重新開始"},
    "pause_button": {"en": "Pause", "zh": "暫停"},
    "resume_button": {"en": "Resume", "zh": "繼續"},
    "start_button": {"en": "Start Session", "zh": "開始"},
    "stop_button": {"en": "Stop Test", "zh": "停止測試"},

    "history_empty": {"en": "No sessions yet", "zh": "尚無紀錄"},
    "history_session": {"en": "SESSION #{n}", "zh": "紀錄 #{n}"},
    "history_overview_empty": {"en": "Your session performance will appear here", "zh": "完成一段紀錄後，效率總覽會顯示在這裡"},
    "history_overview": {
        "en": "{count} sessions  •  avg {average} EXP/hr  •  best {best} EXP/hr",
        "zh": "{count} 段紀錄  •  平均 {average} EXP/小時  •  最佳 {best} EXP/小時",
    },
    "history_hp_loss": {"en": "HP LOSS", "zh": "HP 損失"},
    "history_mp_loss": {"en": "MP LOSS", "zh": "MP 損失"},
    "history_economy_line": {
        "en": "Mesos +{mesos:,}  ·  HP potions {hp_uses} ({hp_cost:,})  ·  MP potions {mp_uses} ({mp_cost:,})  ·  shared {shared_uses} ({shared_cost:,})",
        "zh": "楓幣 +{mesos:,}  ·  HP 藥水 {hp_uses} 次（{hp_cost:,}）  ·  MP 藥水 {mp_uses} 次（{mp_cost:,}）  ·  共用 {shared_uses} 次（{shared_cost:,}）",
    },
    "history_recovery_savings": {
        "en": "Natural/skill recovery saved: HP {hp} · MP {mp} mesos",
        "zh": "自然／技能回復預估節省：HP {hp} 楓幣 · MP {mp} 楓幣",
    },
    "history_context_line": {
        "en": "{job}  ·  {map}",
        "zh": "{job}  ·  {map}",
    },
    "context_detecting": {"en": "Detecting…", "zh": "辨識中…"},
    "context_unknown": {"en": "Not detected", "zh": "尚未辨識"},
    "context_refresh": {"en": "Refresh", "zh": "重新辨識"},
    "context_refreshing": {"en": "Detecting…", "zh": "辨識中…"},
    "context_header": {"en": "CONTEXT", "zh": "遊戲資訊"},
    "drop_lookup_header": {"en": "MAP DROP LOOKUP", "zh": "地圖掉落速查"},
    "drop_lookup_button": {"en": "Lookup drops", "zh": "掉落速查"},
    "drop_lookup_loading_short": {"en": "Loading…", "zh": "載入中…"},
    "drop_lookup_hint": {
        "en": "Use the detected map to see spawned monsters and their published drop data.",
        "zh": "依目前自動辨識的地圖，快速查看該地圖生成怪物與公開掉落資料。",
    },
    "drop_lookup_no_map": {
        "en": "No map detected yet. Refresh context or enter a map fallback in Settings.",
        "zh": "尚未辨識到地圖；可重新辨識，或到設定填寫地圖手動備援。",
    },
    "drop_lookup_loading": {
        "en": "Fetching drop data for {map} in the background…",
        "zh": "正在背景載入「{map}」的掉落資料…",
    },
    "drop_lookup_error": {
        "en": "Could not load the public database: {detail}",
        "zh": "無法載入公開資料庫：{detail}",
    },
    "drop_lookup_summary": {
        "en": "{map}  ·  {monsters} spawned monster(s)  ·  data updated {generated}",
        "zh": "{map}  ·  {monsters} 種生成怪物  ·  資料更新於 {generated}",
    },
    "drop_lookup_monster_meta": {
        "en": "spawn points {spawns}  ·  drop rows {drops}",
        "zh": "生成點 {spawns}  ·  掉落項目 {drops}",
    },
    "drop_lookup_no_monsters": {"en": "No monster spawn data for this map.", "zh": "這張地圖沒有可用的怪物生成資料。"},
    "drop_lookup_no_drops": {"en": "No published drop rows for this monster.", "zh": "此怪物目前沒有公開掉落項目。"},
    "drop_lookup_source_unknown": {"en": "Source not specified", "zh": "未標示資料來源"},
    "history_restarted_early": {"en": "restarted early", "zh": "提前重啟"},
    "history_export_button": {"en": "Export CSV", "zh": "匯出 CSV"},
    "history_export_title": {"en": "Export session history", "zh": "匯出紀錄"},
    "history_export_no_data": {"en": "There are no sessions to export yet.", "zh": "目前沒有可匯出的紀錄。"},
    "history_export_success": {"en": "Exported {n} session(s).", "zh": "已匯出 {n} 段紀錄。"},
    "history_export_failed": {"en": "Could not export the history: {detail}", "zh": "無法匯出紀錄：{detail}"},
    "history_clear_button": {"en": "Clear History", "zh": "清除紀錄"},
    "history_clear_confirm_title": {"en": "Clear history", "zh": "清除紀錄"},
    "history_clear_confirm_prompt": {
        "en": "Delete all {n} session(s)? This can't be undone.",
        "zh": "刪除全部 {n} 筆紀錄？此動作無法復原。",
    },
    "history_delete_confirm_title": {"en": "Delete session", "zh": "刪除紀錄"},
    "history_delete_confirm_prompt": {
        "en": "Delete \"{name}\"? This can't be undone.",
        "zh": "刪除「{name}」？此動作無法復原。",
    },

    "settings_window_scale": {"en": "WINDOW SCALE", "zh": "視窗縮放"},
    "settings_always_on_top": {"en": "Always on top", "zh": "永遠置頂"},
    "settings_session_interval": {"en": "SESSION INTERVAL", "zh": "紀錄區間"},
    "settings_context": {"en": "GAME CONTEXT", "zh": "遊戲資訊"},
    "settings_auto_context": {"en": "Detect job and map automatically", "zh": "自動辨識職業與地圖"},
    "settings_job_override": {"en": "Job fallback", "zh": "職業手動備援"},
    "settings_map_override": {"en": "Map fallback", "zh": "地圖手動備援"},
    "settings_apply_context": {"en": "Apply context", "zh": "套用遊戲資訊"},
    "settings_display": {"en": "DISPLAY", "zh": "顯示項目"},
    "settings_show_hp": {"en": "Show HP", "zh": "顯示 HP"},
    "settings_show_mp": {"en": "Show MP", "zh": "顯示 MP"},
    "settings_show_exp": {"en": "Show EXP", "zh": "顯示經驗值"},
    "settings_show_exp_pct": {"en": "Show EXP percentage", "zh": "顯示經驗值百分比"},
    "settings_show_level": {"en": "Show level", "zh": "顯示等級"},
    "settings_show_exp_diff": {"en": "Show EXP gained", "zh": "顯示經驗值增加"},
    "settings_show_exp_rate": {"en": "Show EXP / hour", "zh": "顯示每小時經驗值"},
    "settings_show_eta": {"en": "Show level-up ETA", "zh": "顯示升級預估時間"},
    "settings_show_proj_exp": {"en": "Show estimated session EXP", "zh": "顯示預估本次經驗值"},
    "settings_show_hp_loss": {"en": "Show HP loss", "zh": "顯示 HP 損失"},
    "settings_show_mp_loss": {"en": "Show MP loss", "zh": "顯示 MP 損失"},
    "settings_language": {"en": "LANGUAGE", "zh": "語言"},
    "settings_check_updates": {"en": "Check for updates", "zh": "檢查更新"},
    "settings_checking_updates": {"en": "Checking for updates...", "zh": "正在檢查更新..."},
    "settings_hud": {"en": "FLOATING HUD", "zh": "懸浮 HUD"},
    "settings_floating_on_start": {"en": "Enter floating HUD after Start", "zh": "按開始後進入懸浮 HUD"},
    "settings_floating_opacity": {"en": "HUD opacity", "zh": "HUD 透明度"},
    "settings_display_fields": {"en": "DISPLAY FIELDS", "zh": "顯示數據"},
    "settings_session": {"en": "SESSION", "zh": "紀錄行為"},
    "settings_auto_stop": {
        "en": "Stop after each record interval", "zh": "每個紀錄區間結束後停止",
    },
    "settings_save_on_restart": {
        "en": "Save to History when restarting", "zh": "重新開始時儲存至紀錄",
    },
    "settings_sampling": {"en": "SAMPLING", "zh": "取樣頻率"},
    "settings_sampling_value": {"en": "Status scan — {seconds:.1f}s", "zh": "狀態列取樣 — {seconds:.1f} 秒"},
    "settings_pickup_sampling_value": {
        "en": "Mesos scan — {seconds:.1f}s",
        "zh": "楓幣擷取 — {seconds:.1f} 秒",
    },
    "settings_economy": {"en": "ECONOMY TRACKING", "zh": "經濟統計"},
    "settings_track_pickup": {"en": "Track mesos from pickup messages", "zh": "從撿取訊息統計楓幣"},
    "settings_track_potions": {"en": "Track potion usage and recovery", "zh": "統計藥水消耗與回復"},
    "settings_default_recovery_hp": {"en": "HP recovery match after slot drop", "zh": "快捷欄下降後的 HP 回復辨識值"},
    "settings_default_recovery_mp": {"en": "MP recovery match after slot drop", "zh": "快捷欄下降後的 MP 回復辨識值"},
    "potion_compact": {"en": "{uses} uses · {cost:,}", "zh": "{uses} 次 · {cost:,}"},
    "potion_inventory_pending": {"en": "detecting initial quantities…", "zh": "正在偵測初始數量…"},
    "potion_inventory_unconfirmed": {"en": "latest OCR, pending confirmation", "zh": "最新 OCR，待確認"},
    "potion_inventory_compact": {"en": "start {values}", "zh": "初始 {values}"},
    "settings_potion_slots_hint": {
        "en": "Cost/uses come only from quantity decreases. Recovery is used only to label a confirmed potion event; natural/skill recovery is never charged.",
        "zh": "藥水次數／成本只在快捷欄數量下降時增加；回復量只用來標記已確認的喝水事件，自然／技能回復不會計入藥水。",
    },
    "settings_potion_slot": {"en": "Slot", "zh": "欄位"},
    "settings_potion_name": {"en": "Name", "zh": "藥水名稱"},
    "settings_potion_cost": {"en": "Cost", "zh": "單價"},
    "settings_potion_recovery": {"en": "Recovery", "zh": "回復量"},
    "settings_potion_kind": {"en": "Type", "zh": "類型"},
    "settings_apply_potions": {"en": "Apply potion settings", "zh": "套用藥水設定"},
    "settings_hp_short": {"en": "HP", "zh": "HP"},
    "settings_mp_short": {"en": "MP", "zh": "MP"},
    "settings_both_short": {"en": "HP/MP", "zh": "HP/MP"},

    "update_title": {"en": "MapleStoryAnalyzer update", "zh": "MapleStoryAnalyzer 更新"},
    "update_available": {
        "en": "Version {version} is available (current {current}).\n\nRelease notes:\n{notes}\n\nDownload it now?",
        "zh": "發現新版本 {version}（目前版本 {current}）。\n\n更新內容：\n{notes}\n\n現在下載嗎？",
    },
    "update_no_notes": {"en": "No release notes.", "zh": "沒有提供更新內容。"},
    "update_current": {"en": "You are already using the latest version ({version}).", "zh": "目前已是最新版本（{version}）。"},
    "update_status_idle": {"en": "Ready to check for updates.", "zh": "準備檢查更新。"},
    "update_status_dev": {"en": "Automatic updates require the packaged Windows app.", "zh": "自動更新需要使用 Windows 打包版。"},
    "update_status_waiting": {"en": "A background check is already running; waiting for its result...", "zh": "背景檢查正在執行，等待結果中..."},
    "update_status_checking": {"en": "Checking GitHub Releases...", "zh": "正在檢查 GitHub Release..."},
    "update_status_available": {"en": "Found version {version}; waiting for confirmation.", "zh": "發現版本 {version}，等待確認。"},
    "update_status_downloading": {"en": "Downloading version {version}...", "zh": "正在下載版本 {version}..."},
    "update_status_latest": {"en": "You are already using the latest version ({version}).", "zh": "目前已是最新版本（{version}）。"},
    "update_status_ready": {"en": "Version {version} downloaded and verified.", "zh": "版本 {version} 已下載並驗證完成。"},
    "update_status_installing": {"en": "Installing version {version} and restarting...", "zh": "正在安裝版本 {version} 並重新啟動..."},
    "update_status_cancelled": {"en": "Update cancelled.", "zh": "已取消更新。"},
    "update_status_error": {"en": "Update failed: {detail}", "zh": "更新失敗：{detail}"},
    "update_ready_title": {"en": "Update downloaded", "zh": "更新已下載"},
    "update_ready": {
        "en": "Version {version} is ready. Restart the app now to install it?",
        "zh": "版本 {version} 已準備完成。現在重新啟動程式並安裝嗎？",
    },
    "update_failed": {"en": "The update could not be completed: {detail}", "zh": "更新無法完成：{detail}"},
    "update_dev_build": {
        "en": "Automatic updates are available in the packaged Windows app.\n\nThis source/development build is not connected to the release updater.",
        "zh": "自動更新功能只提供給 Windows 打包版使用。\n\n目前是原始碼／開發版，未連接正式版更新器。",
    },

    "unit_min": {"en": "min", "zh": "分鐘"},
    "unit_min_short": {"en": "m", "zh": "分"},
    "unit_times_value": {"en": "{n} uses", "zh": "{n} 次"},
    "history_duration_early": {
        "en": "{dur}{unit} of {target}{unit}, {label}",
        "zh": "{dur}{unit}／{target}{unit}，{label}",
    },

    "rename_dialog_title": {"en": "Rename session", "zh": "重新命名紀錄"},
    "rename_dialog_prompt": {"en": "Session name:", "zh": "紀錄名稱："},
}


def t(key: str, lang: Lang, **kwargs: object) -> str:
    entry = _STRINGS.get(key)
    if entry is None:
        return key  # missing translation key -- fail loud-ish rather than KeyError mid-render
    text = entry.get(lang, entry["en"])
    return text.format(**kwargs) if kwargs else text
