# Maple Insight

*[繁體中文版 README](README.md)*

A small always-on-top HUD that watches your MapleStory game window and tracks
your HP, MP, EXP, and Level in real time. It uses OCR (optical character
recognition) to read the stat numbers straight off your screen — no typing,
no macros, no game files or memory access. Use it to see how much EXP/HP/MP
you're burning per grinding session, compare sessions, and get a rough ETA
to your next level.

Read-only: it only looks at your screen, it never clicks, types, or sends
anything to the game.

![MapleStoryAnalyzer running alongside the game](docs/screenshot.jpg)

*(HUD shown in Chinese — switch to English any time in Settings.)*

▶️ **[Demo video](https://youtu.be/Joqzcg6g798)** — see it running live.

## Features

- **Live tracking** — LV/HP/MP/EXP update every 0.3 seconds by default, with a
  progress bar for each. A status pill shows Calibrating/Tracking/Idle/
  Paused/Stopped, and a countdown chip shows how long is left in the current
  session. **The app opens Stopped and does not start tracking on launch** —
  hit **Start** to begin a session; the first second or two shows
  "Calibrating..." while it confirms the real max HP/MP and starting EXP
  before counting anything.
- **Pause/Resume** — pause the current session any time. LV/HP/MP/EXP keep
  reading live while paused, but nothing is added to EXP gained or HP/MP
  lost and the timer stops counting down; Resume picks back up from where
  it left off, and whatever happened during the pause isn't counted.
- **Stop Test** — after Start, both the full view and floating HUD provide
  Pause and Stop Test controls. Stop saves the current interval to History,
  freezes its timer, and lets you start a fresh interval without restoring the
  full interface first.
- **Sessions** — stats reset on a timer (default 10 minutes, adjustable) so
  "EXP diff" always means "since this session started." When the timer ends
  it commits to History and stops by default (toggle off in Settings to go
  back to immediately starting the next session instead). Hit **Restart
  Session** any time to end the current one early — it saves to History by
  default too (also toggleable).
- **History** — every finished session becomes a card: start→end time, EXP
  gained (with %), mesos, separate HP/MP potion uses and costs, natural versus
  potion recovery, and estimated mesos saved by natural/skill recovery.
  Newest session is always at the top. Click a card's title to give it a custom
  name (e.g. "Ellinia Forest"), or the "×" in its top-right corner to delete it.
  History can also be exported as an Excel-friendly UTF-8 CSV.
- **Mesos tracking** — reads newly appearing `楓幣` pickup messages on the right
  side of the game, so opening the inventory is not required. Persistent OCR
  lines are deduplicated before they are added; if OCR drops the `楓幣` glyphs,
  a conservative pickup-text fallback still recovers the amount.
- **Potion/recovery tracking** — watches up to eight shortcut slots. Only
  enabled/configured slots are tracked; when no slot is enabled, shortcut OCR
  and potion accounting stay idle. The eight cell coordinates are still
  prepared from the game client geometry, and only configured cells are sent
  to OCR. Entering a name, price, type, and fixed recovery amount classifies
  the slot and calculates cost. Potion uses and costs increase only when a shortcut
  quantity decreases. A lower OCR value must be confirmed twice, and
  implausibly large jumps are ignored, so a drink animation or a missing digit
  cannot become hundreds of potions. The recovery amount labels a recovery as
  potion recovery only after a confirmed slot drop; natural/skill recovery
  without a slot drop is never charged as a potion and is valued at 1.2
  mesos/HP and 2.1 mesos/MP.
- **Map drop lookup** — after the map is detected, click **Lookup drops** to
  load the map's spawned monsters and their public drop rows inside the app.
  Each monster expands/collapses independently and shows category, source,
  published probability, and quantity range. The ↗ buttons open the original
  [MapleMemory database](https://morrisrrrrrrr-svg.github.io/). Data is fetched
  in the background and cached for the session; a missing probability is shown
  as `—` rather than guessed.
- **Resolution-aware OCR** — capture boxes are mapped to the actual game client
  pixels with aspect-preserving scaling, horizontal letterbox handling, and
  bottom anchoring. Numeric OCR retries only structurally weak or low-confidence
  crops with enlarged/contrast-normalized input instead of running slow
  detection OCR on every tick.
- **Automatic updates** — the packaged Windows app checks GitHub Releases in the
  background, offers newer versions, verifies the downloaded ZIP with SHA-256,
  then uses a detached updater to restart and replace the app without touching
  your settings or history. You can also use **Settings → Check for updates**.
- **Premium workspace UI** — the Live, History, and Settings tabs use a layered
  obsidian/sapphire card layout with a scrollable overview and a collapsible
  drop panel, while Start can still switch to the translucent horizontal HUD.
- **Settings**:
  - **Window scale** — adjust the whole-window scale live with a +/− stepper.
    If the current CustomTkinter runtime cannot apply it safely, the UI marks
    the value for the next launch instead.
  - **Always on top** — toggle whether the HUD stays above the game.
  - **Floating HUD** — optionally hide the tabs, stay on top, and become
    semi-transparent after Start. Opacity is adjustable from 45% to 100%; the
    HUD button restores the full interface.
  - **Language** — switch between 中文 and English any time, instantly.
  - **Session interval** — how often a session auto-resets (1–60 min).
  - **Sampling** — status/OCR interval, 0.3 seconds by default (adjustable
    from 0.2 to 1.0 seconds).
  - **Stop automatically when the timer ends** — on by default; turn off to
    have the timer immediately start the next session instead (the old
    behavior).
  - **Save to History when restarting** — on by default; turn off to have
    Restart discard the current session instead of saving it.
  - **Display fields** — individually show/hide level, HP, MP, EXP, EXP gain,
    EXP/hour, ETA, HP/MP loss, mesos, potion, recovery, and savings fields. The
    floating HUD can therefore contain only the values you need.
  - **Economy tracking** — toggle pickup/potion tracking and configure the
    shortcut-slot names, prices, types, and recovery values.
- **Level-up ETA** — once a session has a few seconds of data, estimates
  time-to-next-level from your current EXP rate.
- **Estimated session EXP** — projects "at this rate, the whole session will
  end up earning about this much" from the current EXP rate.

## Requirements

- **Windows** — a live Windows desktop is required; this won't run for real on
  macOS/Linux. On Windows 10 1903+, the app prefers Windows Graphics Capture
  targeted at the MapleStory HWND, so a foreground analyzer, browser, or other
  window cannot replace the pixels sent to OCR.
- MapleStory installed and running.
- **Game resolution: 1366×768 or larger recommended.** Verified working at
  both 1366×768 and 1920×1080. Capture coordinates follow the actual client
  pixels automatically. Bigger windows render the stat text larger and read
  more reliably; very small windows misread more often.

## Install

No Python or setup needed — just the `MapleStoryAnalyzer` folder containing
`MapleStoryAnalyzer.exe`. Put it wherever you like (e.g. `Desktop\MapleStoryAnalyzer`)
and keep the folder intact — the .exe needs the files alongside it.

## Launch tutorial

1. Have MapleStory running and visible. It does not need to be focused before
   launch, and with Windows Graphics Capture it does not need to stay in front
   while tracking.
2. Double-click `MapleStoryAnalyzer.exe`. Windows may show a SmartScreen
   warning on first run since it isn't code-signed — click **More info →
   Run anyway**.
3. A small window titled "Maple Insight" opens, always on top, on the
   **Live** tab by default. Once the game window is found, LV/HP/MP/EXP should
   start filling in within a second or two.
4. Click into the game and play normally — the HUD keeps reading in the
   background. Before grinding, use **Settings → Economy Tracking** to map
   potion slots and prices. Once a map is detected, use **Live → Lookup drops**
   and click a monster row to expand its item list. Switch to **History** any
   time to see past sessions.
5. New releases are offered in the background; you can also use **Settings →
   Check for updates**. After the download finishes, accept the restart and the
   updater will install it automatically.
6. Close the window like any other app when you're done — nothing needs to be
   shut down separately.

## Run with Python instead

Prefer running from source instead of the .exe? You'll need:

- **Python 3.10** (via the `py` launcher, e.g. `py -3.10`).
- MapleStory installed and running (same as above).

Open a terminal in this project's folder and run once:

```powershell
py -3.10 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python .venv\Scripts\pywin32_postinstall.py -install
```

This creates a `.venv` folder with everything the app needs (OCR engine, UI
toolkit, screen-capture libraries). You only need to do this once; after
that, launch with:

```powershell
.venv\Scripts\python scripts\run_overlay.py
```

Same Launch tutorial and Troubleshooting apply either way — the app behaves
identically whether started via the .exe or this command.

## Publish a release (maintainers)

This source currently contains `APP_VERSION = 1.0.25`. For each release, bump
`APP_VERSION` in `src/maple_analyzer/version.py`, commit the change, and push a
matching tag, for example:

```powershell
git tag v1.0.25
git push origin v1.0.25
```

GitHub Actions installs dependencies, builds
`MapleStoryAnalyzer-v1.0.25-win64.zip`, writes a SHA-256 checksum, and publishes
the Release automatically. Packaged users will be offered the update next time
the app starts.

## Troubleshooting

- **All fields show `--` / blank.** Check that the game is not minimized and
  is still rendering. Windows Graphics Capture reads the target window without
  requiring foreground focus; if WGC is unavailable, the app falls back to
  desktop capture and other windows must stay off the stat strip.
- **Status pill says "Game window not found."** The HUD looks for a window
  titled `新楓之谷` by default. If your client's window title is different,
  this won't match it.
- **Status pill says "Stat panel is covered."** Windows Graphics Capture is
  unavailable and the app has fallen back to desktop capture, so another
  window is sitting over the game's bottom-left LV/HP/MP/EXP strip. Move it
  away and tracking resumes within a couple of seconds. With WGC available,
  foreground windows do not cause this state.
- **The stat bar is hidden behind the Windows taskbar.** With the game
  maximised, the bottom stat strip can sit underneath the taskbar. Run the game
  windowed, or set the taskbar to auto-hide.
- **Status pill says "Game window minimized."** Restore the game window. It
  does not need to be the foreground window, but a minimized or non-rendering
  window has no fresh pixels for WGC to deliver.
- **A field is occasionally wrong for one tick, then corrects itself.**
  OCR misreads happen — usually caused by a combat effect or floating damage
  number covering a stat bar for a frame. The HUD carries forward the last
  known good value instead of flashing blank, so a single bad tick shouldn't
  be visible.
- **The map reads "第3軍管" and drop lookup says not found.** The tiny map font
  can confuse 營 with a similar glyph; numbered barracks names are normalized
  for that OCR error. A map fallback can still be entered in Settings if needed.
- **HP/MP loss looks too high, too low, or plain wrong.** Treat HP/MP loss as
  a rough figure, not an exact count. Two reasons. *Sampling:* the HUD reads
  the bars every 300 ms by default and adds up the drops between consecutive
  reads, so damage you heal back inside one sampling gap is never seen, and several
  hits landing in the same gap count as one. *OCR:* HP/MP loss is a running
  total, so a single misread digit (824 read as 24) adds a large phantom loss
  that stays in the session total for good. EXP doesn't have this problem —
  it's just end minus start, so a bad reading in the middle corrects itself on
  the next tick. If a session's HP/MP looks absurd, hit **Restart Session** to
  start the count over.
- **Numbers look frozen / EXP not moving even though I'm playing.** Check the
  status pill — if it says "Tracking," data is flowing; if HP/MP/EXP truly
  haven't changed, that's genuinely idle (not a bug). If the pill shows an
  error instead, see the two bullets above.
- **Text is too small/cramped, or the window is an awkward size.** Settings
  → Window Scale, use the +/− stepper; it normally applies immediately. If the
  UI says to restart, reopen the app to apply it safely. There's also a
  scrollbar in Settings if some options are cut off at very small scales.
- **Want it to stop covering the game.** Settings → turn off "Always on top,"
  or just move/resize the window like any other.
- **EXP occasionally pauses for a second before updating.** This is by design:
  each reading is checked for plausibility, and a frame that fails is discarded
  in favour of the last good value until the next frame reads correctly. Half a
  second of staleness beats showing a wrong number.
- **Session numbers look "off" after clicking Restart quickly twice.** By
  design, a restart within 1 second of the previous one is ignored (avoids
  logging a meaningless 0-duration entry) — this is expected, not a bug.
- **Mesos or potions are not increasing.** Keep the right-side pickup feed and
  lower-right shortcut bar visible and unobstructed. Potion costs are based only
  on shortcut quantity decreases; the fixed recovery amount only confirms a
  potion recovery after such a decrease, so unreadable quantity text is not
  silently converted into a potion use. The interval cost estimate waits for
  about 60 seconds of data so the first drink does not create a short-sample
  spike.
- **Drop lookup cannot load.** It needs a connection to the public [MapleMemory
  map database](https://morrisrrrrrrr-svg.github.io/maps.html). Check the network
  and try again; if OCR has not found a map, enter a map fallback in Settings.
  A `—` probability means the source did not publish a numeric value, not that
  the item has a zero chance.
- **The window looks busy immediately after launch.** OCR models load in the
  background now; the HUD should remain interactive while the status says
  “Loading OCR”. Wait for it to finish instead of launching another copy.
- **Automatic update failed.** Check the network and make sure the application
  folder is writable; Windows may require administrator permission if it is
  under `Program Files`. You can download the GitHub Release ZIP and replace
  the `MapleStoryAnalyzer` folder manually. Settings and history in
  `%LOCALAPPDATA%` are kept separately.

## Data and privacy

Settings and history are stored outside the program folder at
`%LOCALAPPDATA%\MapleStoryAnalyzer` (`settings.json` and `history.json`). The
app remains screen-only: it does not read game files or memory and never sends
input to the game.

