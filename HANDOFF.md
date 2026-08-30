# Maple Insight / MapleStoryAnalyzer 工作交接

最後更新：2026-08-30（Asia/Taipei）  
正式專案：<https://github.com/ab159852-pixel/maplestory-analyzer>  
正式分支：`main`  
目前公開版本：`v1.0.41`  
`v1.0.41` 發佈提交：`bf316e6698ffdb7468005d6b7ba9cdaec2abd9d9`

## 1. 另一台電腦開始工作的方式

請把 GitHub 當成唯一的程式碼來源，不要從舊的解壓縮目錄、`dist` 或先前版本複製原始碼。

```powershell
git clone https://github.com/ab159852-pixel/maplestory-analyzer.git
cd maplestory-analyzer
git switch main
git pull --ff-only origin main
git status --short --branch
```

開始修改前，`git status` 應該是乾淨的。正式版資訊位於：

- `src/maple_analyzer/version.py`
- `.github/workflows/release.yml`

若只要執行已發佈版本，請下載 Release 中的 `MapleStoryAnalyzer-v1.0.41-win64.zip`，完整解壓縮後執行資料夾內的 `MapleStoryAnalyzer.exe`，不要單獨移動 exe。

## 2. 新 Codex 對話可直接貼上的接手指令

```text
請接續 Maple Insight / MapleStoryAnalyzer 專案。先完整閱讀 HANDOFF.md，確認目前 main、最新 Release 與本機 git status，再開始修改。不要從 dist 或舊版本回推原始碼。這次先以另一台設備的實測問題為主：快捷欄第 6 格實際為 1107，但程式顯示 107；第 7 格實際為 1134 且辨識正確。請把這張實測截圖重新加入對話後，先建立可重現測試，再修正，且不得破壞第 7 格、背景地圖辨識、楓幣去重、多解析度與自動更新。
```

## 3. 最新實測證據與目前優先問題

2026-08-30 在另一台設備的最新截圖顯示：

- 遊戲快捷欄第 6 格實際數量：`1107`
- 遊戲快捷欄第 7 格實際數量：`1134`
- Maple Insight 顯示第 6 格：`107`（少掉一個連續的 `1`）
- Maple Insight 顯示第 7 格：`1134`（正確）
- 地圖已顯示為 `寺院通道II`。
- 截圖當時尚未按「開始」，因此 LV／HP／MP／EXP 顯示 `--` 不能單獨當作狀態 OCR 故障證據。

這份 BMP 截圖沒有提交到 GitHub，以免用大型未壓縮圖片增加專案體積。接手時請使用者重新附上原始截圖，並從原圖裁出快捷欄第 6、7 格作為回歸樣本。

### 最可能的故障類型

第 7 格和地圖在同一張畫面中可以正常辨識，所以目前證據不支持「整張畫面或所有格子都整體位移」。第 6 格的 `1107 -> 107` 更像是：

1. 數字專用 OCR 使用 CTC greedy decode 時，把相鄰重複字元 `11` 合併成一個 `1`；或
2. 第 6 格數量遮罩把兩個窄的 `1` 合成單一連通區；或
3. 第 6 格局部 crop／前處理少保留了一條可區分兩個 `1` 的間隙。

不要先用整體平移或任意加寬所有快捷格修正。這會再次破壞第 7 格或讓相鄰格數字混入。應先在原始第 6 格 crop 上比較：原圖、quantity strip、各 numeric view、模型 raw text、leading-one recovery 與最後 cache/merge 結果。

## 4. 必須維持的產品規則

- 只對「設定頁面已指定」的快捷格執行 OCR；未設定格子不得浪費 OCR。
- 快捷欄數量範圍固定為 `0..9999`。
- HP 與 MP 藥水分開設定格子、單價、次數與成本。
- 藥水消耗只以快捷欄數量變動為主要依據；自然回復／技能回復不得直接算成喝水。
- OCR 誤判造成的暫時大量扣款，後續數量恢復時必須可逆向更正。
- 快捷欄應在程式開啟後、尚未按「開始」時就背景建立初始值。
- 職業與地圖也必須在程式開啟後背景辨識，讓掉落物速查不依賴開始測試。
- 地圖名稱使用小地圖資訊的第二排；必須保留 `II`／`III`／`IV` 等尾碼。
- 楓幣通知需要高頻取樣、涵蓋多排行訊息，並以事件／位置／時間去重，不能漏算也不能重複計算。
- 前景視窗不得遮蔽指定遊戲視窗擷取；優先使用 WGC／指定 HWND 的背景擷取。
- 所有快捷欄與狀態區域必須由實際遊戲 client frame 尺寸映射，不能以某台設備的螢幕解析度直接硬編碼。
- HUD 按鈕不得遮住顯示項目或遊戲 OCR 區域。
- 自動更新必須真正關閉舊程序、啟動外部更新器、替換整個 one-folder 套件並重開新版。

## 5. 主要程式位置

- `src/maple_analyzer/capture.py`：遊戲 HWND 選取、WGC／PrintWindow／桌面相容擷取與遮擋判斷。
- `src/maple_analyzer/regions.py`：參考座標、多解析度映射、快捷欄 8 格區域。
- `src/maple_analyzer/ocr.py`：RapidOCR、數字專用 OCR、快捷欄前處理、`1` 重複字元修復、`3/9` 本地模板與 OCR cache。
- `src/maple_analyzer/monitor.py`：未開始測試時的背景職業／地圖／快捷欄取樣。
- `src/maple_analyzer/overlay.py`：UI、工作執行緒、即時資料合併、開始／暫停／停止與 HUD。
- `src/maple_analyzer/economy.py`：藥水與楓幣帳本、可逆更正及費用計算。
- `src/maple_analyzer/drop_lookup.py`：地圖正規化、怪物／掉落物速查。
- `src/maple_analyzer/updates.py`：更新檢查、下載及外部更新器啟動。
- `tests/test_ocr_pipeline.py`：快捷欄 OCR 的主要單元與回歸測試。
- `tests/test_regions.py`、`tests/test_regions_resolution.py`：多解析度區域測試。
- `tests/test_context.py`：職業／地圖背景 OCR。
- `tests/test_economy.py`：藥水扣款、誤判回歸及楓幣去重。
- `tests/test_updates.py`：外部更新器與舊程序終止。

## 6. 下一輪修復的正確順序

1. 從最新原始 BMP 依目前 capture metadata 取得遊戲 client frame，而不是從桌面截圖座標直接裁切。
2. 將第 6 格 `1107` 和第 7 格 `1134` 的 cell、quantity strip、所有 numeric views 輸出成診斷圖。
3. 記錄數字模型每個 view 的 raw text，確認 `1107 -> 107` 發生在模型解碼、leading-one recovery，還是後段 cache merge。
4. 先加入失敗的回歸測試；測試必須同時斷言第 6 格為 `1107`、第 7 格為 `1134`。
5. 使用局部且可證明的修正處理重複 `1`，不要更動整體快捷欄 parent geometry，除非診斷圖證明 parent 本身錯位。
6. 跑快捷欄、區域、背景 context、經濟帳本與更新器測試，再跑完整 pytest。
7. 只有通過實圖回歸及完整測試後才提升版本、commit、push tag，讓 GitHub Actions 發佈。

## 7. 本機測試與執行

建議 Python 3.11：

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt -r requirements.win.lock.txt -r requirements-dev.txt
.venv\Scripts\python -m pytest -q
.venv\Scripts\python scripts\run_overlay.py
```

針對本次問題先跑：

```powershell
.venv\Scripts\python -m pytest -q tests/test_ocr_pipeline.py tests/test_regions.py tests/test_regions_resolution.py tests/test_context.py tests/test_economy.py tests/test_updates.py
```

## 8. 發佈與版本規則

不要重用舊 tag，也不要只上傳 exe。發佈前：

1. 修改 `src/maple_analyzer/version.py`。
2. 確認 `git status` 只含本次預期修改。
3. 執行測試並保留結果。
4. commit 並 push `main`。
5. 建立同版號 tag，例如 `v1.0.42`，再 push tag。
6. 等 GitHub Actions 產生 zip 與 `checksums.txt`。
7. 用前一正式版實際走一次「檢查更新 → 下載 → 關閉舊程序 → 替換 → 自動重啟 → 版本更新」端到端測試。

交接文件本身不是新功能版本，不需要僅為加入此文件而提升 `APP_VERSION` 或建立新 tag。
