# Gemini 3.5 Transcribe 影音語音辨識與語者分離系統 🎙️

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Model](https://img.shields.io/badge/ASR-gemini--3.5--transcribe-8E75B2.svg)](https://ai.google.dev/)
[![LLM](https://img.shields.io/badge/LLM-Gemini_3.5_Flash_Lite-4285F4.svg)](https://ai.google.dev/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

基於 Google **Gemini 3.5 Transcribe** 模型的端到端影音辨識與多格式字幕產生系統。支援長影音自動剝離音訊、動態均分切段、語者分離（Speaker Diarization）、專有詞彙同音字校正、台灣繁體中文自動轉換、**Reflective Translation 兩階段反思翻譯與雙語字幕**，以及一鍵匯出 **SRT / VTT / TXT / JSON** 四種格式字幕。

---

## 📑 目錄
- [🌟 核心特色](#-核心特色)
- [🧩 系統架構流程](#-系統架構流程)
- [🛠️ 核心避坑防護設計](#️-核心避坑防護設計)
- [🚀 快速開始](#-快速開始)
  - [1. 環境準備](#1-環境準備)
  - [2. 安裝依賴](#2-安裝依賴)
  - [3. 設定 API Key](#3-設定-api-key)
  - [4. 啟動 Web 介面](#4-啟動-web-介面)
- [🖥️ 介面功能詳解](#️-介面功能詳解)
- [🧪 執行單元測試](#-執行單元測試)
- [📂 專案架構](#-專案架構)
- [❓ 常見問題與排錯 (FAQ)](#-常見問題與排錯-faq)
- [🙏 參考與致謝](#-參考與致謝)

---

## 🌟 核心特色

1. **影音雙棲支援 & 2GB 大檔上傳**：
   - 支援 MP4, MOV, MKV, AVI, WEBM, WAV, MP3, M4A, FLAC, OGG。
   - 內建設定解除 Streamlit 預設 200MB 上傳限制，**最高支援 2GB（2048 MB）超長高畫質影片直接上傳**，由後端自動提取為 16kHz 單聲道 16-bit PCM WAV。
2. **長音訊無零頭動態均分**：
   - 音訊超過 30 分鐘時，採動態平均切分：`count = ceil(總時長 / 1200秒)`。
   - 每段時長均等且皆小於 20 分鐘，既避開 API 30 分鐘上限，又杜絕產生數秒鐘的極短零頭段落。
3. **高精度語者分離 & 純淨無標籤雙模式**：
   - **開啟時**：透過 Interactions API 識別不同發言者（`語者 1`, `語者 2`...），跨段落自動加上命名空間（如 `[第1段-語者1]`、`[第2段-語者1]`）。
   - **關閉時**：全面壓制語者前綴，匯出之 SRT / VTT / TXT 與預覽畫面皆為乾淨純淨字幕，**絕不殘留 `[語者 1]` 假定標籤**。
4. **專有詞彙 LLM 兩階段校正**：
   - 突破官方 API 限制（原生 `custom_vocabulary` 與語者分離互斥）。
   - 轉錄後交由 Gemini Flash 進行專有名詞、同音字精準校準，完全不影響時間戳與語者標記。
5. **網頁即時切換五款 LLM 模型**：
   - 側邊欄隨選校正、翻譯與推斷模型，預設 **Gemini 3.5 Flash Lite**，並提供 **Gemini 3.5 Flash**、**Gemini 3.6 Flash**、**Gemini 3.7 Flash**、**Gemini 3.8 Flash**，無需更改任何程式碼。
6. **台灣繁體中文 (OpenCC s2twp) 自動轉換**：
   - 解決 Google 語音模型原生偏向輸出簡體字的問題。
   - 預設自動進行台灣用語與標準正體轉換（如「軟件」->「軟體」、「項目」->「專案」）。
7. **現有字幕直讀載入（.srt / .vtt 獨立翻譯）**：
   - 支援直接拖曳上傳現有的 `.srt` 或 `.vtt` 字幕檔，自動精準解析時間戳與原有字幕內容。
   - **無需重新上傳龐大影音或耗時轉錄**，可直接套用專有詞彙保護與反思式翻譯，大幅節省時間與 API 配額。
8. **語意合句器 (Sentence-Level Merging) 徹底根除超翻**：
   - 解決傳統逐行翻譯因主從子句拆裂導致的「跨行超翻」與「倒裝錯位」問題。
   - 自動結合標點（`.?!`、`。？！`）、語者更換、對話停頓間隔（`>0.8s`）及安全上限，將破碎字幕重組為完整的語意句塊（SentenceBlock）進行高品質翻譯。
9. **雙軌字幕對齊架構 (Dual-Mode Alignment)**：
   - **模式 1（原行等比切分）**：100% 維持原始行數與時間戳，內建**零空白鐵律（Zero-Empty Guarantee）**保護極短碎片（如 0.3 秒 `well.`），並具備**子句逗號優先吸附**與**引號專有名詞防撕裂保護**。
   - **模式 2（影視級智慧重劃，預設）**：依台灣繁體閱讀節奏（單行 20~25 字）重新分行並平滑時間戳，消除破碎短字幕。
   - **雙語對照字幕**：自動鎖定模式 1，確保繁體中文譯文與英文原文 1:1 絕對對齊。
10. **429 頻率限制防護、批次擴容（50 句）與主動節流**：
    - 擴大單批容量至 50 句（`BATCH_SIZE = 50`），將 260+ 句的字幕請求數驟降至 5~6 次，**天然保持在免費版 15 RPM 上限的 40% 以內**。
    - **智慧冷卻重試（Smart Backoff）**：遭遇 429 RESOURCE_EXHAUSTED 時，自動解析 Google 回傳的 `retryDelay` 進行休眠冷卻，前端即時提示剩餘秒數並自動重試，告別連環報錯。
    - **批次間主動節流（Pacing）**：批次間自動加入 2.0 秒預設延遲，平滑分散流量。
11. **AI 跨段語者統一與真名推斷**：
    - 由 Gemini 分析對話中的自我介紹（如「我是 Sarah」）或稱謂證據（如「Evan 你進度如何」）自動對齊跨段代號並標記真實姓名（如 `[Evan (語者 1)]`）；無確鑿證據則忠實保留代號，嚴防幻覺。
12. **四大格式一鍵匯出**：
    - **SRT**：標準影音播放器字幕（支援純譯文、雙語對照或純淨無標籤）。
    - **VTT**：網頁 HTML5 `<video>` 播放字幕。
    - **TXT**：發言者標註或純時間戳之會議逐字稿。
    - **JSON**：含字詞層級時間戳（`word_info`）之完整結構化數據。
13. **隱私與安全**：
    - 轉錄完成後立即於 `finally` 區塊刪除 Google Files API 雲端副本與本機暫存檔案。

---

## 🧩 系統架構流程

```mermaid
graph TD
    A[使用者上傳影音檔案 最大 2GB] --> B[audio_processor.py]
    B -->|影片檔案| C[FFmpeg 抽取音訊為 16kHz WAV]
    B -->|音訊檔案| C
    C -->|檢查時長| D{大於 30 分鐘?}
    D -->|是| E[動態均分為 N 段<br/>每段 <= 20 分鐘]
    D -->|否| F[單段處理]
    E --> G[gemini_client.py]
    F --> G
    G --> H[Google Files API 暫存上傳]
    H --> I[Interactions API 批次轉錄<br/>gemini-3.5-transcribe]
    I --> J[即時清理 Files API 暫存]
    I --> K[transcript_formatter.py]
    K --> L[字詞時間戳解析 & CJK 空格防護]
    K --> M[語者分離判斷: 開啟/無標籤純淨模式]
    M --> N[智慧自然斷句與字幕分行]
    N --> O[vocabulary_corrector.py<br/>Gemini Flash 專有詞彙校正]
    O --> P[OpenCC 繁體中文轉換 s2twp]
    P --> Q{外語語音?<br/>反思翻譯}
    Q -->|是| R[translator.py<br/>兩階段反思翻譯 & 雙語對照]
    Q -->|否| S[Streamlit Web 介面]
    R --> S
    S -->|可選| T[AI 跨段語者對齊與人名推斷]
    S --> U[匯出 SRT / VTT / TXT / JSON]
```

---

## 🛠️ 核心避坑防護設計

| 潛在問題 / API 陷阱 | 官方 / 框架原生行為 | 本專案解決方案 |
| :--- | :--- | :--- |
| **Streamlit 200MB 上傳門禁** | `st.file_uploader` 預設限制 200MB，超過直接在網頁端拒收報錯 | 內建 [`.streamlit/config.toml`](.streamlit/config.toml)，將上限調升至 2048MB（2GB） |
| **語者分離無標註** | 缺少 `timestamp_granularities: ["word"]` 時，API 回傳 200 但完全不帶語者代號 | 強制在請求體中帶入 `timestamp_granularities: ["word"]` |
| **關閉語者分離仍帶標籤** | 若無語者資訊，多數系統會 fallback 強制指派「語者 1」 | 實作純淨模式判斷，關閉語者分離時嚴格過濾所有 `[語者 1]` 標籤 |
| **專有詞彙與語者互斥** | 在 Interactions API 同時傳入 `custom_vocabulary` 與語者分離會直接報錯 HTTP 400 | 改由兩階段管線處理：取得轉錄結果後，交由 Gemini Flash 進行二次同音詞對齊校正 |
| **長字幕翻譯 JSON 報錯** | 翻譯上百句字幕時，LLM 輸出引號未跳脫拋出 `Expecting ',' delimiter` | 採 40 句分批處理 + Pydantic Schema 約束 + Regex 容錯萃取器 |
| **macOS MIME 型態錯誤** | macOS 內建推斷 WAV 為 `audio/x-wav`，與 API payload 的 `audio/wav` 不符報錯 400 | 上傳時顯式聲明 `audio/wav`，並動態抓取雲端資源 MIME 嚴格對齊傳入 |
| **切段零頭散段** | 60 分 5 秒若切 30m+30m+5s，尾段浪費配額且辨識差 | 採均分演算法 `ceil(3605/1200) = 4` 段，每段 15 分鐘均勻分配 |
| **跨段語者代號衝突** | 第 2 段的 `spk:0` 與第 1 段的 `spk:0` 無關，直連會造成誤讀 | 加入段落命名空間 `第1段-語者1`，並提供 AI 依語境推斷對齊 |
| **中文單字空格陷阱** | 字詞標註連綴時英文需空格，中文會變成「你 好 世 界」 | 實作 CJK 字元判定與標點符號自動貼合演算法 |
| **預設輸出簡體中文** | Google 語音模型訓練基底多為簡體中文 | 內建 OpenCC 台灣習慣用語轉換器（`s2twp`），預設自動轉繁體 |
| **Free Tier 15 RPM 限制與 429 報錯** | 翻譯長字幕時連續快速發送請求，30 秒內觸發 15 RPM 上限並引發連環失敗 | 擴大單批為 50 句（請求數降至 6 次以內），內建 429 `retryDelay` 指數退避冷卻重試與 2 秒批次間主動節流 |
| **字幕翻譯倒裝與跨行超翻** | 原字幕在子句間碎裂，逐行翻譯導致前後文錯位倒裝與嚴重超翻 | 實作 `Sentence Merger` 跨行語意合句器，先合成完整句塊翻譯，再雙軌對齊回字幕時間軸 |
| **雙語字幕短碎片空白與引號斷裂** | 等比分行時短碎片（如 0.3s `well.`）誤吸附句尾句號導致空字幕，或腰斬引號名詞 | 實作**零空白鐵律（Zero-Empty Guarantee）**、子句標點優先吸附與**引號區間（Quote Spans）**保護 |
| **現有字幕獨立翻譯需求** | 手邊已有現成 SRT/VTT 檔案，卻被迫重新上傳大檔案影音執行冗長轉錄 | 支援直接拖曳上傳 `.srt` 與 `.vtt`，自動解析時間軸與文字後直接執行專有詞彙校正與反思翻譯 |

---

## 🚀 快速開始

### 1. 環境準備
- 系統支援：macOS, Linux, Windows
- Python 版本：`Python 3.12+`

### 2. 安裝依賴

本專案自帶 `imageio-ffmpeg` 二進位執行檔，**無需另外手動安裝系統級 FFmpeg**！

**推薦使用 `uv` 進行快速安裝：**
```bash
# 安裝 uv (若尚未安裝)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 複製儲存庫
git clone https://github.com/japen0617/gemini_Transcribe.git
cd gemini_Transcribe

# 建立虛擬環境並安裝依賴
uv sync
```

*或使用一般 pip：*
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r <(python3 -c "import toml; [print(d) for d in toml.load('pyproject.toml')['project']['dependencies']]")
```

### 3. 設定 API Key
請前往 [Google AI Studio](https://aistudio.google.com/) 取得免費的 Gemini API Key。

複製範本檔案建立 `.env`：
```bash
cp .env.example .env
```
在 `.env` 中填入金鑰：
```env
GEMINI_API_KEY=your_gemini_api_key_here
```
*(亦可直接在 Web 側邊欄輸入，兩者皆支援)*

### 4. 啟動 Web 介面
```bash
uv run streamlit run app.py
# 或使用虛擬環境：
# source .venv/bin/activate && streamlit run app.py
```
啟動後於瀏覽器打開 `http://localhost:8501` 即可開始體驗！

---

## 🖥️ 介面功能詳解

1. **左側邊欄**：
   - **Gemini Developer API Key**：輸入或讀取環境變數。
   - **開啟語者分離**：勾選以識別不同發言者；**取消勾選則輸出純淨字幕（完全無語者標籤）**。
   - **語言偏好**：預設 `繁體中文 (zh-TW)`，支援自動偵測、英文、日文等。
   - **字幕輸出字體**：預設 `繁體中文 (台灣習慣用語 s2twp)`，提供標準繁體、簡體或原始輸出。
   - **反思式翻譯開關**：預設勾選，當偵測為外語時自動啟動兩階段反思翻譯為台灣繁體正體。
   - **專有詞彙與翻譯模型**：可自由挑選 `Gemini 3.5 Flash Lite` (預設)、`3.5 Flash`、`3.6 Flash`、`3.7 Flash`、`3.8 Flash`。
   - **斷句參數微調**：停頓間隔秒數門檻（預設 1.2 秒）與單行最大字數（預設 30 字）。
2. **主操作區**：
   - **影音 / 字幕檔拖曳上傳**：
     - 上傳影音檔案時，一鍵執行音訊抽取、動態均分與語音辨識。
     - **直接上傳 `.srt` 或 `.vtt` 時**，按鈕自動切換為「📄 載入現有字幕檔案並準備翻譯」，瞬間載入並呈現預覽，無需音訊檔即可直接反思翻譯。
   - **專有詞彙與術語清單**：同時支援【轉錄校正】與【反思翻譯】：
     - **保留原文**：直接輸入英文詞彙（如 `switch, AP, Extreme Switching`），翻譯時強制鎖定保留英文，絕不誤翻為中文。
     - **指定譯法**：輸入對照格式（如 `network -> 網路`, `stacking -> 堆疊`），翻譯時嚴格採用此譯名。
   - **會議筆記**：可輸入參與者名單備忘，提供 AI 更充份的推斷線索。
   - **反思翻譯選項與智慧進度回饋**：
     - 可勾選「嚴格維持原字幕行數與時間戳（模式 1）」，預設採用模式 2（影視級智慧重劃）。
     - 即時顯示當前翻譯句數進度，若遭遇 API 頻率限制，自動顯示 429 冷卻倒數提示。
   - **字幕顯示模式即時切換**：可隨時切換「正體中文譯文」、「雙語對照字幕 (繁體中文 + 原文)」或「原始轉錄文字」，即時切換預覽無延遲。
   - **AI 反思筆記展示**：折疊面板呈現術語保留考量與口語調整細節。
   - **AI 跨段對齊**：語者分離開啟時，可點擊「🤖 執行跨段對齊與人名推斷」查看稱謂證據報告。
   - **字幕即時預覽與四大下載按鈕**：直接下載 `.srt`, `.vtt`, `.txt`, `.json`。

---

## 🧪 執行單元測試

本專案附帶 **28 項完整的自動化單元測試**，涵蓋動態切段、音訊提取、CJK 空格過濾、時間戳計算、字幕解析、雙軌對齊、零空白保證、引號防斷裂與 429 智慧冷卻重試驗證：
```bash
uv run pytest -v
```

測試涵蓋重點：
- `test_calculate_chunk_plan_short` / `long`：驗證短音訊不切段、長音訊動態均分無零頭散段。
- `test_extract_audio_and_duration`：驗證音訊抽取與採樣率 16000Hz 轉換。
- `test_cjk_detection_and_space`：驗證中文不空格、英文正確空格。
- `test_timestamp_formatting`：驗證 SRT/VTT 毫秒與逗號/點號格式。
- `test_convert_subtitles_script`：驗證 OpenCC 台灣常用語繁體字轉換。
- `test_diarization_disabled`：驗證關閉語者分離時完全無 `[語者 1]` 標籤。
- `test_parse_srt_content` & `test_srt_upload_result_structure_and_export`：驗證直讀上傳之 SRT/VTT 解析與匯出生命週期。
- `test_merge_subtitles_to_sentences`：驗證跨行標點合句、語者換人切分與單句安全上限。
- `test_split_translation_proportionally`：驗證模式 1 等比切分與字元無損還原。
- `test_split_translation_zero_empty_and_quote_protection`：驗證零空白鐵律（極短碎片 `well.` 絕不為空）與引號專有名詞（`「Meet Extreme Switching」`）防撕裂保護。
- `test_reflow_translation_to_subtitles`：驗證模式 2 影視智慧重劃長句為 20~25 字平滑字幕。
- `test_reflective_translate_dual_mode_and_bilingual_lock`：驗證雙語字幕強制鎖定模式 1 行行對齊。
- `test_reflective_translate_429_cooling_and_retry`：驗證遭遇 429 頻率限制時自動讀取 `retryDelay` 休眠冷卻並成功重試。
- `test_meet_extreme_switching_real_file`：使用真實 657 行 Extreme Switching SRT 驗證 263 句完整語意合句與雙軌重切分。
- `test_batching_translation`：驗證超過 50 句字幕時之自動分批與主動節流機制。
- `test_safe_parse_with_unescaped_quotes`：驗證譯文包含未跳脫引號時的 Regex 容錯恢復。

---

## 📂 專案架構

```
gemini_Transcribe/
├── .streamlit/
│   └── config.toml             # Streamlit 配置：放寬上傳上限至 2GB
├── app.py                      # Streamlit 前端網頁應用
├── audio_processor.py          # 影片音訊剝離、時長偵測與動態均分切段模組
├── gemini_client.py            # Gemini Interactions API 與 Files API 客戶端
├── transcript_formatter.py     # 字詞時間戳解析、CJK 空格處理、OpenCC 轉換與字幕匯出
├── vocabulary_corrector.py     # 專有詞彙同音字校正與 AI 跨段語者真名推斷
├── translator.py               # /reflective-translation 反思式翻譯、雙語字幕與分批容錯模組
├── pyproject.toml              # 專案依賴與 pytest 配置
├── .env.example                # 環境變數設定範本
├── .gitignore                  # Git 忽略設定
└── tests/
    ├── test_audio_processor.py       # 切段演算法與音訊處理單元測試
    ├── test_transcript_formatter.py  # CJK 空格、斷句、字體轉換、純淨模式與字幕測試
    └── test_translator.py            # 反思式翻譯、雙語字幕、引號容錯與分批測試
```

---

## ❓ 常見問題與排錯 (FAQ)

#### Q1：為什麼上傳長影片時轉錄時間較久？
> 因為系統採取高品質的批次轉錄端點（Interactions API）並要求字詞級的時間戳與語者標記，每 20 分鐘音訊通常需要約 30~60 秒的雲端模型運算時間，請稍候進度條推進。

#### Q2：如果音訊剛好超過 30 分鐘（例如 31 分鐘），會怎麼切？
> 系統會以 20 分鐘為基準單位進行均分：`ceil(31 / 20) = 2` 段，自動均分為兩段各 15.5 分鐘，完全避開 30 分鐘上限，且絕不會切出「30 分鐘 + 1 分鐘」這種零頭散段。

#### Q3：雲端暫存檔案是否安全？
> 非常安全。音訊上傳至 Google Files API 完成轉錄後，程式會在 `finally` 區塊立即發出 DELETE 請求銷毀雲端暫存檔；本機切段檔案亦會在轉錄完成後自動刪除。

#### Q4：為什麼以前上傳超過 200MB 的影片會被擋住？
> 這是 Streamlit 網頁框架原生的預設限制（`maxUploadSize = 200`）。專案已內建 `.streamlit/config.toml` 將限制提高至 **2GB（2048 MB）**，大容量長影片已可直接拖曳上傳由後端自動剝離音訊。

#### Q5：我不需要 `[語者 1]` 標籤，如何產出乾淨字幕？
> 只要在左側邊欄將「開啟語者分離」取消勾選，系統會進入無標籤純淨模式，所有預覽與匯出之 SRT / VTT / TXT 檔案都不會出現任何語者代號。

#### Q6：遇到 429 RESOURCE_EXHAUSTED (Too Many Requests) 該如何處理？
> 這是 Google 免費版（Free Tier）對每分鐘請求次數的限制（15 RPM）。本專案已將每批翻譯容量提升至 50 句（`BATCH_SIZE = 50`），260+ 句的字幕任務只需 5~6 次請求，正常情況下不會超標。若遇到偶發頻率限制，系統會自動在前端提示並休眠 Google 要求的秒數（約 20~24 秒），冷卻結束後自動重試接續翻譯，無需手動重來。若需完全免等待，亦可於 Google AI Studio 綁定 Billing 開啟以量計費（Pay-as-you-go），RPM 上限將立即提升至 1,000+。

#### Q7：如果我已經有現成的 .srt 或 .vtt 英文字幕，一定要重新上傳音訊轉錄嗎？
> 完全不需要！您可以直接將 `.srt` 或 `.vtt` 檔案拖入上傳區，系統會自動偵測並切換為「載入現有字幕檔案並準備翻譯」，瞬間載入所有時間戳與文字，直接套用專有詞彙保護與反思式翻譯，既省時又節省轉錄額度。

---

## 🙏 參考與致謝

- [Evan Lin: [AI 實戰] Gemini 3.5 Transcribe 的兩顆模型：即時逐字稿與語者分離](https://www.evanlin.com/gemini-3.5-transcribe-live/)
- [Google AI Studio 官方文件](https://ai.google.dev/)
- [Streamlit 開發社群](https://streamlit.io/)
- [OpenCC 開源中文轉換專案](https://github.com/BYVoid/OpenCC)

---

## 📄 授權條款
本專案採用 [MIT License](LICENSE) 開源授權。
