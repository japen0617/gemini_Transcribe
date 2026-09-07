# Gemini 3.5 Transcribe 影音語音辨識與語者分離系統 🎙️

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Model](https://img.shields.io/badge/ASR-gemini--3.5--transcribe-8E75B2.svg)](https://ai.google.dev/)
[![LLM](https://img.shields.io/badge/LLM-Gemini_3.5_Flash_Lite-4285F4.svg)](https://ai.google.dev/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

基於 Google **Gemini 3.5 Transcribe** 模型的端到端影音辨識與多格式字幕產生系統。支援長影音自動剝離音訊、動態均分切段、語者分離（Speaker Diarization）、專有詞彙同音字校正、台灣繁體中文自動轉換，以及一鍵匯出 **SRT / VTT / TXT / JSON** 四種格式字幕。

---

## 📑 目錄
- [🌟 核心特色](#-核心特色)
- [🧩 系統架構流程](#-系統架構流程)
- [🛠️ 核心避坑防護設計](#️-核心避坑防護設計)
- [🚀 快速開始](#-快速開始)
  - [環境準備](#1-環境準備)
  - [安裝依賴](#2-安裝依賴)
  - [設定-api-key](#3-設定-api-key)
  - [啟動-web-介面](#4-啟動-web-介面)
- [🖥️ 介面功能詳解](#️-介面功能詳解)
- [🧪 執行單元測試](#-執行單元測試)
- [📂 專案架構](#-專案架構)
- [❓ 常見問題與排錯-faq](#-常見問題與排錯-faq)
- [🙏 參考與致謝](#-參考與致謝)

---

## 🌟 核心特色

1. **影音雙棲支援**：支援 MP4, MOV, MKV, AVI, WEBM, WAV, MP3, M4A, FLAC, OGG。上傳影片自動提取為 16kHz 單聲道 16-bit PCM WAV。
2. **長音訊無零頭動態均分**：
   - 音訊超過 30 分鐘時，採動態平均切分：`count = ceil(總時長 / 1200秒)`。
   - 每段時長均等且皆小於 20 分鐘，既避開 API 30 分鐘上限，又杜絕產生數秒鐘的極短零頭段落。
3. **高精度語者分離 (Speaker Diarization)**：
   - 透過 `gemini-3.5-transcribe` Interactions API 識別不同發言者（`語者 1`, `語者 2`...）。
   - 跨段落自動加上命名空間（如 `[第1段-語者1]`、`[第2段-語者1]`），避免錯誤假設不同段落的同名代號為同一人。
4. **專有詞彙 LLM 兩階段校正**：
   - 突破官方 API 限制（原生 `custom_vocabulary` 與語者分離互斥）。
   - 轉錄後交由 Gemini Flash 進行專有名詞、同音字精準校準，完全不影響時間戳與語者標記。
5. **網頁即時切換五款 LLM 模型**：
   - 側邊欄隨選校正與推斷模型，預設 **Gemini 3.5 Flash Lite**，並提供 **Gemini 3.5 Flash**、**Gemini 3.6 Flash**、**Gemini 3.7 Flash**、**Gemini 3.8 Flash**。
6. **台灣繁體中文 (OpenCC s2twp) 自動轉換**：
   - 解決 Google 語音模型原生偏向輸出簡體字的問題。
   - 預設自動進行台灣用語與標準正體轉換（如「軟件」->「軟體」、「項目」->「專案」）。
7. **AI 跨段語者統一與真名推斷**：
   - 由 Gemini 分析對話中的自我介紹（如「我是 Sarah」）或稱謂證據（如「Evan 你進度如何」）自動對齊跨段代號並標記真實姓名（如 `[Evan (語者 1)]`）；無確鑿證據則忠實保留代號，嚴防幻覺。
8. **四大格式一鍵匯出**：
   - **SRT**：標準影音播放器字幕。
   - **VTT**：網頁 HTML5 `<video>` 播放字幕。
   - **TXT**：發言者標註之純文字會議逐字稿。
   - **JSON**：含字詞層級時間戳（`word_info`）之完整結構化數據。
9. **隱私與安全**：
   - 轉錄完成後立即刪除 Google Files API 雲端副本與本機暫存檔案。

---

## 🧩 系統架構流程

```mermaid
graph TD
    A[使用者上傳影音檔案] --> B[audio_processor.py]
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
    K --> M[智慧自然斷句與字幕分行]
    M --> N[vocabulary_corrector.py<br/>Gemini Flash 專有詞彙校正]
    N --> O[OpenCC 繁體中文轉換 s2twp]
    O --> P[Streamlit Web 介面即時預覽]
    P -->|可選| Q[AI 跨段語者對齊與人名推斷]
    P --> R[匯出 SRT / VTT / TXT / JSON]
```

---

## 🛠️ 核心避坑防護設計

| 潛在問題 / API 陷阱 | 官方行為 | 本專案解決方案 |
| :--- | :--- | :--- |
| **語者分離無標註** | 缺少 `timestamp_granularities: ["word"]` 時，API 回傳 200 但完全不帶語者代號 | 強制在請求體中帶入 `timestamp_granularities: ["word"]` |
| **專有詞彙與語者互斥** | 在 Interactions API 同時傳入 `custom_vocabulary` 與語者分離會直接報錯 HTTP 400 | 改由兩階段管線處理：取得轉錄結果後，交由 Gemini Flash 進行二次同音詞對齊校正 |
| **macOS MIME 型態錯誤** | macOS 內建推斷 WAV 為 `audio/x-wav`，與 API payload 的 `audio/wav` 不符報錯 400 | 上傳時顯式聲明 `audio/wav`，並動態抓取雲端資源 MIME 嚴格對齊傳入 |
| **切段零頭散段** | 60 分 5 秒若切 30m+30m+5s，尾段浪費配額且辨識差 | 採均分演算法 `ceil(3605/1200) = 4` 段，每段 15 分鐘均勻分配 |
| **跨段語者代號衝突** | 第 2 段的 `spk:0` 與第 1 段的 `spk:0` 無關，直連會造成誤讀 | 加入段落命名空間 `第1段-語者1`，並提供 AI 依語境推斷對齊 |
| **中文單字空格陷阱** | 字詞標註連綴時英文需空格，中文會變成「你 好 世 界」 | 實作 CJK 字元判定與標點符號自動貼合演算法 |
| **預設輸出簡體中文** | Google 語音模型訓練基底多為簡體中文 | 內建 OpenCC 台灣習慣用語轉換器（`s2twp`），預設自動轉繁體 |

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
   - **開啟語者分離**：勾選以識別不同說話者。
   - **語言偏好**：預設 `繁體中文 (zh-TW)`，支援自動偵測、英文、日文等。
   - **字幕輸出字體**：預設 `繁體中文 (台灣習慣用語 s2twp)`，提供標準繁體、簡體或原始輸出。
   - **專有詞彙校正模型**：可自由挑選 `Gemini 3.5 Flash Lite` (預設)、`3.5 Flash`、`3.6 Flash`、`3.7 Flash`、`3.8 Flash`。
   - **斷句參數微調**：停頓間隔秒數門檻（預設 1.2 秒）與單行最大字數（預設 30 字）。
2. **主操作區**：
   - **拖曳上傳**：支援各類影音格式。
   - **專有詞彙輸入**：以逗號或換行輸入自訂關鍵字（如：`Anthropic, PyTorch, 永豐金, 抗重力`）。
   - **會議筆記**：可輸入參與者名單備忘，提供 AI 更充份的推斷線索。
   - **轉錄與進度條**：多階段進度狀態顯示。
   - **AI 跨段對齊按鈕**：點擊「🤖 執行跨段對齊與人名推斷」，可查看推斷依據報告。
   - **字幕即時預覽與四大下載按鈕**：直接下載 `.srt`, `.vtt`, `.txt`, `.json`。

---

## 🧪 執行單元測試

本專案附帶完整的自動化單元測試，涵蓋動態切段、音訊提取、CJK 空格過濾、時間戳計算與字幕格式化：
```bash
uv run pytest -v
```

測試涵蓋重點：
- `test_calculate_chunk_plan_short`：驗證短音訊不切段。
- `test_calculate_chunk_plan_long`：驗證 35 分鐘均分 2 段、60 分 5 秒均分 4 段且皆 <= 20 分鐘。
- `test_extract_audio_and_duration`：驗證音訊抽取與採樣率 16000Hz 轉換。
- `test_cjk_detection_and_space`：驗證中文不空格、英文正確空格。
- `test_timestamp_formatting`：驗證 SRT/VTT 毫秒與逗號/點號格式。
- `test_convert_subtitles_script`：驗證 OpenCC 台灣常用語繁體字轉換。

---

## 📂 專案架構

```
gemini_Transcribe/
├── app.py                      # Streamlit 前端網頁應用
├── audio_processor.py          # 影片音訊剝離、時長偵測與動態均分切段模組
├── gemini_client.py            # Gemini Interactions API 與 Files API 客戶端
├── transcript_formatter.py     # 字詞時間戳解析、CJK 空格處理、OpenCC 轉換與字幕匯出
├── vocabulary_corrector.py     # 專有詞彙同音字校正與 AI 跨段語者真名推斷
├── pyproject.toml              # 專案依賴與 pytest 配置
├── .env.example                # 環境變數設定範本
├── .gitignore                  # Git 忽略設定
└── tests/
    ├── test_audio_processor.py       # 切段演算法與音訊處理單元測試
    └── test_transcript_formatter.py  # CJK 空格、斷句、字體轉換與字幕測試
```

---

## ❓ 常見問題與排錯 (FAQ)

#### Q1：為什麼上傳長影片時轉錄時間較久？
> 因為系統採取高品質的批次轉錄端點（Interactions API）並要求字詞級的時間戳與語者標記，每 20 分鐘音訊通常需要約 30~60 秒的雲端模型運算時間，請稍候進度條推進。

#### Q2：如果音訊剛好超過 30 分鐘（例如 31 分鐘），會怎麼切？
> 系統會以 20 分鐘為基準單位進行均分：`ceil(31 / 20) = 2` 段，自動均分為兩段各 15.5 分鐘，完全避開 30 分鐘上限，且絕不會切出「30 分鐘 + 1 分鐘」這種零頭散段。

#### Q3：雲端暫存檔案是否安全？
> 非常安全。音訊上傳至 Google Files API 完成轉錄後，程式會在 `finally` 區塊立即發出 DELETE 請求銷毀雲端暫存檔；本機切段檔案亦會在轉錄完成後自動刪除。

---

## 🙏 參考與致謝

- [Evan Lin: [AI 實戰] Gemini 3.5 Transcribe 的兩顆模型：即時逐字稿與語者分離](https://www.evanlin.com/gemini-3.5-transcribe-live/)
- [Google AI Studio 官方文件](https://ai.google.dev/)
- [Streamlit 開發社群](https://streamlit.io/)
- [OpenCC 開源中文轉換專案](https://github.com/BYVoid/OpenCC)

---

## 📄 授權條款
本專案採用 [MIT License](LICENSE) 開源授權。
