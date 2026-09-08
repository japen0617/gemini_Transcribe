import os
import re
import shutil
import tempfile
import logging
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

from audio_processor import (
    get_audio_duration,
    extract_audio_to_wav,
    calculate_chunk_plan,
    split_wav_into_chunks
)
from gemini_client import GeminiTranscribeClient
from transcript_formatter import (
    extract_words_from_interaction_response,
    group_words_into_subtitles,
    generate_srt,
    generate_vtt,
    generate_txt,
    generate_json_export,
    convert_subtitles_script
)

from vocabulary_corrector import (
    correct_subtitles_with_vocabulary,
    infer_and_align_speakers
)
from translator import reflective_translate_subtitles


# Load environment variables from .env
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Gemini 3.5 Transcribe 語音辨識與語者分離系統",
    page_icon="🎙️",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        color: #666;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .speaker-badge {
        display: inline-block;
        background-color: #f0f4f9;
        color: #1a73e8;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
        margin-right: 6px;
    }
    .time-badge {
        color: #888;
        font-size: 0.8rem;
        margin-right: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- Sidebar -----------------
with st.sidebar:
    st.header("⚙️ 系統設定")
    
    # API Key Handling: Priority to sidebar, fallback to .env
    env_api_key = os.getenv("GEMINI_API_KEY", "")
    api_key_input = st.text_input(
        "Gemini Developer API Key",
        value=env_api_key,
        type="password",
        help="輸入 Google AI Studio 取得的 Gemini Developer API 金鑰"
    )
    api_key = api_key_input.strip() or env_api_key

    if not api_key:
        st.warning("⚠️ 請輸入或在 .env 配置 GEMINI_API_KEY")
    else:
        st.success("✅ API Key 已就緒")

    st.divider()
    st.subheader("轉錄模型與參數")
    st.caption("核心模型：**gemini-3.5-transcribe** (Interactions API)")
    
    enable_diarization = st.checkbox(
        "開啟語者分離 (Speaker Diarization)",
        value=True,
        help="標註發言者 (spk:0, spk:1)。單次切段上限為 30 分鐘。"
    )

    lang_choice = st.selectbox(
        "語言偏好 (選填)",
        ["繁體中文 (zh-TW)", "自動偵測", "簡體中文 (zh-CN)", "英文 (en-US)", "日文 (ja-JP)"],
        index=0
    )
    lang_code_map = {
        "繁體中文 (zh-TW)": ["zh-TW"],
        "自動偵測": [],
        "簡體中文 (zh-CN)": ["zh-CN"],
        "英文 (en-US)": ["en-US"],
        "日文 (ja-JP)": ["ja-JP"]
    }
    selected_language_codes = lang_code_map[lang_choice]

    st.divider()
    st.subheader("字幕字體設定")
    script_mode_map = {
        "繁體中文 (台灣習慣用語 s2twp)": "s2twp",
        "繁體中文 (台灣標準字體 s2tw)": "s2tw",
        "繁體中文 (標準繁體 s2t)": "s2t",
        "簡體中文 (t2s)": "t2s",
        "保持模型原始輸出": "none"
    }
    selected_script_label = st.selectbox(
        "字幕輸出字體",
        list(script_mode_map.keys()),
        index=0,
        help="自動將轉錄文字進行繁簡字體轉換，預設為台灣繁體中文（含常用語轉換）"
    )
    selected_script_mode = script_mode_map[selected_script_label]

    st.divider()
    st.subheader("🌐 反思式翻譯 (Reflective Translation)")
    auto_translate = st.checkbox(
        "若非正體中文則自動反思翻譯",
        value=True,
        help="依據 /reflective-translation 兩階段自審規範，自動將英文、日文等外語字幕翻譯為台灣繁體正體"
    )

    st.divider()
    st.subheader("專有詞彙與語者推斷模型")

    
    llm_models_map = {
        "Gemini 3.5 Flash Lite": "gemini-3.5-flash-lite",
        "Gemini 3.5 Flash": "gemini-3.5-flash",
        "Gemini 3.6 Flash": "gemini-3.6-flash",
        "Gemini 3.7 Flash": "gemini-3.7-flash",
        "Gemini 3.8 Flash": "gemini-3.8-flash"
    }
    
    selected_llm_label = st.selectbox(
        "選擇校正與推斷模型",
        list(llm_models_map.keys()),
        index=0,
        help="預設為 Gemini 3.5 Flash Lite，用於專有詞彙同音字校正與對話線索跨段語者姓名推斷"
    )
    selected_llm_model = llm_models_map[selected_llm_label]

    st.divider()
    st.subheader("斷句與切分設定")

    pause_threshold = st.slider(
        "語音停頓斷句閾值 (秒)",
        min_value=0.5,
        max_value=3.0,
        value=1.2,
        step=0.1,
        help="當同一語者停頓超過此秒數時，自動斷開為下一條字幕"
    )
    max_chars = st.slider(
        "單行字幕最大字數",
        min_value=15,
        max_value=60,
        value=30,
        step=1,
        help="單條字幕建議字數，超過時於自然語意處換行切分"
    )
    cleanup_temp = st.checkbox("轉錄成功後清理暫存檔", value=True)

# ----------------- Main Area -----------------
st.markdown('<div class="main-title">🎙️ Gemini 3.5 Transcribe 影音辨識與語者分離系統</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">支援長影音自動剝離與無零頭分段、字詞時間戳、語者分離標籤、專有詞彙校正與多格式字幕匯出</div>', unsafe_allow_html=True)

# Upload Area
uploaded_file = st.file_uploader(
    "選擇或拖曳影音檔案至此（支援 MP4, MOV, MKV, AVI, WEBM, WAV, MP3, M4A, FLAC, OGG）",
    type=["mp4", "mov", "mkv", "avi", "webm", "wav", "mp3", "m4a", "aac", "flac", "ogg"]
)

col1, col2 = st.columns([1, 1])
with col1:
    vocab_text = st.text_area(
        "專有詞彙與術語清單（同時適用於轉錄校正與反思翻譯）",
        placeholder="例如：\nswitch, AP, Extreme Switching\nnetwork -> 網路（或 network 翻譯成網路）\n“Fabric” , “Wing” 保留原文\nAnthropic, PyTorch, 永豐金",
        help="保留原文：直接輸入英文詞彙（如 switch, AP）或加上「保留原文」，翻譯時將強制鎖定保留英文，絕不誤翻為中文。\n指定譯法：可輸入符號對照（network -> 網路）或自然語言（network 翻譯成網路），翻譯時將嚴格採用此譯名。",
        height=140
    )
    custom_vocab = [v.strip() for v in vocab_text.splitlines() if v.strip()]

with col2:
    context_notes = st.text_area(
        "會議筆記 / 與會者備忘（選填，供 AI 人名推斷）",
        placeholder="例如：與會人員：Evan (主持人), Samantha (前端主管), Alex (架構師)...",
        help="輸入會議背景或可能之發言者名單，便於後續 AI 比對對話線索推斷語者真名"
    )

# Session state initialization
if "transcription_results" not in st.session_state:
    st.session_state.transcription_results = None
if "speaker_aligned" not in st.session_state:
    st.session_state.speaker_aligned = False
if "speaker_report" not in st.session_state:
    st.session_state.speaker_report = None

start_btn = st.button("🚀 開始音訊處理與語音轉錄", type="primary", use_container_width=True)

if start_btn:
    if not api_key:
        st.error("❌ 請先提供 Gemini API Key！")
        st.stop()
    if not uploaded_file:
        st.error("❌ 請先上傳影音檔案！")
        st.stop()

    work_dir = tempfile.mkdtemp(prefix="transcribe_")
    try:
        # Save uploaded file
        input_filename = uploaded_file.name
        input_path = os.path.join(work_dir, input_filename)
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        progress_bar = st.progress(0)
        status_box = st.empty()

        # Step 1: Extract audio to 16kHz PCM WAV
        status_box.info("步驟 1/5：檢查檔案類型並剝離/轉碼為 16kHz 16-bit 單聲道 WAV 音訊...")
        progress_bar.progress(10)
        
        extracted_wav = os.path.join(work_dir, "extracted_audio.wav")
        extract_audio_to_wav(input_path, extracted_wav)
        total_duration = get_audio_duration(extracted_wav)

        status_box.info(f"步驟 2/5：音訊總長度 {total_duration/60:.2f} 分鐘 ({total_duration:.1f} 秒)，正在進行切段規劃...")
        progress_bar.progress(25)

        # Step 2: Calculate chunk plan
        chunk_plan = calculate_chunk_plan(total_duration)
        chunk_count = len(chunk_plan)
        chunks = split_wav_into_chunks(extracted_wav, chunk_plan, os.path.join(work_dir, "chunks"))
        
        status_box.info(f"步驟 2/5：音訊已均分為 {chunk_count} 個區段（每段 <= 20 分鐘，無零頭散段）")
        progress_bar.progress(35)

        # Step 3: Transcribe each chunk via Gemini Interactions API
        client = GeminiTranscribeClient(api_key=api_key)
        all_words = []
        uploaded_cloud_files = []

        for idx, chunk_info in enumerate(chunks):
            status_box.info(f"步驟 3/5：正在轉錄第 {idx + 1} / {chunk_count} 段音訊 ({chunk_info['duration']/60:.1f} 分鐘)...")
            progress_bar.progress(35 + int(35 * (idx / chunk_count)))

            # Upload to Google Files API with explicit audio/wav MIME type
            cloud_file = client.upload_file(chunk_info["file_path"], mime_type="audio/wav")
            uploaded_cloud_files.append(cloud_file.name)

            # Call gemini-3.5-transcribe Interactions API (strictly match cloud_file.mime_type)
            actual_mime = getattr(cloud_file, "mime_type", None) or "audio/wav"
            response_data = client.transcribe_audio(
                file_uri=cloud_file.uri,
                mime_type=actual_mime,
                enable_diarization=enable_diarization,
                language_codes=selected_language_codes,
                progress_callback=lambda msg: status_box.info(f"第 {idx + 1}/{chunk_count} 段：{msg}")
            )


            # Extract words
            chunk_words, fallback_text, has_diarization = extract_words_from_interaction_response(
                response_data,
                chunk_index=idx,
                chunk_start_sec=chunk_info["start"],
                chunk_count=chunk_count,
                enable_diarization=enable_diarization
            )
            all_words.extend(chunk_words)

        progress_bar.progress(75)

        # Step 4: Group words into subtitles
        status_box.info("步驟 4/5：進行字詞智慧斷句與語者時間戳整合...")
        subtitles = group_words_into_subtitles(
            all_words,
            pause_threshold=pause_threshold,
            max_chars=max_chars
        )

        progress_bar.progress(85)

        # Step 5: Custom Vocabulary Correction via Gemini Flash (if provided)
        if custom_vocab:
            status_box.info(f"步驟 5/5：正在利用 {selected_llm_label} 比對專有詞彙進行同音字精準校正...")
            subtitles = correct_subtitles_with_vocabulary(
                api_key=api_key,
                subtitles=subtitles,
                custom_vocabulary=custom_vocab,
                model_name=selected_llm_model
            )

        # Step 6: Convert subtitle script (Traditional / Simplified)
        if selected_script_mode != "none":
            status_box.info(f"轉換字幕字體：{selected_script_label}...")
            subtitles = convert_subtitles_script(subtitles, mode=selected_script_mode)

        # Step 7: Reflective Translation if non-Chinese and enabled
        translated_subs = None
        bilingual_subs = None
        reflection_notes = []

        sample_text = " ".join([s.get("text", "") for s in subtitles[:6]])
        cjk_count = sum(1 for c in sample_text if '\u4e00' <= c <= '\u9fff')
        ascii_count = sum(1 for c in sample_text if c.isascii() and c.isalpha())
        is_foreign = (ascii_count > cjk_count * 2) or (lang_choice in ["英文 (en-US)", "日文 (ja-JP)"])

        if auto_translate and is_foreign:
            def _update_trans_status(cur, total):
                status_box.info(f"步驟 7：偵測到非中文語音，正在以 {selected_llm_label} 執行反思式兩階段翻譯（進度：{cur}/{total} 句）...")
            status_box.info(f"步驟 7：偵測到非中文語音，正在以 {selected_llm_label} 執行反思式兩階段翻譯...")
            translated_subs, bilingual_subs, reflection_notes = reflective_translate_subtitles(
                api_key=api_key,
                subtitles=subtitles,
                custom_vocabulary=custom_vocab,
                model_name=selected_llm_model,
                progress_callback=_update_trans_status
            )

        progress_bar.progress(100)
        status_box.success("🎉 轉錄與字幕產出完成！" + ("（已完成反思翻譯）" if translated_subs else ""))

        # Cleanup cloud files immediately
        if cleanup_temp:
            for c_name in uploaded_cloud_files:
                client.delete_file(c_name)

        # Store in session state
        st.session_state.transcription_results = {
            "subtitles": translated_subs if translated_subs else subtitles,
            "raw_subtitles": subtitles,
            "translated_subtitles": translated_subs,
            "bilingual_subtitles": bilingual_subs,
            "reflection_notes": reflection_notes,
            "total_duration": total_duration,
            "chunk_count": chunk_count,
            "filename": input_filename
        }

        st.session_state.speaker_aligned = False
        st.session_state.speaker_report = None


    except Exception as e:
        st.error(f"❌ 處理過程發生錯誤：{str(e)}")
        logger.exception("Transcription pipeline failed")
    finally:
        if cleanup_temp and os.path.exists(work_dir):
            try:
                shutil.rmtree(work_dir)
            except Exception:
                pass

# ----------------- Results Display -----------------
if st.session_state.transcription_results:
    res = st.session_state.transcription_results
    subtitles = res["subtitles"]
    total_dur = res["total_duration"]
    chunk_count = res["chunk_count"]

    st.divider()
    st.subheader("📊 轉錄成果與統計")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("總音訊時長", f"{total_dur/60:.2f} 分鐘")
    m2.metric("切分段數", f"{chunk_count} 段")
    m3.metric("總字幕句數", f"{len(subtitles)} 句")
    distinct_speakers_list = [s.get("speaker", "").strip() for s in subtitles if s.get("speaker", "").strip()]
    distinct_speakers_count = len(set(distinct_speakers_list))
    if distinct_speakers_count > 0:
        m4.metric("識別語者數", f"{distinct_speakers_count} 位")
    else:
        m4.metric("語者分離", "未開啟")

    # AI Speaker Reconcile & Name Inference Button (Only show if diarization is enabled & speakers exist)
    if distinct_speakers_count > 0:
        st.write("---")
        r_col1, r_col2 = st.columns([2, 1])
        with r_col1:
            st.write("💡 **AI 跨段語者對齊與人名推斷**")
            st.caption("依據對話線索（自我介紹、指名稱謂）統一跨段語者代號並標註真實姓名（若無直接依據則忠實保留代號）")
        with r_col2:
            if st.button("🤖 執行跨段對齊與人名推斷", use_container_width=True):
                with st.spinner(f"{selected_llm_label} 正在分析對話語意與稱謂證據..."):
                    aligned_subs, rep = infer_and_align_speakers(
                        api_key=api_key,
                        subtitles=subtitles,
                        user_notes=context_notes,
                        model_name=selected_llm_model
                    )

                    if selected_script_mode != "none":
                        aligned_subs = convert_subtitles_script(aligned_subs, mode=selected_script_mode)

                    st.session_state.transcription_results["subtitles"] = aligned_subs
                    st.session_state.speaker_aligned = True
                    st.session_state.speaker_report = rep
                    st.rerun()


        if st.session_state.speaker_report and "speakers" in st.session_state.speaker_report:
            with st.expander("🔍 檢視 AI 語者推斷依據報告", expanded=True):
                for spk in st.session_state.speaker_report["speakers"]:
                    label = spk.get("label")
                    name = spk.get("name")
                    evidence = spk.get("evidence")
                    if name:
                        st.markdown(f"- **{name}** (`{label}`) — 依據：{evidence}")
                    else:
                        st.markdown(f"- `{label}` — 對話中未發現直接姓名線索，保留代號")

    # AI Reflective Translation Action
    st.write("---")
    tr_col1, tr_col2 = st.columns([2, 1])
    with tr_col1:
        st.write("🌐 **反思式翻譯為正體中文 (Reflective Translation)**")
        st.caption("依據 /reflective-translation 兩階段規範進行口語潤飾、術語保護與自審反思筆記")
    with tr_col2:
        if st.button("🌐 執行反思式翻譯", use_container_width=True):
            with st.spinner(f"{selected_llm_label} 正在執行反思式翻譯與術語審查..."):
                t_subs, b_subs, r_notes = reflective_translate_subtitles(
                    api_key=api_key,
                    subtitles=res.get("raw_subtitles", subtitles),
                    custom_vocabulary=custom_vocab,
                    model_name=selected_llm_model
                )
                st.session_state.transcription_results["translated_subtitles"] = t_subs
                st.session_state.transcription_results["bilingual_subtitles"] = b_subs
                st.session_state.transcription_results["reflection_notes"] = r_notes
                st.session_state.transcription_results["subtitles"] = t_subs
                st.rerun()

    # Reflection Notes Expander
    if res.get("reflection_notes"):
        with st.expander("💡 檢視反思式翻譯與術語抉擇筆記 (Reflection Notes)", expanded=True):
            for note in res["reflection_notes"]:
                st.markdown(f"- {note}")

    # Subtitle Display Mode Switcher
    if res.get("translated_subtitles"):
        sub_mode = st.radio(
            "選擇字幕預覽與下載模式",
            ["正體中文譯文", "雙語對照字幕 (繁體中文 + 原文)", "原始轉錄文字"],
            index=0,
            horizontal=True
        )
        if sub_mode == "正體中文譯文":
            active_subtitles = res["translated_subtitles"]
        elif sub_mode == "雙語對照字幕 (繁體中文 + 原文)":
            active_subtitles = res["bilingual_subtitles"]
        else:
            active_subtitles = res.get("raw_subtitles", subtitles)
    else:
        active_subtitles = subtitles

    # Subtitle Preview
    st.subheader("📝 字幕即時預覽")
    with st.container(height=380):
        for sub in active_subtitles:
            spk_label = sub.get("speaker", "").strip()
            start_fmt = f"{sub['start']:.1f}s"
            end_fmt = f"{sub['end']:.1f}s"
            rendered_text = sub["text"].replace("\n", "<br>")
            spk_badge = f'<span class="speaker-badge">[{spk_label}]</span> ' if spk_label else ""
            st.markdown(
                f'<span class="time-badge">[{start_fmt} - {end_fmt}]</span>'
                f'{spk_badge}'
                f'{rendered_text}',
                unsafe_allow_html=True
            )

    # Download Buttons
    st.subheader("📥 匯出字幕與逐字稿")
    base_stem = Path(res["filename"]).stem
    
    srt_content = generate_srt(active_subtitles)
    vtt_content = generate_vtt(active_subtitles)
    txt_content = generate_txt(active_subtitles)
    json_content = generate_json_export(active_subtitles, metadata={"filename": res["filename"], "duration": total_dur})


    d1, d2, d3, d4 = st.columns(4)
    d1.download_button(
        label="下載 SRT 字幕檔",
        data=srt_content,
        file_name=f"{base_stem}.srt",
        mime="application/x-subrip",
        use_container_width=True
    )
    d2.download_button(
        label="下載 VTT 字幕檔",
        data=vtt_content,
        file_name=f"{base_stem}.vtt",
        mime="text/vtt",
        use_container_width=True
    )
    d3.download_button(
        label="下載 TXT 逐字稿",
        data=txt_content,
        file_name=f"{base_stem}.txt",
        mime="text/plain",
        use_container_width=True
    )
    d4.download_button(
        label="下載 JSON 完整結構",
        data=json_content,
        file_name=f"{base_stem}.json",
        mime="application/json",
        use_container_width=True
    )
