import os
import time
import logging
from typing import Dict, Any, List, Optional, Callable
import requests
from google import genai
from google.genai.errors import APIError

logger = logging.getLogger(__name__)

INTERACTIONS_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/interactions"

class GeminiTranscribeClient:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required.")
        self.api_key = api_key
        self.genai_client = genai.Client(api_key=api_key)

    def upload_file(
        self,
        file_path: str,
        mime_type: Optional[str] = "audio/wav",
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> Any:
        """
        Uploads audio file using Google Files API with explicit MIME type.
        """
        if progress_callback:
            progress_callback(f"正在上傳音訊檔案至 Google Files API: {os.path.basename(file_path)}...")
        
        logger.info(f"Uploading file {file_path} to Google Files API (mime_type={mime_type})...")
        upload_config = {"mime_type": mime_type} if mime_type else None
        uploaded = self.genai_client.files.upload(file=file_path, config=upload_config)
        logger.info(f"Uploaded file name={uploaded.name}, uri={uploaded.uri}, mime_type={uploaded.mime_type}, state={uploaded.state}")

        
        # In rare cases with large files, poll until ACTIVE if PROCESSING
        poll_count = 0
        while str(uploaded.state).upper() == "PROCESSING" and poll_count < 30:
            time.sleep(2)
            uploaded = self.genai_client.files.get(name=uploaded.name)
            poll_count += 1
            
        return uploaded

    def delete_file(self, file_name: str) -> None:
        """
        Deletes temporary file from Google Files API immediately upon completion.
        """
        try:
            logger.info(f"Deleting uploaded file from Files API: {file_name}")
            self.genai_client.files.delete(name=file_name)
        except Exception as e:
            logger.warning(f"Failed to delete file {file_name} from Files API: {e}")

    def transcribe_audio(
        self,
        file_uri: str,
        mime_type: str = "audio/wav",
        enable_diarization: bool = True,
        language_codes: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> Dict[str, Any]:
        """
        Call gemini-3.5-transcribe via Interactions API.
        Enables diarization and word timestamps.
        Note: custom_vocabulary must NOT be included in this request to avoid HTTP 400.
        """
        if progress_callback:
            progress_callback("正在呼叫 gemini-3.5-transcribe (Interactions API)...")

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key
        }

        mode_config: Dict[str, Any] = {
            "type": "verbatim",
            # Crucial: timestamp_granularities ["word"] is required to return speaker tags and word annotations!
            "timestamp_granularities": ["word"]
        }
        if enable_diarization:
            mode_config["diarization_mode"] = "speaker"

        payload = {
            "model": "gemini-3.5-transcribe",
            "input": [
                {
                    "type": "audio",
                    "uri": file_uri,
                    "mime_type": mime_type
                }
            ],
            # Note: snake_case generation_config is strictly required by Interactions API REST endpoint
            "generation_config": {
                "transcription_config": {
                    "language_codes": language_codes if language_codes else [],
                    "mode": mode_config
                }
            }
        }

        logger.info(f"Sending Interactions API request to {INTERACTIONS_ENDPOINT}")
        response = requests.post(INTERACTIONS_ENDPOINT, headers=headers, json=payload, timeout=600)
        
        if response.status_code != 200:
            err_text = response.text
            logger.error(f"Interactions API failed with status {response.status_code}: {err_text}")
            raise RuntimeError(f"Interactions API error ({response.status_code}): {err_text}")

        data = response.json()

        # If asynchronous processing is indicated, poll the interaction
        interaction_id = data.get("id")
        status = data.get("status", "completed")
        
        poll_interval = 2
        total_waited = 0
        while status in ("in_progress", "pending") and interaction_id and total_waited < 600:
            if progress_callback:
                progress_callback(f"轉錄處理中... 已等待 {total_waited} 秒")
            time.sleep(poll_interval)
            total_waited += poll_interval
            
            get_url = f"{INTERACTIONS_ENDPOINT}/{interaction_id}"
            get_resp = requests.get(get_url, headers=headers, timeout=60)
            if get_resp.status_code == 200:
                data = get_resp.json()
                status = data.get("status", "completed")
            else:
                logger.warning(f"Polling interaction {interaction_id} failed ({get_resp.status_code})")

        return data
