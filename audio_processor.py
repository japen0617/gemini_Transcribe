import os
import math
import subprocess
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any

import imageio_ffmpeg

logger = logging.getLogger(__name__)

def get_ffmpeg_executable() -> str:
    """Get the path to the ffmpeg executable bundled via imageio-ffmpeg."""
    return imageio_ffmpeg.get_ffmpeg_exe()


def get_audio_duration(file_path: str) -> float:
    """
    Get duration in seconds of an audio or video file using ffmpeg.
    """
    ffmpeg_exe = get_ffmpeg_executable()
    # Run ffmpeg -i file_path and parse duration from stderr
    cmd = [ffmpeg_exe, "-i", file_path]
    process = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    
    # Parse Duration: HH:MM:SS.xx
    duration_str = None
    for line in process.stderr.splitlines():
        if "Duration:" in line:
            parts = line.split("Duration:")[1].split(",")[0].strip()
            duration_str = parts
            break

    if not duration_str:
        raise ValueError(f"Could not determine duration for file: {file_path}")

    h, m, s = duration_str.split(":")
    total_seconds = float(h) * 3600 + float(m) * 60 + float(s)
    return total_seconds


def extract_audio_to_wav(
    input_file: str,
    output_wav: str,
    sample_rate: int = 16000,
    channels: int = 1
) -> str:
    """
    Extract or convert media file (video or audio) to 16-bit PCM WAV.
    - Sample rate: 16000 Hz
    - Channels: 1 (Mono)
    - Codec: pcm_s16le
    """
    ffmpeg_exe = get_ffmpeg_executable()
    os.makedirs(os.path.dirname(os.path.abspath(output_wav)), exist_ok=True)
    
    cmd = [
        ffmpeg_exe,
        "-y",               # Overwrite output
        "-i", input_file,   # Input file
        "-vn",              # Disable video recording
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", str(channels),
        output_wav
    ]
    
    logger.info(f"Extracting audio: {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg audio extraction failed: {result.stderr}")
        
    return output_wav


def calculate_chunk_plan(
    total_seconds: float,
    threshold_seconds: float = 1800.0,  # 30 minutes threshold
    unit_seconds: float = 1200.0        # 20 minutes unit
) -> List[Tuple[float, float]]:
    """
    Calculate chunk boundaries.
    Rule:
    - If total_seconds <= 30 minutes (1800s): Single chunk [0.0, total_seconds]
    - If total_seconds > 30 minutes:
      count = ceil(total_seconds / unit_seconds)
      Divide total_seconds evenly into count chunks so each chunk is <= 20 minutes (and strictly <= 30 min)
      without leaving tiny scraps.
    """
    if total_seconds <= threshold_seconds:
        return [(0.0, total_seconds)]
    
    chunk_count = math.ceil(total_seconds / unit_seconds)
    if chunk_count <= 1:
        return [(0.0, total_seconds)]
        
    avg_duration = total_seconds / chunk_count
    plan: List[Tuple[float, float]] = []
    
    for i in range(chunk_count):
        start = i * avg_duration
        end = total_seconds if (i == chunk_count - 1) else (i + 1) * avg_duration
        plan.append((round(start, 3), round(end, 3)))
        
    return plan


def split_wav_into_chunks(
    input_wav: str,
    chunk_plan: List[Tuple[float, float]],
    output_dir: str
) -> List[Dict[str, Any]]:
    """
    Slice the WAV file into chunks according to the chunk plan.
    Returns list of dicts: [{'index': 0, 'file_path': '...', 'start': 0.0, 'end': 1200.0, 'duration': 1200.0}]
    """
    ffmpeg_exe = get_ffmpeg_executable()
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = Path(input_wav).stem
    chunks = []
    
    # If only 1 chunk, no need to re-encode if already WAV
    if len(chunk_plan) == 1:
        duration = chunk_plan[0][1] - chunk_plan[0][0]
        return [{
            "index": 0,
            "file_path": input_wav,
            "start": chunk_plan[0][0],
            "end": chunk_plan[0][1],
            "duration": duration
        }]
        
    for i, (start, end) in enumerate(chunk_plan):
        duration = end - start
        chunk_file = os.path.join(output_dir, f"{base_name}_chunk_{i+1:02d}.wav")
        
        cmd = [
            ffmpeg_exe,
            "-y",
            "-ss", f"{start:.3f}",
            "-t", f"{duration:.3f}",
            "-i", input_wav,
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            chunk_file
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg split chunk {i+1} failed: {result.stderr}")
                
        chunks.append({
            "index": i,
            "file_path": chunk_file,
            "start": start,
            "end": end,
            "duration": duration
        })
        
    return chunks
