import os
import wave
import struct
import tempfile
import pytest
from audio_processor import (
    calculate_chunk_plan,
    extract_audio_to_wav,
    get_audio_duration
)

def test_calculate_chunk_plan_short():
    # <= 30 minutes (1800s): Single chunk
    plan_10m = calculate_chunk_plan(600.0)
    assert len(plan_10m) == 1
    assert plan_10m[0] == (0.0, 600.0)

    plan_30m = calculate_chunk_plan(1800.0)
    assert len(plan_30m) == 1
    assert plan_30m[0] == (0.0, 1800.0)

def test_calculate_chunk_plan_long():
    # 35 minutes (2100s) -> ceil(2100 / 1200) = 2 chunks
    plan_35m = calculate_chunk_plan(2100.0)
    assert len(plan_35m) == 2
    # Each chunk should be 2100 / 2 = 1050s
    assert plan_35m[0] == (0.0, 1050.0)
    assert plan_35m[1] == (1050.0, 2100.0)

    # 60m 5s (3605s) -> ceil(3605 / 1200) = 4 chunks
    plan_60m = calculate_chunk_plan(3605.0)
    assert len(plan_60m) == 4
    # All chunks must be <= 1200s (20 mins)
    for start, end in plan_60m:
        duration = end - start
        assert duration <= 1200.0
    assert plan_60m[0][0] == 0.0
    assert plan_60m[-1][1] == 3605.0

def test_extract_audio_and_duration():
    # Create a 2-second synthetic WAV file (44100Hz stereo)
    with tempfile.TemporaryDirectory() as tmpdir:
        input_wav = os.path.join(tmpdir, "input.wav")
        output_wav = os.path.join(tmpdir, "output_16k_mono.wav")
        
        sample_rate = 44100
        duration_sec = 2
        total_samples = sample_rate * duration_sec
        
        with wave.open(input_wav, "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            # Write 2 seconds of silence/sine
            data = struct.pack(f"<{total_samples * 2}h", *([0] * (total_samples * 2)))
            wf.writeframes(data)
            
        # Extract and convert to 16kHz mono
        extract_audio_to_wav(input_wav, output_wav)
        assert os.path.exists(output_wav)
        
        # Verify with wave module
        with wave.open(output_wav, "rb") as wf:
            assert wf.getframerate() == 16000
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            
        dur = get_audio_duration(output_wav)
        assert abs(dur - 2.0) < 0.2


def test_calculate_audio_rms_and_slice():
    from audio_processor import calculate_audio_rms, slice_wav_segment

    with tempfile.TemporaryDirectory() as tmpdir:
        test_wav = os.path.join(tmpdir, "rms_test.wav")
        sample_rate = 16000
        sec1_samples = sample_rate * 1
        sec2_samples = sample_rate * 1

        with wave.open(test_wav, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            # 1 second of silence (0), 1 second of signal (amplitude 3000)
            data1 = struct.pack(f"<{sec1_samples}h", *([0] * sec1_samples))
            data2 = struct.pack(f"<{sec2_samples}h", *([3000] * sec2_samples))
            wf.writeframes(data1 + data2)

        # Test silence RMS
        silence_rms = calculate_audio_rms(test_wav, 0.0, 1.0)
        assert silence_rms < 1.0

        # Test voice RMS
        voice_rms = calculate_audio_rms(test_wav, 1.0, 2.0)
        assert abs(voice_rms - 3000.0) < 10.0

        # Test slice
        slice_out = os.path.join(tmpdir, "slice.wav")
        slice_wav_segment(test_wav, 1.0, 1.0, slice_out)
        assert os.path.exists(slice_out)
        dur = get_audio_duration(slice_out)
        assert abs(dur - 1.0) < 0.2

