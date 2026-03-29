#!/usr/bin/env python3
"""
SSTV Decoder — Converts an SSTV audio WAV file back to an image.

Usage:
    python3 sstv_decode.py <input.wav> <output.png> [--mode MODE]

If --mode is not specified, the decoder auto-detects the mode from the
VIS code in the calibration header.

Supported modes:
    martin1, martin2, scottie1, scottie2, scottieDX, robot36, robot72

The decoder:
    1. Reads the WAV (any sample rate — resamples to 44100 if needed)
    2. Detects the calibration header and VIS code (auto-mode detection)
    3. Locates scan line boundaries via sync pulse detection
    4. Demodulates each line's color channels from FM audio
    5. Reconstructs the image and saves as PNG
"""

import sys
import os
import argparse
import numpy as np

# Allow importing from same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sstv_common as sstv


def decode_sstv(input_path, output_path, mode_name=None,
                sample_rate=sstv.SAMPLE_RATE):
    """
    Decode an SSTV WAV file to an image.

    Parameters:
        input_path: Path to input WAV file
        output_path: Path for output image (PNG recommended)
        mode_name: SSTV mode name, or None for auto-detection
        sample_rate: Expected sample rate (WAV will be resampled if different)
    """
    print("SSTV Decoder")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print()

    # Read WAV file
    print("  Reading WAV file...")
    audio, sr = sstv.read_wav(input_path)
    duration = len(audio) / sr
    print(f"  Audio: {duration:.1f}s, {sr} Hz, {len(audio):,} samples")

    # Detect VIS code
    print("  Detecting VIS code...")
    vis_code, data_start = sstv.detect_vis_code(audio, sr)

    if vis_code >= 0 and vis_code in sstv.VIS_MAP:
        detected_mode = sstv.VIS_MAP[vis_code]
        print(f"  Detected: {detected_mode} (VIS {vis_code})")

        if mode_name is None:
            mode_name = detected_mode
        elif mode_name != detected_mode:
            print(f"  [NOTE] Overriding detected mode with: {mode_name}")
    else:
        if vis_code >= 0:
            print(f"  [WARNING] Unknown VIS code: {vis_code}")
        else:
            print("  [WARNING] Could not detect VIS code")

        if mode_name is None:
            print("  Falling back to martin1")
            mode_name = 'martin1'
            # Try to find data start from sync pulses
            data_start = 0

    if mode_name not in sstv.MODES:
        print(f"Error: Unknown mode '{mode_name}'")
        sys.exit(1)

    mode = sstv.MODES[mode_name]
    width = mode['width']
    height = mode['height']

    print(f"  Mode:       {mode_name}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Color:      {mode['color'].upper()}")
    print()

    # Compute expected samples per line
    line_samples, layout = sstv.compute_line_samples(mode, sr)
    print(f"  Expected samples per line: {line_samples}")

    # Extract image data starting after VIS code
    image_audio = audio[data_start:]

    if mode['family'] in ('martin', 'scottie'):
        img = _decode_rgb_mode(image_audio, mode, width, height,
                               line_samples, layout, sr)
    elif mode['family'] == 'robot':
        if mode['chroma_format'] == '420':
            img = _decode_robot36(image_audio, mode, width, height,
                                  line_samples, layout, sr)
        else:
            img = _decode_robot72(image_audio, mode, width, height,
                                  line_samples, layout, sr)
    else:
        print(f"Error: Unsupported family '{mode['family']}'")
        sys.exit(1)

    # Save image
    print(f"  Saving image ({width}x{height})...")
    from PIL import Image
    out_img = Image.fromarray(img, 'RGB')
    out_img.save(output_path)
    print(f"  Done! Saved to {output_path}")


def _decode_rgb_mode(audio, mode, width, height, line_samples, layout, sr):
    """
    Decode Martin or Scottie mode image data.

    Uses sync pulse detection to align line boundaries, then falls back
    to fixed timing if sync detection is unreliable.
    """
    # Strategy: Use the known line length to step through the audio.
    # For each line, try to find the sync pulse to fine-tune alignment.
    # If sync is found, use it; otherwise use fixed stepping.

    r_img = np.zeros((height, width), dtype=np.uint8)
    g_img = np.zeros((height, width), dtype=np.uint8)
    b_img = np.zeros((height, width), dtype=np.uint8)

    layout_dict = {label: (start, n) for label, start, n in layout}

    # For Scottie, the first line is special (has leading sync)
    if mode['family'] == 'scottie':
        first_line_offset = int((mode['sync_ms'] + mode['sync_porch_ms']) * sr / 1000)
    else:
        first_line_offset = 0

    pos = 0

    # Try to find the first sync pulse to get initial alignment
    if mode['family'] == 'martin':
        # Martin: sync is at the start of each line — layout starts with sync
        initial_sync = _find_first_sync(audio, sr, mode['sync_ms'])
        if initial_sync >= 0:
            pos = initial_sync
    elif mode['family'] == 'scottie':
        # Scottie layout: [Sep_lead][Green][Sep1][Blue][Sync][Porch][Red]
        # The mid-line sync (between blue and red) is the most reliable
        # anchor. Find it, then compute the line start from there.
        #
        # First line in encoder: [Sync9ms][Porch1.5ms][Green][Sep][Blue][Sync][Porch][Red]
        # The first mid-line sync is at: 9 + 1.5 + chan + sep + chan = 9 + 1.5 + 138.24 + 1.5 + 138.24 = 288.48ms
        first_mid_sync_est = int((mode['sync_ms'] + mode['sync_porch_ms'] +
                                   mode['chan_ms'] + mode['sep_ms'] +
                                   mode['chan_ms']) * sr / 1000)
        # Search for the mid-line sync near this position
        mid_sync = _find_sync_near(audio, first_mid_sync_est,
                                    sr, mode['sync_ms'], line_samples,
                                    search_window=0.3)
        if mid_sync >= 0:
            # The sync in the layout is at layout_dict['sync'][0]
            sync_offset_in_layout = layout_dict['sync'][0]
            pos = mid_sync - sync_offset_in_layout
        else:
            # Fallback: estimate from header end
            sync_porch_samples = int((mode['sync_ms'] + mode['sync_porch_ms']) * sr / 1000)
            green_start_in_layout = layout_dict['green'][0]
            pos = sync_porch_samples - green_start_in_layout
        pos = max(0, pos)

    print(f"  Starting decode at sample {pos}")
    print(f"  Decoding {height} lines...")

    for line_num in range(height):
        remaining = len(audio) - pos
        if remaining < line_samples // 2:
            print(f"  [WARNING] Audio ended at line {line_num}/{height}")
            break

        if (line_num + 1) % 32 == 0 or line_num == height - 1:
            print(f"    Line {line_num + 1}/{height}")

        # Pad with silence if the last line is truncated
        if remaining < line_samples:
            line_audio = np.concatenate([audio[pos:], np.zeros(line_samples - remaining)])
        else:
            line_audio = audio[pos:pos + line_samples]

        if mode['family'] == 'martin':
            r_row, g_row, b_row = sstv.decode_martin_line(line_audio, mode, sr)
        else:
            r_row, g_row, b_row = sstv.decode_scottie_line(line_audio, mode, sr)

        r_img[line_num, :len(r_row)] = r_row[:width]
        g_img[line_num, :len(g_row)] = g_row[:width]
        b_img[line_num, :len(b_row)] = b_row[:width]

        # Advance position
        pos += line_samples

        # Try to refine alignment by finding next sync pulse
        if pos + line_samples <= len(audio):
            sync_pos = _find_sync_near(audio, pos, sr, mode['sync_ms'],
                                       line_samples)
            if sync_pos >= 0:
                if mode['family'] == 'martin':
                    pos = sync_pos
                elif mode['family'] == 'scottie':
                    # For Scottie, sync is mid-line. The line starts at
                    # sep_lead before green, which is before the sync.
                    # The sync we find belongs to THIS line, positioned
                    # within the line. So we align based on where in the
                    # layout the sync sits.
                    sync_offset_in_line = layout_dict['sync'][0]
                    candidate = sync_pos - sync_offset_in_line
                    # Only use if it doesn't jump too far
                    if abs(candidate - pos) < line_samples // 4:
                        pos = candidate

    # Assemble RGB image
    img = np.stack([r_img, g_img, b_img], axis=2)
    return img


def _decode_robot36(audio, mode, width, height, line_samples, layout, sr):
    """Decode Robot 36 (YCbCr 4:2:0) image data."""
    y_img = np.zeros((height, width), dtype=np.float64)
    cr_img = np.zeros((height, width), dtype=np.float64)
    cb_img = np.zeros((height, width), dtype=np.float64)

    pos = 0
    initial_sync = _find_first_sync(audio, sr, mode['sync_ms'])
    if initial_sync >= 0:
        pos = initial_sync

    print(f"  Starting decode at sample {pos}")
    print(f"  Decoding {height} lines (Robot 36, YCbCr 4:2:0)...")

    # Temporary storage for chrominance (4:2:0: chroma shared between pairs)
    last_cr = np.full(width, 128.0)
    last_cb = np.full(width, 128.0)

    for line_num in range(height):
        remaining = len(audio) - pos
        if remaining < line_samples // 2:
            print(f"  [WARNING] Audio ended at line {line_num}/{height}")
            break

        if (line_num + 1) % 32 == 0 or line_num == height - 1:
            print(f"    Line {line_num + 1}/{height}")

        if remaining < line_samples:
            line_audio = np.concatenate([audio[pos:], np.zeros(line_samples - remaining)])
        else:
            line_audio = audio[pos:pos + line_samples]

        y_row, chroma_row, chroma_type = sstv.decode_robot36_line(
            line_audio, mode, line_num, sr
        )

        y_img[line_num, :len(y_row)] = y_row[:width].astype(np.float64)

        if chroma_type == 'cr':
            last_cr = chroma_row[:width].astype(np.float64)
            cr_img[line_num] = last_cr
            # Apply to previous line too (4:2:0 sharing)
            if line_num > 0:
                cr_img[line_num - 1] = last_cr
        else:
            last_cb = chroma_row[:width].astype(np.float64)
            cb_img[line_num] = last_cb
            if line_num > 0:
                cb_img[line_num - 1] = last_cb

        pos += line_samples

        # Refine alignment
        if pos + line_samples <= len(audio):
            sync_pos = _find_sync_near(audio, pos, sr, mode['sync_ms'],
                                       line_samples)
            if sync_pos >= 0:
                pos = sync_pos

    # Fill any remaining chroma gaps
    for i in range(height):
        if np.all(cr_img[i] == 0) and i > 0:
            cr_img[i] = cr_img[i - 1]
        if np.all(cb_img[i] == 0) and i > 0:
            cb_img[i] = cb_img[i - 1]

    # Convert YCbCr to RGB
    print("  Converting YCbCr to RGB...")
    r, g, b = sstv.ycbcr_to_rgb(y_img, cb_img, cr_img)
    img = np.stack([r, g, b], axis=2)
    return img


def _decode_robot72(audio, mode, width, height, line_samples, layout, sr):
    """Decode Robot 72 (YCbCr 4:2:2) image data."""
    y_img = np.zeros((height, width), dtype=np.float64)
    cr_img = np.zeros((height, width), dtype=np.float64)
    cb_img = np.zeros((height, width), dtype=np.float64)

    pos = 0
    initial_sync = _find_first_sync(audio, sr, mode['sync_ms'])
    if initial_sync >= 0:
        pos = initial_sync

    print(f"  Starting decode at sample {pos}")
    print(f"  Decoding {height} lines (Robot 72, YCbCr 4:2:2)...")

    for line_num in range(height):
        remaining = len(audio) - pos
        if remaining < line_samples // 2:
            print(f"  [WARNING] Audio ended at line {line_num}/{height}")
            break

        if (line_num + 1) % 32 == 0 or line_num == height - 1:
            print(f"    Line {line_num + 1}/{height}")

        if remaining < line_samples:
            line_audio = np.concatenate([audio[pos:], np.zeros(line_samples - remaining)])
        else:
            line_audio = audio[pos:pos + line_samples]

        y_row, cr_row, cb_row = sstv.decode_robot72_line(line_audio, mode, sr)

        y_img[line_num, :len(y_row)] = y_row[:width].astype(np.float64)
        cr_img[line_num, :len(cr_row)] = cr_row[:width].astype(np.float64)
        cb_img[line_num, :len(cb_row)] = cb_row[:width].astype(np.float64)

        pos += line_samples

        # Refine alignment
        if pos + line_samples <= len(audio):
            sync_pos = _find_sync_near(audio, pos, sr, mode['sync_ms'],
                                       line_samples)
            if sync_pos >= 0:
                pos = sync_pos

    # Convert YCbCr to RGB
    print("  Converting YCbCr to RGB...")
    r, g, b = sstv.ycbcr_to_rgb(y_img, cb_img, cr_img)
    img = np.stack([r, g, b], axis=2)
    return img


# ============================================================================
# Sync Detection Helpers
# ============================================================================

def _find_first_sync(audio, sr, sync_ms, search_limit_s=2.0):
    """
    Find the first horizontal sync pulse in the audio.

    Searches within the first search_limit_s seconds after the VIS header
    region.

    Returns:
        Sample index of the sync pulse start, or -1 if not found.
    """
    search_len = int(search_limit_s * sr)
    search_audio = audio[:min(len(audio), search_len)]

    # Compute instantaneous frequency
    inst_freq = sstv._instantaneous_frequency(search_audio, sr)

    # Smooth
    kernel_size = int(0.002 * sr)
    if kernel_size < 1:
        kernel_size = 1
    kernel = np.ones(kernel_size) / kernel_size
    smooth = np.convolve(inst_freq, kernel, mode='same')

    # Find first region near 1200 Hz with appropriate duration
    min_sync_samples = int(sync_ms * 0.5 * sr / 1000)
    max_sync_samples = int(sync_ms * 2.0 * sr / 1000)

    in_sync = False
    sync_start = 0

    for i in range(len(smooth)):
        if abs(smooth[i] - sstv.FREQ_SYNC) < 150:
            if not in_sync:
                in_sync = True
                sync_start = i
        else:
            if in_sync:
                length = i - sync_start
                if min_sync_samples <= length <= max_sync_samples:
                    return sync_start
                in_sync = False

    return -1


def _find_sync_near(audio, expected_pos, sr, sync_ms, line_samples,
                    search_window=0.15):
    """
    Find a sync pulse near the expected position.

    Searches within a window around expected_pos.

    Parameters:
        audio: Full audio array
        expected_pos: Expected sample position of sync start
        sr: Sample rate
        sync_ms: Expected sync duration in ms
        line_samples: Samples per line
        search_window: Search window as fraction of line_samples

    Returns:
        Sample index of sync start, or -1 if not found.
    """
    window = int(line_samples * search_window)
    search_start = max(0, expected_pos - window)
    search_end = min(len(audio), expected_pos + window)

    if search_end - search_start < 100:
        return -1

    segment = audio[search_start:search_end]
    inst_freq = sstv._instantaneous_frequency(segment, sr)

    # Smooth
    kernel_size = int(0.002 * sr)
    if kernel_size < 1:
        kernel_size = 1
    kernel = np.ones(kernel_size) / kernel_size
    smooth = np.convolve(inst_freq, kernel, mode='same')

    min_sync_samples = int(sync_ms * 0.5 * sr / 1000)
    max_sync_samples = int(sync_ms * 2.0 * sr / 1000)

    in_sync = False
    sync_start_local = 0
    best_sync = -1
    best_dist = float('inf')

    target = expected_pos - search_start  # Expected position within segment

    for i in range(len(smooth)):
        if abs(smooth[i] - sstv.FREQ_SYNC) < 150:
            if not in_sync:
                in_sync = True
                sync_start_local = i
        else:
            if in_sync:
                length = i - sync_start_local
                if min_sync_samples <= length <= max_sync_samples:
                    dist = abs(sync_start_local - target)
                    if dist < best_dist:
                        best_dist = dist
                        best_sync = sync_start_local + search_start
                in_sync = False

    return best_sync


def main():
    parser = argparse.ArgumentParser(
        description='SSTV Decoder — Convert SSTV audio to an image',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
The decoder auto-detects the mode from the VIS code in the calibration header.
Use --mode to override if auto-detection fails.

Supported modes:
  martin1, martin2, scottie1, scottie2, scottieDX, robot36, robot72
        """
    )
    parser.add_argument('input', help='Input SSTV WAV file')
    parser.add_argument('output', help='Output image file (PNG recommended)')
    parser.add_argument('--mode', '-m', default=None,
                        choices=list(sstv.MODES.keys()),
                        help='SSTV mode (default: auto-detect from VIS)')

    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    decode_sstv(args.input, args.output, args.mode)


if __name__ == '__main__':
    main()
