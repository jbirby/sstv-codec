#!/usr/bin/env python3
"""
Shared module for SSTV (Slow-Scan Television) encoding/decoding.

Contains:
  - Mode definitions with exact timing constants (Martin, Scottie, Robot)
  - Calibration header generator/detector (leader + VIS code)
  - Frequency modulation (pixel value -> audio frequency)
  - Frequency demodulation (audio -> pixel values)
  - Sync pulse generation and detection
  - Color space conversion (RGB <-> YCbCr for Robot modes)
"""

import numpy as np

# ============================================================================
# Constants
# ============================================================================

SAMPLE_RATE = 44100      # CD-quality audio sample rate (Hz)

# SSTV frequency mapping (Hz)
FREQ_BLACK  = 1500       # Black level
FREQ_WHITE  = 2300       # White level
FREQ_RANGE  = 800        # WHITE - BLACK
FREQ_SYNC   = 1200       # Sync pulse frequency
FREQ_VIS_START = 1900    # Leader / calibration tone
FREQ_VIS_BIT1 = 1100     # VIS data bit = 1
FREQ_VIS_BIT0 = 1300     # VIS data bit = 0

# Header timing (seconds)
HDR_LEADER_MS   = 300.0   # Leader tone duration (ms)
HDR_BREAK_MS    = 10.0    # Break pulse duration (ms)
HDR_VIS_BIT_MS  = 30.0    # Each VIS bit duration (ms)

# ============================================================================
# Mode Definitions
# ============================================================================
# Each mode is a dict with:
#   vis_code    : 7-bit VIS code (int)
#   width       : Image width in pixels
#   height      : Image height in pixels (scan lines)
#   color       : 'rgb' or 'ycbcr'
#   family      : 'martin', 'scottie', or 'robot'
#   sync_ms     : Horizontal sync pulse duration (ms)
#   sync_porch_ms: Sync porch duration (ms)
#   sep_ms      : Separator pulse duration (ms)
#   chan_ms      : Per-channel scan time (ms) -- for RGB modes
#   For Robot modes, additional keys:
#     y_scan_ms  : Luminance scan time per line (ms)
#     cr_scan_ms : Cr (R-Y) chrominance scan time per line (ms)
#     cb_scan_ms : Cb (B-Y) chrominance scan time per line (ms)
#     porch_ms   : Porch after sync (ms)
#     sep_porch_ms: Separator porch (ms)
#     chroma_format: '420' or '422'

MODES = {
    # ------------------------------------------------------------------
    # Martin modes: sync at line START, color order G-B-R
    # ------------------------------------------------------------------
    'martin1': {
        'vis_code': 44,
        'width': 320,
        'height': 256,
        'color': 'rgb',
        'family': 'martin',
        'color_order': 'gbr',
        'sync_ms': 4.862,
        'sync_porch_ms': 0.572,
        'sep_ms': 0.572,
        'chan_ms': 146.432,
    },
    'martin2': {
        'vis_code': 40,
        'width': 320,
        'height': 256,
        'color': 'rgb',
        'family': 'martin',
        'color_order': 'gbr',
        'sync_ms': 4.862,
        'sync_porch_ms': 0.572,
        'sep_ms': 0.572,
        'chan_ms': 73.216,
    },

    # ------------------------------------------------------------------
    # Scottie modes: sync in MIDDLE of line (between blue and red),
    #                color order G-B-R
    # ------------------------------------------------------------------
    'scottie1': {
        'vis_code': 60,
        'width': 320,
        'height': 256,
        'color': 'rgb',
        'family': 'scottie',
        'color_order': 'gbr',
        'sync_ms': 9.0,
        'sync_porch_ms': 1.5,
        'sep_ms': 1.5,
        'chan_ms': 138.240,
    },
    'scottie2': {
        'vis_code': 56,
        'width': 320,
        'height': 256,
        'color': 'rgb',
        'family': 'scottie',
        'color_order': 'gbr',
        'sync_ms': 9.0,
        'sync_porch_ms': 1.5,
        'sep_ms': 1.5,
        'chan_ms': 88.064,
    },
    'scottieDX': {
        'vis_code': 76,
        'width': 320,
        'height': 256,
        'color': 'rgb',
        'family': 'scottie',
        'color_order': 'gbr',
        'sync_ms': 9.0,
        'sync_porch_ms': 1.5,
        'sep_ms': 1.5,
        'chan_ms': 345.600,
    },

    # ------------------------------------------------------------------
    # Robot modes: YCbCr color space with chroma subsampling
    # ------------------------------------------------------------------
    'robot36': {
        'vis_code': 8,
        'width': 320,
        'height': 240,
        'color': 'ycbcr',
        'family': 'robot',
        'chroma_format': '420',
        'sync_ms': 9.0,
        'porch_ms': 3.0,
        'y_scan_ms': 88.0,
        'sep_ms': 4.5,
        'sep_porch_ms': 1.5,
        'cr_scan_ms': 44.0,
        'cb_scan_ms': 44.0,
    },
    'robot72': {
        'vis_code': 12,
        'width': 320,
        'height': 240,
        'color': 'ycbcr',
        'family': 'robot',
        'chroma_format': '422',
        'sync_ms': 9.0,
        'porch_ms': 3.0,
        'y_scan_ms': 138.0,
        'sep_ms': 4.5,
        'sep_porch_ms': 1.5,
        'cr_scan_ms': 69.0,
        'cb_scan_ms': 69.0,
    },
}

# Build VIS code -> mode name lookup
VIS_MAP = {m['vis_code']: name for name, m in MODES.items()}


# ============================================================================
# Color Space Conversion (for Robot modes)
# ============================================================================

def rgb_to_ycbcr(r, g, b):
    """
    Convert RGB (0-255) to YCbCr (16-235 / 16-240 range).

    Uses the ITU-R BT.601 conversion used by SSTV:
        Y  =  16 + (65.481 * R + 128.553 * G +  24.966 * B) / 255
        Cb = 128 + (-37.797 * R -  74.203 * G + 112.0   * B) / 255
        Cr = 128 + (112.0   * R -  93.786 * G -  18.214 * B) / 255

    For SSTV we simplify to the 0-255 range used by frequency mapping:
        Y  = 0.299*R + 0.587*G + 0.114*B
        Cb = 128 - 0.169*R - 0.331*G + 0.500*B
        Cr = 128 + 0.500*R - 0.419*G - 0.081*B
    """
    r = np.asarray(r, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    y  = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b

    return (np.clip(y, 0, 255),
            np.clip(cb, 0, 255),
            np.clip(cr, 0, 255))


def ycbcr_to_rgb(y, cb, cr):
    """
    Convert YCbCr back to RGB (0-255).

    Inverse of the above:
        R = Y + 1.402 * (Cr - 128)
        G = Y - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128)
        B = Y + 1.772 * (Cb - 128)
    """
    y  = np.asarray(y, dtype=np.float64)
    cb = np.asarray(cb, dtype=np.float64)
    cr = np.asarray(cr, dtype=np.float64)

    r = y + 1.402 * (cr - 128.0)
    g = y - 0.344136 * (cb - 128.0) - 0.714136 * (cr - 128.0)
    b = y + 1.772 * (cb - 128.0)

    return (np.clip(r, 0, 255).astype(np.uint8),
            np.clip(g, 0, 255).astype(np.uint8),
            np.clip(b, 0, 255).astype(np.uint8))


# ============================================================================
# Calibration Header (Leader + VIS Code)
# ============================================================================

def generate_header(vis_code, sample_rate=SAMPLE_RATE):
    """
    Generate the SSTV calibration header:
        1. Leader tone: 300ms at 1900 Hz
        2. Break: 10ms at 1200 Hz
        3. Leader tone: 300ms at 1900 Hz
        4. VIS start bit: 30ms at 1200 Hz
        5. 7 data bits (LSB first): 30ms each (1100 Hz=1, 1300 Hz=0)
        6. Even parity bit: 30ms
        7. VIS stop bit: 30ms at 1200 Hz

    Parameters:
        vis_code: 7-bit VIS code integer (0-127)
        sample_rate: Audio sample rate

    Returns:
        numpy array of float64 audio samples
    """
    parts = []

    # Leader 1: 300ms at 1900 Hz
    parts.append(_tone(FREQ_VIS_START, HDR_LEADER_MS, sample_rate))
    # Break: 10ms at 1200 Hz
    parts.append(_tone(FREQ_SYNC, HDR_BREAK_MS, sample_rate))
    # Leader 2: 300ms at 1900 Hz
    parts.append(_tone(FREQ_VIS_START, HDR_LEADER_MS, sample_rate))

    # VIS start bit: 30ms at 1200 Hz
    parts.append(_tone(FREQ_SYNC, HDR_VIS_BIT_MS, sample_rate))

    # 7 data bits, LSB first
    ones_count = 0
    for i in range(7):
        bit = (vis_code >> i) & 1
        ones_count += bit
        freq = FREQ_VIS_BIT1 if bit == 1 else FREQ_VIS_BIT0
        parts.append(_tone(freq, HDR_VIS_BIT_MS, sample_rate))

    # Even parity bit
    parity = ones_count % 2  # 1 if odd number of 1s, 0 if even
    freq = FREQ_VIS_BIT1 if parity == 1 else FREQ_VIS_BIT0
    parts.append(_tone(freq, HDR_VIS_BIT_MS, sample_rate))

    # VIS stop bit: 30ms at 1200 Hz
    parts.append(_tone(FREQ_SYNC, HDR_VIS_BIT_MS, sample_rate))

    return np.concatenate(parts)


def detect_vis_code(audio, sample_rate=SAMPLE_RATE):
    """
    Detect the VIS code from an SSTV calibration header.

    Strategy:
        1. Find the leader tone (1900 Hz sustained energy)
        2. Find the break pulse (1200 Hz, 10ms)
        3. Find the second leader tone
        4. Find the VIS start bit (1200 Hz, 30ms)
        5. Decode 7 data bits + parity + stop bit

    Parameters:
        audio: numpy array of audio samples
        sample_rate: Audio sample rate

    Returns:
        (vis_code, data_start_sample) or (-1, 0) if detection fails.
        data_start_sample is the sample index where image data begins
        (after the VIS stop bit).
    """
    # Step 1: Compute instantaneous frequency across the entire signal
    inst_freq = _instantaneous_frequency(audio, sample_rate)

    # Step 2: Find the VIS start sequence:
    #   Leader (~1900 Hz) -> Break (~1200 Hz, 10ms) -> Leader (~1900 Hz) -> Start bit (~1200 Hz, 30ms)
    # We search for the pattern: sustained 1900 -> dip to 1200 -> sustained 1900 -> dip to 1200

    vis_bit_samples = int(HDR_VIS_BIT_MS * sample_rate / 1000)
    leader_samples = int(HDR_LEADER_MS * sample_rate / 1000)
    break_samples = int(HDR_BREAK_MS * sample_rate / 1000)

    # Smooth the frequency estimate for robust detection
    kernel_size = int(0.005 * sample_rate)  # 5ms smoothing
    if kernel_size < 1:
        kernel_size = 1
    kernel = np.ones(kernel_size) / kernel_size
    smooth_freq = np.convolve(inst_freq, kernel, mode='same')

    # Find the break pulse: a narrow region near 1200 Hz surrounded by 1900 Hz
    # Search for the pattern: freq near 1900 -> freq near 1200 -> freq near 1900
    min_leader_len = int(0.1 * sample_rate)  # At least 100ms of leader

    break_pos = -1
    i = min_leader_len

    while i < len(smooth_freq) - leader_samples - vis_bit_samples * 10:
        # Check if current region is near 1200 Hz (break pulse)
        if abs(smooth_freq[i] - FREQ_SYNC) < 150:
            # Check that preceding region was near 1900 Hz (leader)
            pre_start = max(0, i - min_leader_len)
            pre_region = smooth_freq[pre_start:i]
            if len(pre_region) > 0 and np.median(pre_region) > 1700:
                # Find end of this 1200 Hz region
                j = i
                while j < len(smooth_freq) and abs(smooth_freq[j] - FREQ_SYNC) < 200:
                    j += 1
                break_len = j - i
                # Break should be roughly 10ms
                if break_len < int(0.05 * sample_rate):
                    # Check post-break region is near 1900 Hz (second leader)
                    post_end = min(len(smooth_freq), j + min_leader_len)
                    post_region = smooth_freq[j:post_end]
                    if len(post_region) > 0 and np.median(post_region) > 1700:
                        break_pos = i
                        break
            i += int(0.01 * sample_rate)
        else:
            i += int(0.002 * sample_rate)

    if break_pos < 0:
        return -1, 0

    # Find the VIS start bit: next sustained 1200 Hz region after the second leader
    # Skip past the break and second leader
    search_start = break_pos + break_samples + int(0.15 * sample_rate)
    vis_start = -1

    for i in range(search_start, min(len(smooth_freq) - vis_bit_samples * 10,
                                      search_start + int(0.5 * sample_rate))):
        if abs(smooth_freq[i] - FREQ_SYNC) < 150:
            vis_start = i
            break

    if vis_start < 0:
        return -1, 0

    # Step 3: Decode VIS bits starting after the start bit
    bit_start = vis_start + vis_bit_samples  # Skip past start bit

    vis_code = 0
    ones_count = 0

    for bit_idx in range(7):
        center = bit_start + int((bit_idx + 0.5) * vis_bit_samples)
        if center >= len(smooth_freq):
            return -1, 0

        # Sample frequency at center of bit period (with a small window)
        half_win = vis_bit_samples // 4
        region_start = max(0, center - half_win)
        region_end = min(len(smooth_freq), center + half_win)
        avg_freq = np.median(smooth_freq[region_start:region_end])

        # 1100 Hz = 1, 1300 Hz = 0
        if avg_freq < 1200:
            bit_val = 1
        else:
            bit_val = 0

        vis_code |= (bit_val << bit_idx)
        ones_count += bit_val

    # Read parity bit
    parity_center = bit_start + int((7 + 0.5) * vis_bit_samples)
    if parity_center < len(smooth_freq):
        half_win = vis_bit_samples // 4
        region_start = max(0, parity_center - half_win)
        region_end = min(len(smooth_freq), parity_center + half_win)
        avg_freq = np.median(smooth_freq[region_start:region_end])
        parity_bit = 1 if avg_freq < 1200 else 0

        # Verify even parity
        if (ones_count + parity_bit) % 2 != 0:
            print(f"  [WARNING] VIS parity check failed (code={vis_code})")

    # Data starts after the stop bit
    data_start = bit_start + 9 * vis_bit_samples  # 7 data + 1 parity + 1 stop

    return vis_code, data_start


# ============================================================================
# Frequency Modulation / Demodulation
# ============================================================================

def pixel_to_freq(value):
    """
    Map a pixel value (0-255) to SSTV audio frequency.
    0 (black) -> 1500 Hz, 255 (white) -> 2300 Hz.
    """
    return FREQ_BLACK + (value / 255.0) * FREQ_RANGE


def freq_to_pixel(freq):
    """
    Map an SSTV audio frequency to pixel value (0-255).
    1500 Hz -> 0 (black), 2300 Hz -> 255 (white).
    """
    val = (freq - FREQ_BLACK) / FREQ_RANGE * 255.0
    return np.clip(val, 0, 255)


def modulate_scanline(pixel_values, duration_ms, sample_rate=SAMPLE_RATE):
    """
    Frequency-modulate a row of pixel values into audio samples.

    Each pixel value (0-255) maps to a frequency between 1500-2300 Hz.
    Uses continuous-phase synthesis for spectral cleanliness.

    Parameters:
        pixel_values: 1D array of pixel values (0-255), length = image width
        duration_ms: Total scan duration for this line segment (ms)
        sample_rate: Audio sample rate

    Returns:
        numpy array of float64 audio samples
    """
    n_pixels = len(pixel_values)
    total_samples = int(duration_ms * sample_rate / 1000)

    if total_samples == 0 or n_pixels == 0:
        return np.array([], dtype=np.float64)

    # Map each sample to its corresponding pixel via linear interpolation
    # This naturally handles non-integer samples-per-pixel
    sample_indices = np.arange(total_samples, dtype=np.float64)
    pixel_positions = sample_indices * n_pixels / total_samples

    # Clamp to valid range and interpolate pixel values
    pixel_positions = np.clip(pixel_positions, 0, n_pixels - 1)
    left = np.floor(pixel_positions).astype(int)
    right = np.minimum(left + 1, n_pixels - 1)
    frac = pixel_positions - left

    interp_values = pixel_values[left] * (1.0 - frac) + pixel_values[right] * frac

    # Convert pixel values to frequencies
    freqs = FREQ_BLACK + (interp_values / 255.0) * FREQ_RANGE

    # Continuous-phase FM synthesis: integrate frequency to get phase
    phase = np.cumsum(2.0 * np.pi * freqs / sample_rate)

    return np.sin(phase)


def demodulate_to_pixels(audio, n_pixels, sample_rate=SAMPLE_RATE):
    """
    Demodulate an audio segment back to pixel values.

    Uses instantaneous frequency estimation via analytic signal.

    Parameters:
        audio: numpy array of audio samples for one scan segment
        n_pixels: Number of pixels to extract
        sample_rate: Audio sample rate

    Returns:
        numpy array of uint8 pixel values, length n_pixels
    """
    if len(audio) < 4 or n_pixels == 0:
        return np.zeros(n_pixels, dtype=np.uint8)

    # Get instantaneous frequency
    inst_freq = _instantaneous_frequency(audio, sample_rate)

    # Map each pixel to a window in the frequency array
    samples_per_pixel = len(inst_freq) / n_pixels
    pixels = np.zeros(n_pixels, dtype=np.float64)

    for i in range(n_pixels):
        start = int(i * samples_per_pixel)
        end = int((i + 1) * samples_per_pixel)
        if end > len(inst_freq):
            end = len(inst_freq)
        if start >= end:
            continue
        # Use median for robustness against transient glitches
        pixels[i] = np.median(inst_freq[start:end])

    # Convert frequency to pixel value
    pixel_values = freq_to_pixel(pixels)
    return np.clip(pixel_values, 0, 255).astype(np.uint8)


# ============================================================================
# Scan Line Assembly (Encoding)
# ============================================================================

def encode_martin_line(g_row, b_row, r_row, mode, sample_rate=SAMPLE_RATE):
    """
    Encode one scan line in Martin format.

    Martin line structure:
        [Sync pulse] [Sync porch] [Green] [Sep] [Blue] [Sep] [Red] [Sep]

    Parameters:
        g_row, b_row, r_row: 1D arrays of pixel values (0-255), length = width
        mode: Mode dict from MODES
        sample_rate: Audio sample rate

    Returns:
        numpy array of float64 audio samples
    """
    parts = []

    # Sync pulse (1200 Hz)
    parts.append(_tone(FREQ_SYNC, mode['sync_ms'], sample_rate))
    # Sync porch (1500 Hz = black level)
    parts.append(_tone(FREQ_BLACK, mode['sync_porch_ms'], sample_rate))
    # Green channel
    parts.append(modulate_scanline(g_row.astype(np.float64), mode['chan_ms'], sample_rate))
    # Separator
    parts.append(_tone(FREQ_BLACK, mode['sep_ms'], sample_rate))
    # Blue channel
    parts.append(modulate_scanline(b_row.astype(np.float64), mode['chan_ms'], sample_rate))
    # Separator
    parts.append(_tone(FREQ_BLACK, mode['sep_ms'], sample_rate))
    # Red channel
    parts.append(modulate_scanline(r_row.astype(np.float64), mode['chan_ms'], sample_rate))
    # Separator
    parts.append(_tone(FREQ_BLACK, mode['sep_ms'], sample_rate))

    return np.concatenate(parts)


def encode_scottie_line(g_row, b_row, r_row, mode, is_first_line=False,
                        sample_rate=SAMPLE_RATE):
    """
    Encode one scan line in Scottie format.

    Scottie line structure (first line has a leading sync):
        First line:  [Sync] [Sync porch] [Green] [Sep] [Blue] [Sync] [Sync porch] [Red] [Sep]
        Other lines: [Sep] [Green] [Sep] [Blue] [Sync] [Sync porch] [Red] [Sep]

    The sync pulse appears between Blue and Red channels (mid-line).

    Parameters:
        g_row, b_row, r_row: 1D arrays of pixel values (0-255)
        mode: Mode dict from MODES
        is_first_line: True for the very first scan line
        sample_rate: Audio sample rate

    Returns:
        numpy array of float64 audio samples
    """
    parts = []

    if is_first_line:
        # First line starts with a leading sync
        parts.append(_tone(FREQ_SYNC, mode['sync_ms'], sample_rate))
        # Sync porch
        parts.append(_tone(FREQ_BLACK, mode['sync_porch_ms'], sample_rate))

    # Separator (before green, except first line which has sync porch)
    if not is_first_line:
        parts.append(_tone(FREQ_BLACK, mode['sep_ms'], sample_rate))

    # Green channel
    parts.append(modulate_scanline(g_row.astype(np.float64), mode['chan_ms'], sample_rate))
    # Separator
    parts.append(_tone(FREQ_BLACK, mode['sep_ms'], sample_rate))
    # Blue channel
    parts.append(modulate_scanline(b_row.astype(np.float64), mode['chan_ms'], sample_rate))

    # Sync pulse (mid-line, between blue and red)
    parts.append(_tone(FREQ_SYNC, mode['sync_ms'], sample_rate))
    # Sync porch
    parts.append(_tone(FREQ_BLACK, mode['sync_porch_ms'], sample_rate))

    # Red channel
    parts.append(modulate_scanline(r_row.astype(np.float64), mode['chan_ms'], sample_rate))

    # Trailing separator
    # (this becomes the leading separator of the next line in continuous tx)
    # We include it here for the last line
    # -- actually, the next line's leading sep handles this. Skip for non-last.
    # For simplicity and correctness, we omit the trailing sep here since
    # the next line starts with one. The final line gets a trailing sep below.

    return np.concatenate(parts)


def encode_robot36_line(y_row, cb_row, cr_row, line_num, mode,
                        sample_rate=SAMPLE_RATE):
    """
    Encode one scan line in Robot 36 format.

    Robot 36 uses YCbCr 4:2:0:
        Every line:       [Sync] [Porch] [Y scan]
        Even lines (0,2): [Sep] [Sep porch] [Cr scan]
        Odd lines (1,3):  [Sep] [Sep porch] [Cb scan]

    Chrominance is half-resolution vertically — each Cr/Cb line applies
    to a pair of luma lines.

    Parameters:
        y_row: 1D array of Y values (0-255), length = width
        cb_row: 1D array of Cb values (0-255), length = width
        cr_row: 1D array of Cr values (0-255), length = width
        line_num: Scan line number (0-based)
        mode: Mode dict from MODES
        sample_rate: Audio sample rate

    Returns:
        numpy array of float64 audio samples
    """
    parts = []

    # Sync pulse
    parts.append(_tone(FREQ_SYNC, mode['sync_ms'], sample_rate))
    # Porch
    parts.append(_tone(FREQ_BLACK, mode['porch_ms'], sample_rate))
    # Y (luminance) scan
    parts.append(modulate_scanline(y_row.astype(np.float64), mode['y_scan_ms'], sample_rate))

    # Chrominance: even lines get Cr (R-Y), odd lines get Cb (B-Y)
    # Separator
    parts.append(_tone(FREQ_SYNC, mode['sep_ms'], sample_rate))
    # Separator porch
    parts.append(_tone(FREQ_BLACK, mode['sep_porch_ms'], sample_rate))

    if line_num % 2 == 0:
        # Even line: Cr (R-Y)
        parts.append(modulate_scanline(cr_row.astype(np.float64), mode['cr_scan_ms'], sample_rate))
    else:
        # Odd line: Cb (B-Y)
        parts.append(modulate_scanline(cb_row.astype(np.float64), mode['cb_scan_ms'], sample_rate))

    return np.concatenate(parts)


def encode_robot72_line(y_row, cb_row, cr_row, mode, sample_rate=SAMPLE_RATE):
    """
    Encode one scan line in Robot 72 format.

    Robot 72 uses YCbCr 4:2:2:
        [Sync] [Porch] [Y scan] [Sep] [Sep porch] [Cr scan] [Sep] [Sep porch] [Cb scan]

    Every line has full Y, Cr, and Cb (at half the Y scan time each).

    Parameters:
        y_row, cb_row, cr_row: 1D arrays of values (0-255)
        mode: Mode dict from MODES
        sample_rate: Audio sample rate

    Returns:
        numpy array of float64 audio samples
    """
    parts = []

    # Sync pulse
    parts.append(_tone(FREQ_SYNC, mode['sync_ms'], sample_rate))
    # Porch
    parts.append(_tone(FREQ_BLACK, mode['porch_ms'], sample_rate))
    # Y scan
    parts.append(modulate_scanline(y_row.astype(np.float64), mode['y_scan_ms'], sample_rate))

    # Cr (R-Y)
    parts.append(_tone(FREQ_SYNC, mode['sep_ms'], sample_rate))
    parts.append(_tone(FREQ_BLACK, mode['sep_porch_ms'], sample_rate))
    parts.append(modulate_scanline(cr_row.astype(np.float64), mode['cr_scan_ms'], sample_rate))

    # Cb (B-Y)
    parts.append(_tone(FREQ_SYNC, mode['sep_ms'], sample_rate))
    parts.append(_tone(FREQ_BLACK, mode['sep_porch_ms'], sample_rate))
    parts.append(modulate_scanline(cb_row.astype(np.float64), mode['cb_scan_ms'], sample_rate))

    return np.concatenate(parts)


# ============================================================================
# Scan Line Parsing (Decoding)
# ============================================================================

def compute_line_samples(mode, sample_rate=SAMPLE_RATE):
    """
    Compute the expected number of samples per scan line for a given mode.

    Returns:
        (total_line_samples, channel_layout)
        channel_layout is a list of (label, start_sample, n_samples) tuples
    """
    ms_to_samp = lambda ms: int(ms * sample_rate / 1000)

    if mode['family'] == 'martin':
        layout = []
        pos = 0
        # Sync
        n = ms_to_samp(mode['sync_ms'])
        layout.append(('sync', pos, n))
        pos += n
        # Sync porch
        n = ms_to_samp(mode['sync_porch_ms'])
        layout.append(('porch', pos, n))
        pos += n
        # Green
        n = ms_to_samp(mode['chan_ms'])
        layout.append(('green', pos, n))
        pos += n
        # Sep
        n = ms_to_samp(mode['sep_ms'])
        layout.append(('sep1', pos, n))
        pos += n
        # Blue
        n = ms_to_samp(mode['chan_ms'])
        layout.append(('blue', pos, n))
        pos += n
        # Sep
        n = ms_to_samp(mode['sep_ms'])
        layout.append(('sep2', pos, n))
        pos += n
        # Red
        n = ms_to_samp(mode['chan_ms'])
        layout.append(('red', pos, n))
        pos += n
        # Sep
        n = ms_to_samp(mode['sep_ms'])
        layout.append(('sep3', pos, n))
        pos += n
        return pos, layout

    elif mode['family'] == 'scottie':
        layout = []
        pos = 0
        # Separator (leading, from previous line)
        n = ms_to_samp(mode['sep_ms'])
        layout.append(('sep_lead', pos, n))
        pos += n
        # Green
        n = ms_to_samp(mode['chan_ms'])
        layout.append(('green', pos, n))
        pos += n
        # Sep
        n = ms_to_samp(mode['sep_ms'])
        layout.append(('sep1', pos, n))
        pos += n
        # Blue
        n = ms_to_samp(mode['chan_ms'])
        layout.append(('blue', pos, n))
        pos += n
        # Sync (mid-line)
        n = ms_to_samp(mode['sync_ms'])
        layout.append(('sync', pos, n))
        pos += n
        # Sync porch
        n = ms_to_samp(mode['sync_porch_ms'])
        layout.append(('porch', pos, n))
        pos += n
        # Red
        n = ms_to_samp(mode['chan_ms'])
        layout.append(('red', pos, n))
        pos += n
        return pos, layout

    elif mode['family'] == 'robot':
        if mode['chroma_format'] == '420':
            # Robot 36 — line length depends on even/odd, use even (longer)
            layout = []
            pos = 0
            n = ms_to_samp(mode['sync_ms'])
            layout.append(('sync', pos, n))
            pos += n
            n = ms_to_samp(mode['porch_ms'])
            layout.append(('porch', pos, n))
            pos += n
            n = ms_to_samp(mode['y_scan_ms'])
            layout.append(('y', pos, n))
            pos += n
            n = ms_to_samp(mode['sep_ms'])
            layout.append(('sep', pos, n))
            pos += n
            n = ms_to_samp(mode['sep_porch_ms'])
            layout.append(('sep_porch', pos, n))
            pos += n
            # Chroma scan (Cr or Cb depending on line number)
            n = ms_to_samp(mode['cr_scan_ms'])
            layout.append(('chroma', pos, n))
            pos += n
            return pos, layout

        else:
            # Robot 72 (4:2:2)
            layout = []
            pos = 0
            n = ms_to_samp(mode['sync_ms'])
            layout.append(('sync', pos, n))
            pos += n
            n = ms_to_samp(mode['porch_ms'])
            layout.append(('porch', pos, n))
            pos += n
            n = ms_to_samp(mode['y_scan_ms'])
            layout.append(('y', pos, n))
            pos += n
            n = ms_to_samp(mode['sep_ms'])
            layout.append(('sep1', pos, n))
            pos += n
            n = ms_to_samp(mode['sep_porch_ms'])
            layout.append(('sep_porch1', pos, n))
            pos += n
            n = ms_to_samp(mode['cr_scan_ms'])
            layout.append(('cr', pos, n))
            pos += n
            n = ms_to_samp(mode['sep_ms'])
            layout.append(('sep2', pos, n))
            pos += n
            n = ms_to_samp(mode['sep_porch_ms'])
            layout.append(('sep_porch2', pos, n))
            pos += n
            n = ms_to_samp(mode['cb_scan_ms'])
            layout.append(('cb', pos, n))
            pos += n
            return pos, layout

    return 0, []


def decode_martin_line(audio, mode, sample_rate=SAMPLE_RATE):
    """
    Decode one Martin-format scan line from audio samples.

    Parameters:
        audio: numpy array of audio samples for exactly one line
        mode: Mode dict
        sample_rate: Audio sample rate

    Returns:
        (r_pixels, g_pixels, b_pixels) - each a uint8 array of length width
    """
    _, layout = compute_line_samples(mode, sample_rate)
    layout_dict = {label: (start, n) for label, start, n in layout}
    width = mode['width']

    g_start, g_len = layout_dict['green']
    b_start, b_len = layout_dict['blue']
    r_start, r_len = layout_dict['red']

    g = demodulate_to_pixels(audio[g_start:g_start + g_len], width, sample_rate)
    b = demodulate_to_pixels(audio[b_start:b_start + b_len], width, sample_rate)
    r = demodulate_to_pixels(audio[r_start:r_start + r_len], width, sample_rate)

    return r, g, b


def decode_scottie_line(audio, mode, sample_rate=SAMPLE_RATE):
    """
    Decode one Scottie-format scan line from audio samples.

    Parameters:
        audio: numpy array of audio samples for one line
               (starting from separator before green)
        mode: Mode dict
        sample_rate: Audio sample rate

    Returns:
        (r_pixels, g_pixels, b_pixels) - each a uint8 array
    """
    _, layout = compute_line_samples(mode, sample_rate)
    layout_dict = {label: (start, n) for label, start, n in layout}
    width = mode['width']

    g_start, g_len = layout_dict['green']
    b_start, b_len = layout_dict['blue']
    r_start, r_len = layout_dict['red']

    g = demodulate_to_pixels(audio[g_start:g_start + g_len], width, sample_rate)
    b = demodulate_to_pixels(audio[b_start:b_start + b_len], width, sample_rate)
    r = demodulate_to_pixels(audio[r_start:r_start + r_len], width, sample_rate)

    return r, g, b


def decode_robot36_line(audio, mode, line_num, sample_rate=SAMPLE_RATE):
    """
    Decode one Robot 36 scan line.

    Returns:
        (y_pixels, chroma_pixels, chroma_type)
        chroma_type is 'cr' for even lines, 'cb' for odd lines
    """
    _, layout = compute_line_samples(mode, sample_rate)
    layout_dict = {label: (start, n) for label, start, n in layout}
    width = mode['width']

    y_start, y_len = layout_dict['y']
    c_start, c_len = layout_dict['chroma']

    y = demodulate_to_pixels(audio[y_start:y_start + y_len], width, sample_rate)
    c = demodulate_to_pixels(audio[c_start:c_start + c_len], width, sample_rate)

    chroma_type = 'cr' if line_num % 2 == 0 else 'cb'
    return y, c, chroma_type


def decode_robot72_line(audio, mode, sample_rate=SAMPLE_RATE):
    """
    Decode one Robot 72 scan line.

    Returns:
        (y_pixels, cr_pixels, cb_pixels)
    """
    _, layout = compute_line_samples(mode, sample_rate)
    layout_dict = {label: (start, n) for label, start, n in layout}
    width = mode['width']

    y_start, y_len = layout_dict['y']
    cr_start, cr_len = layout_dict['cr']
    cb_start, cb_len = layout_dict['cb']

    y = demodulate_to_pixels(audio[y_start:y_start + y_len], width, sample_rate)
    cr = demodulate_to_pixels(audio[cr_start:cr_start + cr_len], width, sample_rate)
    cb = demodulate_to_pixels(audio[cb_start:cb_start + cb_len], width, sample_rate)

    return y, cr, cb


# ============================================================================
# Sync Pulse Detection
# ============================================================================

def find_sync_pulses(audio, sample_rate=SAMPLE_RATE, min_sync_ms=3.0,
                     max_sync_ms=15.0):
    """
    Find horizontal sync pulses (1200 Hz regions) in the audio.

    Returns:
        List of (start_sample, end_sample) tuples for each detected sync pulse.
    """
    inst_freq = _instantaneous_frequency(audio, sample_rate)

    # Smooth for robustness
    kernel_size = int(0.002 * sample_rate)  # 2ms smoothing
    if kernel_size < 1:
        kernel_size = 1
    kernel = np.ones(kernel_size) / kernel_size
    smooth = np.convolve(inst_freq, kernel, mode='same')

    # Find regions where frequency is near 1200 Hz
    is_sync = np.abs(smooth - FREQ_SYNC) < 150

    # Find runs of sync
    pulses = []
    in_sync = False
    sync_start = 0
    min_samples = int(min_sync_ms * sample_rate / 1000)
    max_samples = int(max_sync_ms * sample_rate / 1000)

    for i in range(len(is_sync)):
        if is_sync[i] and not in_sync:
            in_sync = True
            sync_start = i
        elif not is_sync[i] and in_sync:
            in_sync = False
            length = i - sync_start
            if min_samples <= length <= max_samples:
                pulses.append((sync_start, i))

    return pulses


# ============================================================================
# WAV I/O Utilities
# ============================================================================

def read_wav(filepath):
    """
    Read a WAV file and return (audio_float64, sample_rate).
    Handles mono/stereo and various bit depths.
    Resamples to SAMPLE_RATE if needed.
    """
    import wave
    import struct

    with wave.open(filepath, 'rb') as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        file_sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    # Convert to float64
    if sample_width == 1:
        # 8-bit unsigned
        samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float64) / 128.0 - 1.0
    elif sample_width == 2:
        # 16-bit signed
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    elif sample_width == 4:
        # 32-bit signed
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float64) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    # Mix to mono if needed
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    # Resample if needed
    if file_sr != SAMPLE_RATE:
        duration = len(samples) / file_sr
        new_len = int(duration * SAMPLE_RATE)
        old_indices = np.arange(len(samples))
        new_indices = np.linspace(0, len(samples) - 1, new_len)
        samples = np.interp(new_indices, old_indices, samples)

    return samples, SAMPLE_RATE


def write_wav(filepath, audio, sample_rate=SAMPLE_RATE):
    """
    Write a float64 audio array to a 16-bit mono WAV file.
    """
    import wave

    # Normalize to [-1, 1]
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.95

    # Convert to 16-bit
    pcm = (audio * 32767).astype(np.int16)

    with wave.open(filepath, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


# ============================================================================
# Internal Helpers
# ============================================================================

def _tone(freq, duration_ms, sample_rate=SAMPLE_RATE):
    """Generate a pure sine tone at the given frequency and duration."""
    n = int(duration_ms * sample_rate / 1000)
    t = np.arange(n, dtype=np.float64) / sample_rate
    return np.sin(2.0 * np.pi * freq * t)


def _instantaneous_frequency(audio, sample_rate=SAMPLE_RATE):
    """
    Estimate instantaneous frequency using the analytic signal (Hilbert transform).

    Returns an array of frequency values (Hz), same length as audio.
    """
    n = len(audio)
    if n < 4:
        return np.full(n, FREQ_VIS_START)

    # Compute analytic signal via FFT-based Hilbert transform
    spectrum = np.fft.fft(audio)
    freqs_fft = np.fft.fftfreq(n)

    # Zero out negative frequencies, double positive
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = 1        # DC
        h[n // 2] = 1   # Nyquist
        h[1:n // 2] = 2  # Positive frequencies
    else:
        h[0] = 1
        h[1:(n + 1) // 2] = 2

    analytic = np.fft.ifft(spectrum * h)

    # Instantaneous phase
    phase = np.unwrap(np.angle(analytic))

    # Instantaneous frequency = d(phase)/dt / (2*pi)
    # Use central differences for smoother estimate
    inst_freq = np.zeros(n)
    inst_freq[1:-1] = (phase[2:] - phase[:-2]) / 2.0 * sample_rate / (2.0 * np.pi)
    inst_freq[0] = inst_freq[1] if n > 1 else 0
    inst_freq[-1] = inst_freq[-2] if n > 1 else 0

    # Clip to reasonable SSTV range
    inst_freq = np.clip(inst_freq, 900, 2500)

    return inst_freq


def list_modes():
    """Print a summary of all supported SSTV modes."""
    print(f"{'Mode':<12} {'VIS':>4} {'Size':>9} {'Color':>6} {'Family':>8}")
    print("-" * 45)
    for name, m in MODES.items():
        size = f"{m['width']}x{m['height']}"
        print(f"{name:<12} {m['vis_code']:>4} {size:>9} {m['color']:>6} {m['family']:>8}")
