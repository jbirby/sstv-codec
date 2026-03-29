# SSTV Codec

A pure Python SSTV (Slow-Scan Television) encoder and decoder. Converts images to standards-compliant SSTV audio WAV files and decodes SSTV audio back into images.

SSTV is the analog image transmission method used by amateur radio operators to send pictures over HF/VHF radio. Each pixel's brightness is frequency-modulated onto audio between 1500 Hz (black) and 2300 Hz (white), with synchronization pulses at 1200 Hz. A single image takes between 36 seconds and 269 seconds depending on the mode.

## Features

- **7 SSTV modes**: Martin M1/M2, Scottie S1/S2/DX, Robot 36/72
- **Automatic mode detection**: Decodes the VIS (Vertical Interval Signalling) code from the calibration header
- **Standards-compliant output**: Generated WAV files are decodable by any SSTV decoder (MMSSTV, Robot36 app, QSSTV, etc.)
- **Real-world recording support**: Decoder handles arbitrary sample rates and has sync pulse tracking for alignment
- **Pure Python + NumPy**: No exotic DSP dependencies

## Supported Modes

| Mode | VIS Code | Resolution | Color | Duration | Notes |
|------|----------|------------|-------|----------|-------|
| Martin M1 | 44 | 320x256 | RGB | ~114s | Most popular in Europe |
| Martin M2 | 40 | 320x256 | RGB | ~58s | Faster Martin variant |
| Scottie S1 | 60 | 320x256 | RGB | ~110s | Most popular in USA |
| Scottie S2 | 56 | 320x256 | RGB | ~71s | Faster Scottie variant |
| Scottie DX | 76 | 320x256 | RGB | ~269s | High quality Scottie |
| Robot 36 | 8 | 320x240 | YCbCr | ~36s | Fast, very popular |
| Robot 72 | 12 | 320x240 | YCbCr | ~72s | Better quality Robot |

## Installation

```bash
pip install numpy Pillow
```

No other dependencies required. Uses only the standard library `wave` module for audio I/O.

## Usage

### Encoding (Image to SSTV Audio)

```bash
python3 sstv_encode.py photo.jpg output.wav --mode martin1
```

The encoder loads any image format PIL supports (JPG, PNG, BMP, TIFF, etc.), resizes it to the mode's native resolution, generates the calibration header with VIS code, and writes a 16-bit mono WAV at 44100 Hz.

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--mode`, `-m` | `martin1` | SSTV mode to use |
| `--sample-rate`, `-sr` | `44100` | Audio sample rate in Hz |

### Decoding (SSTV Audio to Image)

```bash
python3 sstv_decode.py recording.wav output.png
```

The decoder auto-detects the mode from the VIS code. If auto-detection fails (e.g., noisy recording with corrupted header), specify the mode manually:

```bash
python3 sstv_decode.py recording.wav output.png --mode scottie1
```

The decoder handles WAV files at any sample rate (resamples internally to 44100 Hz), mono or stereo.

## How SSTV Works

### Signal Structure

An SSTV transmission consists of:

1. **Calibration Header**: 300ms leader tone at 1900 Hz, 10ms break at 1200 Hz, 300ms leader at 1900 Hz, then a VIS code identifying the mode (7 data bits + parity, each 30ms, at 1100/1300 Hz).

2. **Scan Lines**: Each line contains sync pulses and color channel data. Pixel brightness maps to frequency: `freq = 1500 + (value / 255) * 800` Hz.

### Mode Families

**Martin** (M1, M2): Sync pulse at the start of each line. Color channels transmitted in Green-Blue-Red order. Scan time per channel: 146.432ms (M1) or 73.216ms (M2).

**Scottie** (S1, S2, DX): Sync pulse in the middle of each line (between Blue and Red). Same GBR color order. Scan time per channel: 138.240ms (S1), 88.064ms (S2), or 345.600ms (DX).

**Robot** (36, 72): Uses YCbCr color space instead of RGB. Robot 36 uses 4:2:0 chroma subsampling (alternating Cr/Cb lines), while Robot 72 uses 4:2:2 (full chroma every line).

### Frequency Mapping

| Frequency | Meaning |
|-----------|---------|
| 1200 Hz | Sync pulse |
| 1500 Hz | Black level |
| 1900 Hz | Leader / calibration tone |
| 2300 Hz | White level |
| 1100 Hz | VIS bit = 1 |
| 1300 Hz | VIS bit = 0 |

## File Structure

```
sstv_common.py   # Shared module: mode defs, DSP, protocol logic (~700 lines)
sstv_encode.py   # Image to WAV encoder (CLI wrapper)
sstv_decode.py   # WAV to image decoder (CLI wrapper)
```

`sstv_common.py` contains all the signal processing, mode definitions, and protocol logic. The encode/decode scripts are thin CLI wrappers that handle argument parsing and orchestrate the workflow.

## Technical Details

### Modulation

Pixel values are frequency-modulated using continuous-phase synthesis. The instantaneous frequency at each audio sample is determined by interpolating the pixel values across the scan duration, then integrating the frequency to produce a smooth phase trajectory. This avoids spectral splatter from discontinuous frequency switching.

### Demodulation

The decoder uses an FFT-based Hilbert transform to compute the analytic signal, from which the instantaneous frequency is derived via phase differentiation. Median filtering over pixel-width windows provides robustness against transient noise.

### Sync Detection

Horizontal sync pulses (1200 Hz) are detected by thresholding the instantaneous frequency estimate. The decoder uses a two-pass approach: initial coarse alignment from the first sync pulse, then per-line refinement by searching near the expected position. This tracks gradual clock drift between transmitter and receiver.

### VIS Code Detection

The decoder searches for the calibration header pattern (1900 Hz leader, 1200 Hz break, 1900 Hz leader) to locate the VIS code. Each VIS bit is decoded by measuring the median frequency over its 30ms period.

## Interoperability

The encoder produces signals that can be decoded by:
- **MMSSTV** (Windows)
- **QSSTV** (Linux)
- **Robot36** (Android)
- **CQ SSTV** (iOS)
- **MultiScan** (macOS)
- Any other standards-compliant SSTV decoder

The decoder can process:
- WAV files produced by this encoder (lossless roundtrip)
- Recordings of real SSTV transmissions
- Audio from any SSTV encoder that follows standard timing

## License

MIT
