---
name: sstv-codec
description: >
  Encode images into SSTV (Slow-Scan Television) audio WAV files and decode
  SSTV WAV recordings back into images. Supports Martin M1/M2, Scottie S1/S2/DX,
  and Robot 36/72 modes with automatic mode detection via VIS code. Use this
  skill whenever the user mentions SSTV, slow-scan television, slow scan TV,
  ham radio image transmission, amateur radio pictures, VIS code, Martin mode,
  Scottie mode, Robot mode, or wants to convert images to/from the analog audio
  format used by amateur radio operators for picture transmission. Also trigger
  when the user has a WAV file recorded from an SSTV transmission and wants to
  extract the image, or wants to create a WAV that sounds like a real SSTV
  signal. Even if the user doesn't say "SSTV" explicitly — if they mention
  sending pictures over radio, ham radio image modes, or converting a photo to
  audio for radio transmission, this is almost certainly the skill they need.
  Covers encoding (image to WAV) and decoding (WAV to image).
---

# SSTV Codec

This skill converts between images and SSTV (Slow-Scan Television) audio
files. SSTV is the analog image transmission method used by amateur radio
operators worldwide to send pictures over radio. Each pixel's brightness is
encoded as an audio frequency between 1500 Hz (black) and 2300 Hz (white),
with sync pulses at 1200 Hz.

The generated WAV files are protocol-correct SSTV transmissions. They could
be played into a radio transmitter and received by any SSTV decoder (MMSSTV,
Robot36 app, QSSTV, etc.), and the decoder can process recordings of real
SSTV transmissions received over the air.

## Quick reference: the SSTV signal

An SSTV transmission consists of:

1. **Calibration header** — 300ms leader tone at 1900 Hz, 10ms break at
   1200 Hz, 300ms leader at 1900 Hz, followed by a digital VIS (Vertical
   Interval Signalling) code that identifies the mode. VIS bits are 30ms
   each: 1100 Hz for 1, 1300 Hz for 0, LSB first, with even parity.

2. **Image data** — Scan lines transmitted left-to-right, one color channel
   at a time. Each pixel value (0-255) maps to a frequency:
   `freq = 1500 + (pixel / 255) * 800` Hz.

3. **Mode families** differ in line structure:
   - **Martin**: Sync pulse at line start. Color order Green-Blue-Red.
   - **Scottie**: Sync pulse mid-line (between Blue and Red). Color order GBR.
   - **Robot**: YCbCr color space with chroma subsampling. Sync at line start.

## Supported modes

| Mode      | VIS | Size    | Color  | Time  | Notes                    |
|-----------|-----|---------|--------|-------|--------------------------|
| martin1   | 44  | 320x256 | RGB    | ~114s | Most popular in Europe   |
| martin2   | 40  | 320x256 | RGB    | ~58s  | Faster Martin variant    |
| scottie1  | 60  | 320x256 | RGB    | ~110s | Most popular in USA      |
| scottie2  | 56  | 320x256 | RGB    | ~71s  | Faster Scottie variant   |
| scottieDX | 76  | 320x256 | RGB    | ~269s | High quality Scottie     |
| robot36   | 8   | 320x240 | YCbCr  | ~36s  | Fast, very popular       |
| robot72   | 12  | 320x240 | YCbCr  | ~72s  | Better quality Robot     |

## How to use this skill

There are three Python files in the same directory as this SKILL.md. Use
them rather than writing SSTV logic from scratch:

- `sstv_common.py` — Shared module with all DSP, mode definitions, and protocol logic
- `sstv_encode.py` — Image to WAV encoder (CLI wrapper)
- `sstv_decode.py` — WAV to image decoder (CLI wrapper)

### Encoding (image to SSTV WAV)

```bash
python3 <skill-path>/sstv_encode.py <input_image> <output.wav> [--mode MODE]
```

The encoder:
1. Loads any image format PIL supports (JPG, PNG, BMP, etc.)
2. Resizes to the mode's native resolution (320x256 or 320x240)
3. Generates the calibration header with VIS code
4. Encodes each scan line with the mode's timing and color format
5. Writes a 16-bit mono WAV at 44100 Hz

Default mode is `martin1`. Use `--mode` to select another. Available modes:
`martin1`, `martin2`, `scottie1`, `scottie2`, `scottieDX`, `robot36`, `robot72`.

### Decoding (SSTV WAV to image)

```bash
python3 <skill-path>/sstv_decode.py <input.wav> <output.png> [--mode MODE]
```

The decoder:
1. Reads the WAV (any sample rate — resamples to 44100 if needed)
2. Auto-detects the mode from the VIS code in the calibration header
3. Locates scan line boundaries via sync pulse detection
4. Demodulates each line's color channels from FM audio
5. For Robot modes, converts YCbCr back to RGB
6. Saves the reconstructed image as PNG

If auto-detection fails (e.g., noisy recording with corrupted header), use
`--mode` to specify the mode manually.

### Typical workflow

**User wants to encode an image:**
1. Run the encoder script with the desired mode
2. Optionally verify by decoding the WAV back and checking the roundtrip
3. Deliver the WAV file to the user

**User wants to decode an SSTV recording:**
1. Run the decoder script on their WAV (auto-detects mode)
2. Show them the decoded image
3. Note: real-world recordings may have noise, clock drift, or
   Doppler shift that degrades the image. The decoder has some tolerance
   for these issues via sync pulse tracking and silence padding.

**User wants a roundtrip demonstration:**
1. Encode the image to WAV
2. Decode the WAV back to a new image
3. Compare the two visually or numerically (roundtrip quality is typically
   95-98% of pixels within 5 values of the original for RGB modes)

**User asks about SSTV format details:**
The quick reference above and the mode table cover the key parameters.
The main things people care about: Martin and Scottie use RGB at 320x256,
Robot uses YCbCr at 320x240, and each pixel becomes an audio frequency
between 1500-2300 Hz.

## Dependencies

The scripts use only `numpy`, `Pillow`, and the standard library `wave` module.
Install if needed:

```bash
pip install numpy Pillow --break-system-packages
```
