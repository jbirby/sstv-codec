#!/usr/bin/env python3
"""
SSTV Encoder — Converts an image to an SSTV audio WAV file.

Usage:
    python3 sstv_encode.py <input_image> <output.wav> [--mode MODE]

Supported modes:
    martin1, martin2, scottie1, scottie2, scottieDX, robot36, robot72

The encoder:
    1. Loads any image format PIL supports (JPG, PNG, BMP, etc.)
    2. Resizes to the mode's native resolution (e.g., 320x256)
    3. Generates the SSTV calibration header (leader + VIS code)
    4. Encodes each scan line with the mode's timing and color format
    5. Writes a 16-bit mono WAV at 44100 Hz
"""

import sys
import os
import argparse
import numpy as np

# Allow importing from same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sstv_common as sstv


def load_and_resize(image_path, width, height):
    """
    Load an image and resize to the target dimensions.

    Returns:
        numpy array of shape (height, width, 3), dtype uint8, RGB
    """
    from PIL import Image

    img = Image.open(image_path).convert('RGB')

    # Resize to exact mode dimensions
    if img.size != (width, height):
        img = img.resize((width, height), Image.LANCZOS)

    return np.array(img, dtype=np.uint8)


def encode_image(image_path, output_path, mode_name='martin1',
                 sample_rate=sstv.SAMPLE_RATE):
    """
    Encode an image as an SSTV WAV file.

    Parameters:
        image_path: Path to input image
        output_path: Path for output WAV
        mode_name: SSTV mode name (e.g., 'martin1', 'scottie1', 'robot36')
        sample_rate: Audio sample rate (default 44100)
    """
    if mode_name not in sstv.MODES:
        print(f"Error: Unknown mode '{mode_name}'")
        print(f"Available modes: {', '.join(sstv.MODES.keys())}")
        sys.exit(1)

    mode = sstv.MODES[mode_name]
    width = mode['width']
    height = mode['height']

    print(f"SSTV Encoder")
    print(f"  Mode:       {mode_name} (VIS {mode['vis_code']})")
    print(f"  Resolution: {width}x{height}")
    print(f"  Color:      {mode['color'].upper()}")
    print(f"  Family:     {mode['family']}")
    print(f"  Input:      {image_path}")
    print(f"  Output:     {output_path}")
    print()

    # Load and resize image
    print("  Loading image...")
    img = load_and_resize(image_path, width, height)
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    # For Robot modes, convert to YCbCr
    if mode['color'] == 'ycbcr':
        print("  Converting to YCbCr...")
        y_img, cb_img, cr_img = sstv.rgb_to_ycbcr(r, g, b)
    else:
        y_img = cb_img = cr_img = None

    # Generate calibration header
    print("  Generating calibration header (VIS code)...")
    header = sstv.generate_header(mode['vis_code'], sample_rate)

    # Encode scan lines
    print(f"  Encoding {height} scan lines...")
    audio_parts = [header]

    for line_num in range(height):
        if (line_num + 1) % 32 == 0 or line_num == height - 1:
            print(f"    Line {line_num + 1}/{height}")

        if mode['family'] == 'martin':
            line_audio = sstv.encode_martin_line(
                g[line_num], b[line_num], r[line_num],
                mode, sample_rate
            )

        elif mode['family'] == 'scottie':
            line_audio = sstv.encode_scottie_line(
                g[line_num], b[line_num], r[line_num],
                mode, is_first_line=(line_num == 0),
                sample_rate=sample_rate
            )

        elif mode['family'] == 'robot':
            if mode['chroma_format'] == '420':
                line_audio = sstv.encode_robot36_line(
                    y_img[line_num], cb_img[line_num], cr_img[line_num],
                    line_num, mode, sample_rate
                )
            else:
                line_audio = sstv.encode_robot72_line(
                    y_img[line_num], cb_img[line_num], cr_img[line_num],
                    mode, sample_rate
                )

        audio_parts.append(line_audio)

    # Add tail silence (enough to ensure last line is fully decodable)
    audio_parts.append(np.zeros(int(0.5 * sample_rate)))

    # Concatenate all audio
    audio = np.concatenate(audio_parts)
    duration = len(audio) / sample_rate

    print(f"  Writing WAV ({duration:.1f}s, {sample_rate} Hz)...")
    sstv.write_wav(output_path, audio, sample_rate)

    file_size = os.path.getsize(output_path)
    print(f"  Done! Output: {file_size:,} bytes ({duration:.1f}s)")


def main():
    parser = argparse.ArgumentParser(
        description='SSTV Encoder — Convert an image to SSTV audio',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported modes:
  martin1    320x256 RGB, 114s  (VIS 44) — Most popular in Europe
  martin2    320x256 RGB, 58s   (VIS 40) — Faster Martin
  scottie1   320x256 RGB, 110s  (VIS 60) — Most popular in USA
  scottie2   320x256 RGB, 71s   (VIS 56) — Faster Scottie
  scottieDX  320x256 RGB, 269s  (VIS 76) — High quality Scottie
  robot36    320x240 YCbCr, 36s (VIS 8)  — Fast, popular
  robot72    320x240 YCbCr, 72s (VIS 12) — Better quality Robot
        """
    )
    parser.add_argument('input', help='Input image file (JPG, PNG, BMP, etc.)')
    parser.add_argument('output', help='Output WAV file')
    parser.add_argument('--mode', '-m', default='martin1',
                        choices=list(sstv.MODES.keys()),
                        help='SSTV mode (default: martin1)')
    parser.add_argument('--sample-rate', '-sr', type=int, default=sstv.SAMPLE_RATE,
                        help=f'Audio sample rate (default: {sstv.SAMPLE_RATE})')

    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    encode_image(args.input, args.output, args.mode, args.sample_rate)


if __name__ == '__main__':
    main()
