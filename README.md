# Fast Video Interpolation Filter for FFmpeg

![Demo Video](assets/bbb_demo_full.mp4)

A high-performance video interpolation filter for FFmpeg, leveraging Nvidia's hardware-based Optical Flow Accelerator (NVOFA FRUC) and the interpolation algorithm from [Nvidia Optical Flow SDK](https://docs.nvidia.com/video-technologies/optical-flow-sdk/nvfruc-programming-guide/index.html). Coupled with Nvidia encoder and decoder, it can interpolate 1920x1080 H.264 video at up to 300 FPS on a desktop GPU. The filter code is based on Ffmpeg's `fps` filter.

---

## Requirements

- **Nvidia GPU**: Turing or newer
- **OS**: Linux
- **Platform**: x86_64

---

## Installation

### Building FFmpeg with the Filter
Run the provided installation script:
```bash
./install.sh
```

### Running the Filter
Once the build is successful, you can test the filter:

```bash
export LD_LIBRARY_PATH=../ffmpeg-nvidia-interpolation/NvOFFRUC/lib/
cd build/ffmpeg

./ffmpeg \
    -hwaccel cuda \
    -hwaccel_output_format cuda \
    -y -i ../../assets/bbb_src.mp4 \
    -filter_complex nvinterpolate=fps=60 \
    -b:v 10M -c:v h264_nvenc out.mp4

ffplay out.mp4
```

---

## Notes

- **Tested Environment:**
  - Ubuntu 24.04
  - CUDA 12.1
  - Nvidia RTX 4090

- **Nvidia Optical Flow SDK Binaries:**
  - The provided binaries are stripped from Optical Flow SDK and included for convenience. Make sure to obtain your own copy from Nvidia Developer Program, and comply with Nvidia’s licensing terms.
 
- **Supported Conversions:**
  - This filter performs reliably only for conversions where the target frame rate is an integer multiple of the source frame rate (e.g., 24 to 48 FPS, 30 to 60 FPS). Non-multiple conversions (e.g., 24 to 30 FPS) are not supported.

---

## Contributions

Contributions are welcome! If you find issues or have ideas for improvements, feel free to open a pull request or file an issue.

