#!/bin/bash
# A minimal Ffmpeg and dependencies install script to showcase interpolation filter. Edit options as neccesary.
set -e

CUR_DIR=$(pwd)
BUILD_DIR=$(pwd)/"build"

mkdir -p "$BUILD_DIR"

# Install required packages
sudo apt update && sudo apt install -y \
    build-essential \
    pkg-config \
    yasm \
    nasm \
    libtool \
    cmake \
    git \
    automake \
    autoconf \
    texinfo \
    wget \
    zlib1g-dev \
    libnuma-dev \
    libnuma1

# Build Nvidia Video Codec SDK
build_nvidia_sdk() {
    if [ ! -d "$BUILD_DIR/nv-codec-headers" ]; then
        echo "Building nv-codec-headers..."
        cd "$BUILD_DIR"
        git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git
        cd nv-codec-headers
        git checkout 75f032b24263c2b684b9921755cafc1c08e41b9d
        make -j$(nproc)
        make PREFIX="$BUILD_DIR/nv-codec-headers" install
    else
        echo "nv-codec-headers already built, skipping."
    fi
}

# Build FFmpeg
build_ffmpeg() {
    if [ ! -d "$BUILD_DIR/ffmpeg" ]; then
        cd "$BUILD_DIR"
        git clone https://git.ffmpeg.org/ffmpeg.git ffmpeg
    fi
    cd "$BUILD_DIR/ffmpeg"
    git checkout be4fcf02
    # copy source files
    cp -r $CUR_DIR/src/* "$BUILD_DIR/ffmpeg/"
    # add nvinterpolate filter registration
    grep -q "ff_vf_nvinterpolate" libavfilter/allfilters.c \
         || echo -e "extern const AVFilter ff_vf_nvinterpolate;\n" >> libavfilter/allfilters.c
    grep -q "ff_vf_nvinterpolate" libavfilter/Makefile \
         || echo -e "OBJS-\$(CONFIG_NVINTERPOLATE_FILTER)          += vf_nvinterpolate.o\n" >> libavfilter/Makefile
    PKG_CONFIG_PATH="$BUILD_DIR/nv-codec-headers/lib/pkgconfig" \
    ./configure \
        --bindir="$BUILD_DIR/ffmpeg" \
        --enable-static \
        --disable-shared \
        --pkg-config-flags="--static" \
        --extra-cflags=-I/usr/local/cuda/include \
        --extra-cflags="-I$BUILD_DIR/nv-codec-headers/include" \
        --extra-cflags="-I$CUR_DIR/NvOFFRUC/include/" \
        --extra-ldflags="-L$BUILD_DIR/nv-codec-headers/lib" \
        --extra-ldflags="-L$CUR_DIR/NvOFFRUC/lib" \
        --extra-ldflags=-L/usr/local/cuda/lib64 \
        --enable-cuda-nvcc \
        --enable-nonfree \
        --enable-gpl \
        --enable-libnpp
    make -j4
}

build_nvidia_sdk
build_ffmpeg

echo "Build completed. Use ./build/ffmpeg/ffmpeg binary."

