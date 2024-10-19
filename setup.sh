#!/bin/bash

# Configure Git
git config --global user.name "paul"
git config --global user.email "paul.plays.a.pun@gmail.com"

# Download and install CMake
wget https://github.com/Kitware/CMake/releases/download/v3.25.1/cmake-3.25.1-Linux-x86_64.sh
chmod +x cmake-3.25.1-Linux-x86_64.sh
sudo ./cmake-3.25.1-Linux-x86_64.sh --skip-license --prefix=/usr/local
rm -rf cmake-3.25.1-Linux-x86_64.sh

# Update the PATH environment variable
echo 'export PATH=/usr/local/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# Download Weights
aws s3 sync s3://llama3-cuda/Meta-Llama-3-8B/ ./Meta-Llama-3-8B

echo "Installation and configuration complete!"

mkdir build