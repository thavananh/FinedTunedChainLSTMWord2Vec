#!/bin/bash

# Update the system
sudo apt-get update
sudo apt-get upgrade -y

# Install necessary packages
sudo apt-get install htop nvtop nano build-essential sudo curl wget unzip git locales -y

# Configure system language
sudo locale-gen en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US:en

# Check if Anaconda file exists
ANACONDA_FILE="Anaconda3-2024.10-1-Linux-x86_64.sh"
if [ ! -f "$ANACONDA_FILE" ]; then
    echo "Anaconda file not found, downloading..."
    sudo curl -O https://repo.anaconda.com/archive/$ANACONDA_FILE
else
    echo "Anaconda file already exists."
fi

# Grant execution permission to the installer file
sudo chmod 777 $ANACONDA_FILE

# Install Anaconda in silent mode (no interaction required)
echo "Installing Anaconda in silent mode..."
sudo bash ./$ANACONDA_FILE -b -p anaconda3

# After installation, configure the shell
echo "Configuring shell to use Anaconda..."
sudo echo "source anaconda3/bin/activate" >> ~/.bashrc
source ~/.bashrc

echo "Anaconda installation successful!"