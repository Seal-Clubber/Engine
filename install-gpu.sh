#!/bin/bash

set -e

# Function to check if running in WSL
is_wsl() {
    if grep -qi microsoft /proc/version; then
        return 0
    else
        return 1
    fi
}

# Function to get Linux distribution and version
get_distribution() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        echo "$ID $VERSION_ID"
    elif [ -f /etc/lsb-release ]; then
        . /etc/lsb-release
        echo "$DISTRIB_ID $DISTRIB_RELEASE"
    else
        echo "Unknown"
    fi
}

# Function to install CUDA Toolkit
install_cuda() {
    local dist=$1
    case $dist in
        "ubuntu 20.04")
            wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
            sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
            wget https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda-repo-ubuntu2004-12-2-local_12.2.2-535.104.05-1_amd64.deb
            sudo dpkg -i cuda-repo-ubuntu2004-12-2-local_12.2.2-535.104.05-1_amd64.deb
            sudo cp /var/cuda-repo-ubuntu2004-12-2-local/cuda-*-keyring.gpg /usr/share/keyrings/
            sudo apt-get update
            sudo apt-get -y install cuda
            ;;
        "ubuntu 22.04")
            wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
            sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
            wget https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda-repo-ubuntu2204-12-2-local_12.2.2-535.104.05-1_amd64.deb
            sudo dpkg -i cuda-repo-ubuntu2204-12-2-local_12.2.2-535.104.05-1_amd64.deb
            sudo cp /var/cuda-repo-ubuntu2204-12-2-local/cuda-*-keyring.gpg /usr/share/keyrings/
            sudo apt-get update
            sudo apt-get -y install cuda
            ;;
        "debian 10")
            wget https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda-repo-debian10-12-2-local_12.2.2-535.104.05-1_amd64.deb
            sudo dpkg -i cuda-repo-debian10-12-2-local_12.2.2-535.104.05-1_amd64.deb
            sudo cp /var/cuda-repo-debian10-12-2-local/cuda-*-keyring.gpg /usr/share/keyrings/
            sudo add-apt-repository contrib
            sudo apt-get update
            sudo apt-get -y install cuda
            ;;
        "debian 11")
            wget https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda-repo-debian11-12-2-local_12.2.2-535.104.05-1_amd64.deb
            sudo dpkg -i cuda-repo-debian11-12-2-local_12.2.2-535.104.05-1_amd64.deb
            sudo cp /var/cuda-repo-debian11-12-2-local/cuda-*-keyring.gpg /usr/share/keyrings/
            sudo add-apt-repository contrib
            sudo apt-get update
            sudo apt-get -y install cuda
            ;;
        *)
            if is_wsl; then
                wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
                sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
                wget https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda-repo-wsl-ubuntu-12-2-local_12.2.2-1_amd64.deb
                sudo dpkg -i cuda-repo-wsl-ubuntu-12-2-local_12.2.2-1_amd64.deb
                sudo cp /var/cuda-repo-wsl-ubuntu-12-2-local/cuda-*-keyring.gpg /usr/share/keyrings/
                sudo apt-get update
                sudo apt-get -y install cuda
            else
                echo "Unsupported distribution for CUDA installation: $dist"
                exit 1
            fi
            ;;
    esac
}

# Function to install NVIDIA Container Toolkit
install_nvidia_container_toolkit() {
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
}

# Main script
main() {
    if ! command -v sudo &> /dev/null; then
        echo "sudo is required but not installed. Please install sudo first."
        exit 1
    fi

    # Check for root privileges
    if [ "$EUID" -ne 0 ]; then
        echo "Please run as root or using sudo"
        exit 1
    fi

    # Detect distribution
    DIST=$(get_distribution)
    echo "Detected distribution: $DIST"

    # Install CUDA Toolkit
    echo "Installing CUDA Toolkit..."
    install_cuda "$DIST"

    # Install NVIDIA Container Toolkit
    echo "Installing NVIDIA Container Toolkit..."
    install_nvidia_container_toolkit

    echo "Installation complete. Please reboot your system to complete the setup."
}

# Run the main function
main
