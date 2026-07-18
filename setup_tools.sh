#!/bin/bash
# setup_tools.sh
set -e

mkdir -p tools
cd tools

# Clean up any broken files
rm -f maven.tar.gz
rm -rf maven apache-maven-*

echo "Downloading Apache Maven from CDN..."
curl -L -o maven.tar.gz https://dlcdn.apache.org/maven/maven-3/3.9.16/binaries/apache-maven-3.9.16-bin.tar.gz
echo "Extracting Maven..."
tar -xzf maven.tar.gz
mv apache-maven-3.9.16 maven
rm maven.tar.gz
echo "Maven installed."

# Download and extract GraalVM CE
if [ ! -d "graalvm" ]; then
    echo "Downloading GraalVM JDK 21..."
    curl -L -o graalvm.tar.gz https://github.com/graalvm/graalvm-ce-builds/releases/download/jdk-21.0.2/graalvm-community-jdk-21.0.2_linux-x64_bin.tar.gz
    echo "Extracting GraalVM..."
    tar -xzf graalvm.tar.gz
    mv graalvm-community-jdk-21.0.2 graalvm
    rm graalvm.tar.gz
    echo "GraalVM JDK 21 installed."
fi

echo "All tools installed successfully."
