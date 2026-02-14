#!/bin/bash
set -e

echo "Starting VCTK dataset download and extraction..."

# Download VCTK Corpus
echo "Downloading VCTK Corpus..."
if wget -O VCTK-Corpus.zip https://datashare.ed.ac.uk/bitstream/handle/10283/2651/VCTK-Corpus.zip; then
    echo "Extracting VCTK Corpus..."
    unzip VCTK-Corpus.zip
    echo "VCTK Corpus extracted successfully"
    rm VCTK-Corpus.zip
else
    echo "Failed to download VCTK Corpus"
    exit 1
fi


# Download and extract LibriSpeech test-s4t
echo "Downloading LibriSpeech test-clean..."
if wget -O test-clean.tar.gz https://www.openslr.org/resources/60/test-clean.tar.gz; then
    echo "Extracting LibriSpeech test-clean..."
    tar -xzf test-clean.tar.gz
    echo "LibriSpeech test-clean extracted successfully"
    rm test-clean.tar.gz
else
    echo "Failed to download LibriSpeech test-clean"
    exit 1
fi

echo "All datasets downloaded and extracted successfully!"