#!/bin/bash

set -e

echo "Creating directories..."
mkdir -p data/bundle_images
mkdir -p data/product_images

export LC_ALL=C

echo "Downloading bundle images (this may take a while)..."
awk -F',' 'NR>1 {gsub(/\r/,""); print $1, $3}' data/bundles_dataset.csv | \
  xargs -n 2 -P 20 sh -c 'curl -s -f -A "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36" "$1" -o "data/bundle_images/$0.jpg" || echo "Failed to download $0"'

echo "Downloading product images (this may take a while)..."
awk -F',' 'NR>1 {gsub(/\r/,""); print $1, $2}' data/product_dataset.csv | \
  xargs -n 2 -P 20 sh -c 'curl -s -f -A "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36" "$1" -o "data/product_images/$0.jpg" || echo "Failed to download $0"'

echo "Download completed successfully!"
