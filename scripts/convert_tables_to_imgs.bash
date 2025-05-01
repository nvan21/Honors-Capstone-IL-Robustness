#!/bin/bash

# Directories
INPUT_DIR="./assets/tables"
OUTPUT_DIR="./assets/images"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Loop through each .tex file
for texfile in "$INPUT_DIR"/*.tex; do
    base=$(basename "$texfile" .tex)

    echo "📄 Processing $base.tex..."

    # Compile LaTeX to PDF
    pdflatex -interaction=nonstopmode -output-directory="$INPUT_DIR" "$texfile" >/dev/null 2>&1

    # Crop the PDF to remove whitespace
    pdfcrop "$INPUT_DIR/$base.pdf" "$INPUT_DIR/${base}-cropped.pdf" >/dev/null

    # Convert cropped PDF to PNG
    convert -density 300 "$INPUT_DIR/${base}-cropped.pdf" -quality 100 "$OUTPUT_DIR/$base.png"

    echo "✅ Saved: $OUTPUT_DIR/$base.png"
done

rm -f "$INPUT_DIR"/*.aux "$INPUT_DIR"/*.log "$INPUT_DIR"/*.pdf "$INPUT_DIR"/*-cropped.pdf
echo "🎉 All done!"
