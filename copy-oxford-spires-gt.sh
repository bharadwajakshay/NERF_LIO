SOURCE_DIR=$1
DEST_DIR=$2

mkdir -p "$DEST_DIR"

find "$SOURCE_DIR" -name "gt-tum.txt" | while read filepath; do
    # Go up 3 levels: trajectory -> processed -> seq_name
    seq_name=$(basename "$(dirname "$(dirname "$(dirname "$filepath")")")")
    
    cp "$filepath" "$DEST_DIR/${seq_name}.txt"
    echo "Copied: $seq_name.txt"
done