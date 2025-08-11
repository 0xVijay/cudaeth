#!/bin/bash

# Progress monitoring script for CUDA compilation
# Helps detect if compilation is hanging vs. just taking a long time

echo "=== CUDA Compilation Progress Monitor ==="
echo "This script monitors the compilation process to detect hanging"
echo

# Function to monitor compilation
monitor_compilation() {
    local make_pid=$1
    local count=0
    
    while kill -0 $make_pid 2>/dev/null; do
        sleep 30
        count=$((count + 1))
        echo "Compilation running for $((count * 30)) seconds..."
        
        # Check if any new .o files were created in the last 30 seconds
        recent_files=$(find build/obj -name "*.o" -newermt "30 seconds ago" 2>/dev/null | wc -l)
        if [ $recent_files -gt 0 ]; then
            echo "  Progress: $recent_files object file(s) updated recently"
        else
            echo "  No recent object file updates"
        fi
        
        # Show memory usage
        if command -v free &> /dev/null; then
            mem_free=$(free -m | grep '^Mem:' | awk '{print $7}')
            echo "  Available memory: ${mem_free}MB"
        fi
        
        # If no progress for 5 minutes (10 iterations), warn
        if [ $count -ge 10 ]; then
            echo "  WARNING: Compilation has been running for 5+ minutes"
            echo "  This may indicate hanging. Consider using minimal target."
        fi
    done
}

# Start compilation in background
echo "Starting compilation..."
make clean
make minimal &
MAKE_PID=$!

# Monitor the process
monitor_compilation $MAKE_PID

# Wait for completion and get exit code
wait $MAKE_PID
EXIT_CODE=$?

echo
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Compilation completed successfully!"
else
    echo "✗ Compilation failed with exit code: $EXIT_CODE"
fi

exit $EXIT_CODE