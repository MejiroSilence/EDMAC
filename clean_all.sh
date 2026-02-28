ps aux | grep "name=bs1" | grep -v "grep" | awk '{print $2}' | xargs kill -9
ps aux | grep "name=4610" | grep -v "grep" | awk '{print $2}' | xargs kill -9