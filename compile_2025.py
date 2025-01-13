import time
import psutil
import os

# Add at the beginning of your script
process = psutil.Process(os.getpid())
start_cpu_time = process.cpu_times()
start_wall_time = time.time()

import ijson
import pandas as pd
from pathlib import Path

#file year
year = 2025
file_path = f"uncompiled_data/new_amplitude_export_{year}.json"

#non-empty columns
columns_keep = [
    "$insert_id",
    "amplitude_id",
    "app",
    "city",
    "client_event_time",
    "client_upload_time",
    "country",
    "data",
    "data_type",
    "device_family",
    "device_id",
    "device_type",
    "dma",
    "event_id",
    "event_properties",
    "event_time",
    "event_type",
    "language",
    "library",
    "os_name",
    "os_version",
    "platform",
    "processed_time",
    "region",
    "server_received_time",
    "server_upload_time",
    "session_id",
    "user_id",
    "user_properties",
    "uuid",
]
path = Path(f"{year}_csv")
if not path.exists():
    path.mkdir(parents=True, exist_ok=True)
#use ijson to read the json files efficiently in memory
with open(file_path, "r") as f:
    objects = ijson.items(f, "item") #creates a generator object

    batch_size = 100000 #can be updated, currently saves per batches of 100,000
    chunk = []
    count = 0 #used to index batch file
    for obj in objects:
        chunk.append(obj)
        if len(chunk) >= batch_size:
            df = pd.DataFrame(chunk)
            output_csv = f"{year}_csv/{file_path.split('.')[0]}_chunk_{count*batch_size}_{(count+1)*batch_size}.csv"
            df = df[columns_keep] #remove empty columns
            df.to_csv(output_csv, index=False)
            count += 1
            chunk = []

    if chunk: #process remaining data if any
        output_csv = f"{year}_csv/{file_path.split('.')[0]}_chunk_{count*batch_size}_{(count+1)*batch_size}.csv"
        df = pd.DataFrame(chunk)
        df = df[columns_keep]
        df.to_csv(output_csv, index=False)


# Replace the end timing with this:
end_cpu_time = process.cpu_times()
end_wall_time = time.time()

# Calculate times
user_cpu_time = end_cpu_time.user - start_cpu_time.user
system_cpu_time = end_cpu_time.system - start_cpu_time.system
total_cpu_time = user_cpu_time + system_cpu_time
wall_time = end_wall_time - start_wall_time

# Format and print the benchmark
print(f"CPU times: user {user_cpu_time/60:.0f}min {user_cpu_time%60:.2f}s, "
    f"sys: {system_cpu_time:.2f} s, "
    f"total: {total_cpu_time/60:.0f}min {total_cpu_time%60:.2f}s")
print(f"Wall time: {wall_time/60:.0f}min {wall_time%60:.2f}s")

# Outputs:

# CPU times: user 7min 51s, sys: 6.23 s, total: 7min 57s
# Wall time: 8min 6s
