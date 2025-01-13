import ijson
import pandas as pd
from pathlib import Path

#file chunk
for part in range(1, 17):
    file_path = f"uncompiled_data/new_export/amplitude_export_chunk_{part}_anonymized.json"

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

    path = Path(f"2024/{part}_csv")
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
                output_csv = f"2024/{part}_csv/{file_path.split('/')[1].split('.')[0]}_subchunk_{count*batch_size}_{(count+1)*batch_size}.csv"
                df = df[columns_keep] #remove empty columns
                df.to_csv(output_csv, index=False)
                count += 1
                chunk = []

        if chunk: #process remaining data if any
            output_csv = f"2024/{part}_csv/{file_path.split('/')[1].split('.')[0]}_subchunk_{count*batch_size}_{(count+1)*batch_size}.csv"
            df = pd.DataFrame(chunk)
            print(df.shape)
            df = df[columns_keep]
            print(df.shape)
            df.to_csv(output_csv, index=False)
