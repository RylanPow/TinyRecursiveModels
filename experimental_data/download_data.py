import wandb
import os

# This downloads raw parquet files from wandb runs

api = wandb.Api()
entity = # ORGANIZATION NAME HERE
project = # PROJECT NAME HERE

# List your 8-character Run IDs here
run_ids = ["rund id1", "run id2", "etc etc"]

for rid in run_ids:
    print(f"Processing Run: {rid} ")
    
    try:
        # access hidden history artifact
        # wandb automatically names these 'run-<id>-history:v0'
        artifact_path = f"{entity}/{project}/run-{rid}-history:v0"
        print(f"Locating artifact: {artifact_path}...")
        artifact = api.artifact(artifact_path)
        
        print(f"Downloading raw files for {rid}...")
        download_path = artifact.download(root=f"./{rid}_raw")
        
        print(f"SUCCESS: Raw data for {rid} saved in {download_path}\n")
        
    except Exception as e:
        print(f"FAILED to download {rid}: {e}\n")

print("All tasks complete.")