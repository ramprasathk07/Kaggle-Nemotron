import os
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import wandb
from dotenv import load_dotenv

load_dotenv()

WANDB_PROJECT_NAME = "my_tensorboard_sync"

def sync_tfevents_to_wandb():
    # Find the tfevents file
    log_dir = r"./logs/tb_logs"
    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    
    if not event_files:
        print("No tfevents file found in", log_dir)
        return
        
    event_file = event_files[0]
    print(f"Reading {event_file}...")
    
    # Load events
    ea = EventAccumulator(event_file)
    ea.Reload()
    
    tags = ea.Tags().get("scalars", [])
    if not tags:
        print("No scalar metrics found in the event file.")
        return
        
    print(f"Found metrics: {tags}")
    
    # Initialize wandb run
    run = wandb.init(project=WANDB_PROJECT_NAME, name="manual-tb-sync")
    print(f"Started WandB run: {run.name} ({run.id})")
    
    # Collect all steps
    steps_data = {}
    
    for tag in tags:
        events = ea.Scalars(tag)
        for event in events:
            step = event.step
            if step not in steps_data:
                steps_data[step] = {}
            steps_data[step][tag] = event.value
            
    # Sort by step and log to wandb
    sorted_steps = sorted(steps_data.keys())
    print(f"Syncing {len(sorted_steps)} steps to WandB...")
    
    for step in sorted_steps:
        metrics = steps_data[step]
        metrics["global_step"] = step
        wandb.log(metrics, step=step)
        
    wandb.finish()
    print("Sync complete! Check your WandB dashboard.")

if __name__ == "__main__":
    sync_tfevents_to_wandb()
