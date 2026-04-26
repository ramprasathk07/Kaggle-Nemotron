import os
import subprocess
from dotenv import load_dotenv

# ============================================================
# HARD-CODED PATHS AND PROJECT NAME – EDIT THESE
TENSORBOARD_LOG_DIR = r"./logs/tb_logs"      # Your TensorBoard log folder
WANDB_PROJECT_NAME = "my_tensorboard_sync"   # Replace with your wandb project name
WANDB_ENTITY = None                          # Optional: your wandb username/team
# ============================================================

def main():
    # 1. Load environment variables from .env file
    load_dotenv()  # Looks for a file named '.env' in the current directory

    # 2. Read API key from environment (now populated from .env)
    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        raise ValueError(
            "WANDB_API_KEY not found. Please create a .env file with:\n"
            "WANDB_API_KEY=your_actual_api_key_here"
        )

    # 3. (Optional) Explicit login with the key – the 'wandb sync' command will also use it
    try:
        import wandb
        wandb.login(key=api_key)
        print("Logged in to wandb.")
    except ImportError:
        print("wandb Python library not available, but CLI will use the environment variable.")

    import sys
    cmd = [sys.executable, "-m", "wandb", "sync", TENSORBOARD_LOG_DIR]
    if WANDB_PROJECT_NAME:
        cmd.extend(["--project", WANDB_PROJECT_NAME])
    if WANDB_ENTITY:
        cmd.extend(["--entity", WANDB_ENTITY])

    # 5. Run the sync – automatically discovers all runs inside the log directory
    print(f"Syncing all TensorBoard runs from: {TENSORBOARD_LOG_DIR}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    # 6. Print output
    if result.returncode == 0:
        print("Sync successful!")
        print(result.stdout)
    else:
        print("Sync failed with errors:")
        print(result.stderr)

if __name__ == "__main__":
    main()