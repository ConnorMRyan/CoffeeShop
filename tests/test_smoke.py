import subprocess
import pytest
import os

def test_train_smoke():
    """Smoke test to ensure training loop starts and runs for a few steps."""
    # Run train.py with AIsaac stub for 10 steps
    cmd = [
        "python3", "coffeeshop/train.py",
        "env=aisaac",
        "trainer.total_steps=10",
        "trainer.update_interval=5",
        "trainer.log_interval=1",
        "trainer.device=cpu",
        "trainer.tracking.use_tb=false",
        "trainer.tracking.use_wandb=false"
    ]
    
    # Set PYTHONPATH to current directory to ensure imports work
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env
    )
    
    try:
        stdout, stderr = process.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        pytest.fail(f"Smoke test timed out. Stdout: {stdout}\nStderr: {stderr}")

    assert process.returncode == 0, f"Training failed with exit code {process.returncode}.\nStdout: {stdout}\nStderr: {stderr}"
    assert "Training completed" in stdout or "step 10" in stdout.lower() or "Step 10" in stdout
