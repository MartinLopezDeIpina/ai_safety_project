
import modal
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))  # repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))       # src/

app = modal.App("ai_safety")

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "transformers", "accelerate", "openai", "numpy")
    .add_local_python_source("config", "model_utils", "_00_collect_behaviors", "utils", "template_inversion", "inference")
)

hf_cache = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("behavior-results", create_if_missing=True)
data_vol = modal.Volume.from_name("behavior-datasets", create_if_missing=True)

@app.function(
    image=image,
    gpu="A100",
    volumes={"/cache": hf_cache, "/root/results": results_vol, "/data": data_vol},
    timeout=15000,
)

def run_collection():
    import os
    os.environ["HF_HUB_CACHE"] = "/cache"
    from _00_collect_behaviors import main
    main()
    results_vol.commit()  # persist writes to results/ back to the Volume

@app.local_entrypoint()
def entry():
    run_collection.remote()
    import subprocess
    # run from root directory to get the results local dir correctly
    subprocess.run(
        ["modal", "volume", "get", "behavior-results", "/", "./src/experiments_replication/results", "--force"])

