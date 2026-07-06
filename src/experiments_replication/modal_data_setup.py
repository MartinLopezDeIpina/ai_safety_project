import modal

data_vol = modal.Volume.from_name("behavior-datasets", create_if_missing=True)

with data_vol.batch_upload(force=True) as batch:
    batch.put_directory("/home/martin/LLMs_Encode_Harmfulness_Refusal_Separately/data", "/")
