def get_device(torch_module):
    if torch_module.cuda.is_available():
        return torch_module.device("cuda")

    mps_backend = getattr(getattr(torch_module, "backends", None), "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch_module.device("mps")

    return torch_module.device("cpu")


def model_load_kwargs(torch_module, device, device_map="auto", load_in_8bit=True):
    if device.type == "cuda":
        return {"device_map": device_map, "load_in_8bit": load_in_8bit}
    if device.type == "mps":
        return {
            "low_cpu_mem_usage": True,
            "torch_dtype": torch_module.float16,
        }
    return {}


def move_model_to_device(model, device):
    if device.type != "cuda":
        return model.to(device)
    return model


def should_compile(device, platform):
    return device.type != "mps" and platform != "win32"
