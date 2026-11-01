"""Optional DataLoader settings; absence preserves the original constructor arguments."""


def loader_options(data_loading):
    workers = data_loading.get("num_workers", 0)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
        raise ValueError("data_loading.num_workers must be a non-negative integer")
    options = {"batch_size": None, "batch_sampler": None, "shuffle": False, "num_workers": workers}
    persistent = data_loading.get("persistent_workers", False)
    if not isinstance(persistent, bool):
        raise ValueError("data_loading.persistent_workers must be a boolean")
    if workers == 0:
        if persistent:
            raise ValueError("persistent_workers requires num_workers > 0")
        # PyTorch forbids prefetch_factor with a synchronous (zero-worker) loader.
        return options
    if "prefetch_factor" in data_loading:
        prefetch = data_loading.get("prefetch_factor", 2)
        if isinstance(prefetch, bool) or not isinstance(prefetch, int) or prefetch < 1:
            raise ValueError("data_loading.prefetch_factor must be a positive integer")
        options["prefetch_factor"] = prefetch
    if "persistent_workers" in data_loading:
        options["persistent_workers"] = persistent
    return options
