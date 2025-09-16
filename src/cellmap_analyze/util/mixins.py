class ComputeConfigMixin:
    def __init__(self, num_workers: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_workers = num_workers
        if num_workers == 1:
            self.compute_args = {"scheduler": "single-threaded", "retries": 5}
        else:
            self.compute_args = {"retries": 5}
