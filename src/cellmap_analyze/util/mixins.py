import os


class ComputeConfigMixin:
    def __init__(self, num_workers: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_workers = num_workers
        if num_workers == 1:
            self.compute_args = {"scheduler": "single-threaded"}
        else:
            self.compute_args = {}
