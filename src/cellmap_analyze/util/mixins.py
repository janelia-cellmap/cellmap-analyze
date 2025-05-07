import os


class ComputeConfigMixin:
    def __init__(self, num_workers: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_workers = num_workers
        if num_workers == 1:
            self.compute_args = {"scheduler": "single-threaded"}
            self.num_local_threads_available = 1
            self.local_config = None
        else:
            n = len(os.sched_getaffinity(0))
            self.compute_args = {}
            self.num_local_threads_available = n
            self.local_config = {
                "jobqueue": {
                    "local": {
                        "ncpus": n,
                        "processes": n,
                        "cores": n,
                        "log-directory": "job-logs",
                        "name": "dask-worker",
                    }
                }
            }
