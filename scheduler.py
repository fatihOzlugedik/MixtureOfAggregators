from torch import optim

class BuildScheduler:
    """
    Wraps a torch.optim.lr_scheduler object but keeps track of its name.

    Example
    -------
    sched_builder = BuildScheduler(optimizer, dataloaders, args)
    scheduler = sched_builder.scheduler        # regular LR-scheduler object
    print(sched_builder.name)                  # e.g. "ReduceLROnPlateau"
    scheduler.step(...)                      # works exactly as before
    """
    def __init__(self, optimizer, args):
        self.optimizer = optimizer
        self.args = args

        self.scheduler, self.name = self._build_scheduler()

    def step(self, *step_args, **step_kwargs):
        """Pass-through for the underlying scheduler's .step()."""
        return self.scheduler.step(*step_args, **step_kwargs)

    def __getattr__(self, item):
        return getattr(self.scheduler, item)

    def _build_scheduler(self):
        """Creates and returns a (scheduler, name) tuple."""
        if self.args.scheduler == "ReduceLROnPlateau":
            mode = "max" if self.args.metric == "f1" else "min"
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode=mode, patience=3, min_lr=5e-6
            )
            name = "ReduceLROnPlateau"

        elif self.args.scheduler == "CosineAnnealingLR":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.args.ep
            )
            name = "CosineAnnealingLR"

        else:
            # Fallback: no-op scheduler that still plays nicely with training code
            class _NullScheduler:
                name = "None"

                def step(self, *_, **__):
                    pass

                def state_dict(self):
                    return {}

                def load_state_dict(self, *_):
                    pass

            scheduler = _NullScheduler()
            name = "None"

        return scheduler, name
