import torch.optim as optim

class LRSchedulerFactory:
    @staticmethod
    def create(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6):
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            min_lr=min_lr
        )
