import os
import time
import torch

class Trainer:
    def __init__(self, model, loss_fn, optimizer, scheduler,
                 train_loader, val_loader, cfg, device):
        self.model        = model
        self.loss_fn      = loss_fn
        self.optimizer    = optimizer
        self.scheduler    = scheduler
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = cfg
        self.device       = device

        self.save_dir  = cfg['training']['checkpoint']['save_dir']
        self.save_every= cfg['training']['checkpoint']['save_every']
        self.log_every = cfg['training']['log_every']

        os.makedirs(self.save_dir, exist_ok=True)

        self.best_val_loss = float('inf')

    def train_epoch(self, epoch):
        self.model.train()

        accum = {
            'total': 0.0, 'identity': 0.0, 'triplet': 0.0,
            'gender': 0.0, 'mean_pos_dist': 0.0, 'mean_neg_dist': 0.0,
        }
        n_batches = 0
        t_start   = time.time()

        for batch_idx, (frames, id_labels, gender_labels) in enumerate(self.train_loader):
            frames        = frames.to(self.device)
            id_labels     = id_labels.to(self.device)
            gender_labels = gender_labels.to(self.device)

            output = self.model(frames, mode='train')

            losses = self.loss_fn(output, id_labels, gender_labels)

            self.optimizer.zero_grad()
            losses['total'].backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)

            self.optimizer.step()

            for k in accum:
                val = losses[k]
                accum[k] += val.item() if hasattr(val, 'item') else val
            n_batches += 1

            if (batch_idx + 1) % self.log_every == 0:
                elapsed = time.time() - t_start
                print(
                    f"Epoch {epoch:03d} | Batch {batch_idx+1:04d}/{len(self.train_loader):04d} | "
                    f"Loss {losses['total'].item():.4f} "
                    f"(id={losses['identity'].item():.4f} "
                    f"tri={losses['triplet'].item():.4f} "
                    f"gen={losses['gender'].item():.4f}) | "
                    f"pos_dist={losses['mean_pos_dist']:.3f} "
                    f"neg_dist={losses['mean_neg_dist']:.3f} | "
                    f"{elapsed:.1f}s"
                )

        avg = {k: v / n_batches for k, v in accum.items()}
        return avg

    def val_epoch(self, epoch):
        self.model.eval()

        accum = {
            'total': 0.0, 'identity': 0.0, 'triplet': 0.0,
            'gender': 0.0, 'mean_pos_dist': 0.0, 'mean_neg_dist': 0.0,
        }
        n_batches = 0

        with torch.no_grad():
            for frames, id_labels, gender_labels in self.val_loader:
                frames        = frames.to(self.device)
                id_labels     = id_labels.to(self.device)
                gender_labels = gender_labels.to(self.device)

                output = self.model(frames, mode='train')
                losses = self.loss_fn(output, id_labels, gender_labels)

                for k in accum:
                    val = losses[k]
                    accum[k] += val.item() if hasattr(val, 'item') else val
                n_batches += 1

        avg = {k: v / n_batches for k, v in accum.items()}
        return avg

    def save_checkpoint(self, epoch, val_losses, is_best=False):
        state = {
            'epoch':           epoch,
            'model_state':     self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'val_losses':      val_losses,
            'best_val_loss':   self.best_val_loss,
        }

        if epoch % self.save_every == 0:
            path = os.path.join(self.save_dir, f'epoch_{epoch:03d}.pth')
            torch.save(state, path)
            print(f"Checkpoint saved: {path}")

        if is_best:
            path = os.path.join(self.save_dir, 'best.pth')
            torch.save(state, path)
            print(f"Best checkpoint saved: {path}  (val_loss={val_losses['total']:.4f})")

    def load_checkpoint(self, path):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state['model_state'])
        self.optimizer.load_state_dict(state['optimizer_state'])
        self.scheduler.load_state_dict(state['scheduler_state'])
        self.best_val_loss = state.get('best_val_loss', float('inf'))
        print(f"Resumed from {path}  (epoch {state['epoch']})")
        return state['epoch']
