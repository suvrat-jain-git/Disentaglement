import os
import time
import torch


class Trainer:

    def __init__(self, model, loss_fn, optimizer, scheduler,
                 train_loader, val_loader, cfg, device):
        """
        Args:
            model:        BioKinematicNet instance
            loss_fn:      CombinedLoss instance
            optimizer:    torch optimizer
            scheduler:    torch lr scheduler
            train_loader: DataLoader — yields (frames, id_labels, gender_labels)
            val_loader:   DataLoader — same format
            cfg:          full config dict — uses cfg['training'] keys
            device:       torch.device
        """
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

        # Track best validation loss for checkpointing
        self.best_val_loss = float('inf')

    # ── Training epoch ─────────────────────────────────────────────────────

    def train_epoch(self, epoch):
        """
        Run one full training epoch.

        Args:
            epoch: current epoch number (for logging)

        Returns:
            dict of average losses over the epoch:
                total, identity, triplet, gender,
                mean_pos_dist, mean_neg_dist
        """
        self.model.train()

        # Accumulators for epoch-level averages
        accum = {
            'total': 0.0, 'identity': 0.0, 'triplet': 0.0,
            'gender': 0.0, 'adversarial': 0.0,
            'mean_pos_dist': 0.0, 'mean_neg_dist': 0.0,
            }
        n_batches = 0
        t_start   = time.time()

        # GRL lambda schedule: 0 at epoch 1, linearly increases to max
        max_epochs = self.cfg['training']['epochs']
        grl_max    = self.cfg['training'].get('grl_lambda_max', 0.1)
        grl_lambda = grl_max * ((epoch - 1) / max_epochs)
        self.model.set_grl_lambda(grl_lambda)

        for batch_idx, (frames, id_labels, gender_labels) in enumerate(self.train_loader):
            # Move to device
            # frames:        [B, T, 1, 224, 224]
            # id_labels:     [B]
            # gender_labels: [B]
            frames        = frames.to(self.device)
            id_labels     = id_labels.to(self.device)
            gender_labels = gender_labels.to(self.device)

            # Forward pass — training mode returns full output dict
            output = self.model(frames, mode='train')

            # Compute combined loss
            losses = self.loss_fn(output, id_labels, gender_labels)

            # Backward pass
            self.optimizer.zero_grad()
            losses['total'].backward()

            # Gradient clipping — prevents exploding gradients from
            # the 3D CNN early in training when features are noisy
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)

            self.optimizer.step()

            # Accumulate for epoch average
            for k in accum:
                val = losses[k]
                accum[k] += val.item() if hasattr(val, 'item') else val
            n_batches += 1

            # Per-batch logging
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

        # Epoch averages
        avg = {k: v / n_batches for k, v in accum.items()}
        return avg

    # ── Validation epoch ───────────────────────────────────────────────────

    def val_epoch(self, epoch):
        """
        Run one full validation epoch (no gradients, no augmentation).

        Args:
            epoch: current epoch number (for logging)

        Returns:
            dict of average losses over the validation set
        """
        self.model.eval()

        accum = {
            'total': 0.0, 'identity': 0.0, 'triplet': 0.0,
            'gender': 0.0, 'adversarial': 0.0,
            'mean_pos_dist': 0.0, 'mean_neg_dist': 0.0,
            }
        n_batches = 0

        # Collect all predictions across the full val set before
        # computing gender CE. This avoids single-class batch artifacts
        # where a batch happens to contain only one gender, causing CE
        # to spike to arbitrarily large values unrelated to model quality.
        all_gender_logits = []
        all_gender_labels = []
        all_id_logits     = []
        all_id_labels     = []
        all_embeddings    = []

        with torch.no_grad():
            for frames, id_labels, gender_labels in self.val_loader:
                frames        = frames.to(self.device)
                id_labels     = id_labels.to(self.device)
                gender_labels = gender_labels.to(self.device)

                output = self.model(frames, mode='train')

                all_gender_logits.append(output['gender_logits'])
                all_gender_labels.append(gender_labels)
                all_id_logits.append(output['id_logits'])
                all_id_labels.append(id_labels)
                all_embeddings.append(output['embedding'])

        # Concatenate across all val batches
        all_gender_logits = torch.cat(all_gender_logits, dim=0)
        all_gender_labels = torch.cat(all_gender_labels, dim=0)
        all_id_logits     = torch.cat(all_id_logits,     dim=0)
        all_id_labels     = torch.cat(all_id_labels,     dim=0)
        all_embeddings    = torch.cat(all_embeddings,    dim=0)

        # Compute losses over the full val set — stable and correct
        l_identity = self.loss_fn.ce_identity(all_id_logits, all_id_labels)
        l_triplet, triplet_stats = self.loss_fn.triplet(
            all_embeddings, all_id_labels
        )
        gender_w = self.loss_fn.gender_weights.to(all_gender_logits.device)
        l_gender = torch.nn.functional.cross_entropy(
            all_gender_logits, all_gender_labels, weight=gender_w
        )
        # Adversarial loss not computed on val
        l_total = (self.loss_fn.w_identity * l_identity
                 + self.loss_fn.w_triplet  * l_triplet
                 + self.loss_fn.w_gender   * l_gender)

        # Gender accuracy — useful additional val metric
        gender_preds   = all_gender_logits.argmax(dim=1)
        gender_acc     = (gender_preds == all_gender_labels).float().mean().item()

        avg = {
            'total':          l_total.item(),
            'identity':       l_identity.item(),
            'triplet':        l_triplet.item(),
            'gender':         l_gender.item(),
            'adversarial':    0.0,   # not computed on val
            'gender_acc':     gender_acc,
            'mean_pos_dist':  triplet_stats['mean_pos_dist'],
            'mean_neg_dist':  triplet_stats['mean_neg_dist'],
        }
        return avg

    # ── Checkpointing ──────────────────────────────────────────────────────

    def save_checkpoint(self, epoch, val_losses, is_best=False):
        """
        Save model + optimizer + scheduler state.

        Always saves a periodic checkpoint every save_every epochs.
        Additionally saves best.pth whenever val_loss improves.

        Args:
            epoch:      current epoch number
            val_losses: dict of validation losses
            is_best:    if True, also save as best.pth
        """
        state = {
            'epoch':           epoch,
            'model_state':     self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'val_losses':      val_losses,
            'best_val_loss':   self.best_val_loss,
        }

        # Periodic checkpoint
        if epoch % self.save_every == 0:
            path = os.path.join(self.save_dir, f'epoch_{epoch:03d}.pth')
            torch.save(state, path)
            print(f"Checkpoint saved: {path}")

        # Best checkpoint
        if is_best:
            path = os.path.join(self.save_dir, 'best.pth')
            torch.save(state, path)
            print(f"Best checkpoint saved: {path}  (val_loss={val_losses['total']:.4f})")

    def load_checkpoint(self, path):
        """
        Load checkpoint from path. Restores model, optimizer, scheduler.

        Args:
            path: path to .pth checkpoint file

        Returns:
            epoch: the epoch at which this checkpoint was saved
        """
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state['model_state'])
        self.optimizer.load_state_dict(state['optimizer_state'])
        self.scheduler.load_state_dict(state['scheduler_state'])
        self.best_val_loss = state.get('best_val_loss', float('inf'))
        print(f"Resumed from {path}  (epoch {state['epoch']})")
        return state['epoch']
