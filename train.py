import argparse
import random
import numpy as np
import torch
import yaml
import os

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

def load_configs():
    cfg = {}
    for name, path in [
        ('model',   'configs/model.yaml'),
        ('training','configs/train.yaml'),
        ('dataset', 'configs/dataset.yaml'),
    ]:
        with open(path, 'r') as f:
            cfg.update(yaml.safe_load(f))
    return cfg

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', default='cuda',
                        help='cuda or cpu')
    return parser.parse_args()

def main():
    args = parse_args()
    cfg  = load_configs()

    set_seed(cfg['training']['seed'])

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    from datasets.fvg_b import build_fvgb_dataloaders
    loaders = build_fvgb_dataloaders(cfg)

    num_classes = loaders['num_classes']
    print(f"Training identities: {num_classes}")
    print(f"Train batches:       {len(loaders['train'])}")
    print(f"Val batches:         {len(loaders['val'])}")
    print(f"Test subjects:       {len(loaders['test_ids'])}")
    for split_name, protocols in [('1%', loaders['protocols_1pct']),
                                   ('5%', loaders['protocols_5pct'])]:
        for pname, pdata in protocols.items():
            print(f"  [{split_name}] {pname}: "
                  f"gallery={len(pdata['gallery'].dataset)}  "
                  f"probe={len(pdata['probe'].dataset)}")

    cfg['model']['identity']['num_classes'] = num_classes
    cfg['model']['gender']['num_classes']   = cfg['dataset']['gender_classes']

    from models.biokinematic_net import BioKinematicNet
    model = BioKinematicNet(cfg['model']).to(device)

    breakdown = model.count_parameters()
    print("\nParameter breakdown:")
    for k, v in breakdown.items():
        print(f"  {k:<20} {v:>10,}")

    opt_cfg  = cfg['training']['optimizer']
    sch_cfg  = cfg['training']['scheduler']

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=opt_cfg['lr'],
        weight_decay=opt_cfg['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=sch_cfg['T_max'],
        eta_min=sch_cfg['eta_min'],
    )

    from losses.combined_loss import CombinedLoss
    loss_w   = cfg['training']['loss_weights']
    loss_fn  = CombinedLoss(
        w_identity=loss_w['identity'],
        w_triplet=loss_w['triplet'],
        w_gender=loss_w['gender'],
        triplet_margin=cfg['training']['triplet']['margin'],
        num_classes=num_classes,
    )

    from trainers.trainer import Trainer
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        cfg=cfg,
        device=device,
    )

    start_epoch = 1
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume) + 1

    epochs = cfg['training']['epochs']
    print(f"\nStarting training from epoch {start_epoch} to {epochs}\n")

    for epoch in range(start_epoch, epochs + 1):
        train_losses = trainer.train_epoch(epoch)
        val_losses = trainer.val_epoch(epoch)

        scheduler.step()

        print(
            f"\nEpoch {epoch:03d}/{epochs} Summary | "
            f"LR={scheduler.get_last_lr()[0]:.6f}\n"
            f"  Train — total={train_losses['total']:.4f}  "
            f"id={train_losses['identity']:.4f}  "
            f"tri={train_losses['triplet']:.4f}  "
            f"gen={train_losses['gender']:.4f}\n"
            f"  Val   — total={val_losses['total']:.4f}  "
            f"id={val_losses['identity']:.4f}  "
            f"tri={val_losses['triplet']:.4f}  "
            f"gen={val_losses['gender']:.4f}\n"
        )

        is_best = val_losses['total'] < trainer.best_val_loss
        if is_best:
            trainer.best_val_loss = val_losses['total']
        trainer.save_checkpoint(epoch, val_losses, is_best=is_best)

    print("Training complete.")

if __name__ == '__main__':
    main()
