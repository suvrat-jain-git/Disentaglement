import numpy as np
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend for server
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _check_mpl():
    if not HAS_MPL:
        raise ImportError("matplotlib not installed. pip install matplotlib")


def plot_training_curves(
    csv_path: str,
    out_dir:  str = 'experiments/plots',
) -> None:
    """
    Plot training and validation loss curves from the CSV log.

    Produces one figure with subplots for:
        - Total loss (train + val)
        - Identity loss
        - Triplet loss
        - Gender loss
        - Orthogonality loss
        - Val gender accuracy

    Args:
        csv_path: path to training_log.csv
        out_dir:  directory to save the PNG
    """
    _check_mpl()
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas not installed. pip install pandas")

    df = pd.read_csv(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plots = [
        ('train_total',    'val_total',       'Total Loss'),
        ('train_identity', None,              'Identity Loss (train)'),
        ('train_triplet',  None,              'Triplet Loss (train)'),
        ('train_gender',   None,              'Gender Loss (train)'),
        ('train_adversarial', None,           'Orthogonality Loss (train)'),
        (None,             'val_gender_acc',  'Val Gender Accuracy'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for ax, (train_col, val_col, title) in zip(axes, plots):
        if train_col and train_col in df.columns:
            ax.plot(df['epoch'], df[train_col], label='Train', color='steelblue')
        if val_col and val_col in df.columns:
            ax.plot(df['epoch'], df[val_col], label='Val',
                    color='tomato', alpha=0.8)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle('BioKinematicNet Training Curves', fontsize=13, y=1.01)
    plt.tight_layout()
    out_path = out_dir / 'training_curves.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {out_path}")


def plot_cmc_curves(
    cmc_dict: dict,
    out_dir:  str = 'experiments/plots',
    max_rank: int = 20,
) -> None:
    """
    Plot CMC curves for multiple protocols on one figure.

    Args:
        cmc_dict: {protocol_name: np.ndarray of shape [max_rank]}
        out_dir:  directory to save PNG
        max_rank: x-axis limit
    """
    _check_mpl()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    colours = ['steelblue', 'tomato', 'seagreen', 'darkorange', 'purple']
    fig, ax  = plt.subplots(figsize=(8, 6))

    for (name, cmc), colour in zip(cmc_dict.items(), colours):
        ranks = np.arange(1, len(cmc) + 1)
        ax.plot(ranks, cmc * 100, label=name, color=colour, linewidth=2)
        ax.scatter(1, cmc[0] * 100, color=colour, zorder=5, s=40)

    ax.set_xlabel('Rank', fontsize=12)
    ax.set_ylabel('Recognition Rate (%)', fontsize=12)
    ax.set_title('CMC Curves — BioKinematicNet', fontsize=13)
    ax.set_xlim(1, max_rank)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    out_path = out_dir / 'cmc_curves.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"CMC curves saved to {out_path}")


def plot_gender_confusion(
    preds:   np.ndarray,
    labels:  np.ndarray,
    out_dir: str = 'experiments/plots',
) -> None:
    """
    Plot gender classification confusion matrix.

    Args:
        preds:   [N] predicted gender labels
        labels:  [N] true gender labels
        out_dir: output directory
    """
    _check_mpl()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = ['Male', 'Female']
    cm = np.zeros((2, 2), dtype=int)
    for p, l in zip(preds, labels):
        cm[l, p] += 1

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im)

    ax.set_xticks([0, 1]); ax.set_xticklabels(classes)
    ax.set_yticks([0, 1]); ax.set_yticklabels(classes)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title('Gender Classification Confusion Matrix')

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black',
                    fontsize=14)

    plt.tight_layout()
    out_path = out_dir / 'gender_confusion.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Gender confusion matrix saved to {out_path}")


def plot_embedding_tsne(
    embeddings: np.ndarray,
    labels:     np.ndarray,
    label_type: str = 'identity',
    out_dir:    str = 'experiments/plots',
    n_subjects: int = 20,
) -> None:
    """
    Plot t-SNE of identity embeddings coloured by subject or gender.

    Args:
        embeddings: [N, D] feature matrix
        labels:     [N]    subject IDs or gender labels
        label_type: 'identity' or 'gender'
        out_dir:    output directory
        n_subjects: max subjects to plot (identity mode only)
    """
    _check_mpl()
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        raise ImportError("scikit-learn not installed. pip install scikit-learn")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Subset for identity mode
    if label_type == 'identity':
        unique = np.unique(labels)[:n_subjects]
        mask   = np.isin(labels, unique)
        embeddings = embeddings[mask]
        labels     = labels[mask]

    print("Running t-SNE...")
    tsne   = TSNE(n_components=2, random_state=42, perplexity=30)
    coords = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))
    unique_labels = np.unique(labels)
    cmap = matplotlib.colormaps.get_cmap('tab20').resampled(len(unique_labels))

    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        name = f'Subject {lbl}' if label_type == 'identity' \
               else ['Male', 'Female'][lbl]
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[cmap(i)], label=name, alpha=0.7, s=20)

    ax.set_title(f't-SNE — coloured by {label_type}', fontsize=13)
    ax.axis('off')
    if label_type == 'gender':
        ax.legend(fontsize=11)

    out_path = out_dir / f'tsne_{label_type}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"t-SNE plot saved to {out_path}")
