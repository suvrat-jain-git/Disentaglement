import os
import csv
import random
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

GALLERY_SEED = 0

GALLERY_SEQ = '02'

PROTOCOLS = {
    'WS': {
        'gallery_sessions': [1, 2],
        'probe': {
            1: ['04','05','06','07','08','09'],
            2: ['04','05','06'],
        },
    },
    'BGHT': {
        'gallery_sessions': [1],
        'probe': {
            1: ['10','11','12'],
        },
    },
    'CL': {
        'gallery_sessions': [2],
        'probe': {
            2: ['07','08','09'],
        },
    },
    'MP': {
        'gallery_sessions': [2],
        'probe': {
            2: ['10','11','12'],
        },
    },
    'ALL': {
        'gallery_sessions': [1, 2],
        'probe': {
            1: ['01','03','04','05','06','07','08','09','10','11','12'],
            2: ['01','03','04','05','06','07','08','09','10','11','12'],
            3: ['01','02','03','04','05','06','07','08','09','10','11','12'],
        },
    },
}

def _load_id_list(path):
    with open(path, 'r') as f:
        ids = [int(line.strip()) for line in f if line.strip()]
    return sorted(ids)

def _load_gender_map(path):
    gender_map = {}
    with open(path, 'r') as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            sid    = int(row[0].strip())
            gender = 0 if row[1].strip().upper() == 'M' else 1
            gender_map[sid] = gender
    return gender_map

def _infer_session(subject_id):
    if subject_id <= 147:
        return 1
    return 2

def _collect_sequences(root, subject_ids, sessions=None, seq_ids=None):
    sil_root    = Path(root) / 'crop_sil'
    subject_set = set(subject_ids) if subject_ids is not None else None
    session_set = set(sessions)    if sessions    is not None else None
    seq_set     = set(seq_ids)     if seq_ids     is not None else None

    sequences = []

    for session_dir in sorted(sil_root.iterdir()):
        if not session_dir.is_dir() or not session_dir.name.startswith('session'):
            continue
        session_num = int(session_dir.name.replace('session', ''))
        if session_set is not None and session_num not in session_set:
            continue

        for subj_dir in sorted(session_dir.iterdir()):
            if not subj_dir.is_dir():
                continue
            try:
                sid = int(subj_dir.name)
            except ValueError:
                continue
            if subject_set is not None and sid not in subject_set:
                continue

            for seq_dir in sorted(subj_dir.iterdir()):
                if not seq_dir.is_dir():
                    continue
                if seq_set is not None and seq_dir.name not in seq_set:
                    continue
                frames = sorted(seq_dir.glob('*.png'))
                if len(frames) == 0:
                    continue
                sequences.append({
                    'subject_id': sid,
                    'session':    session_num,
                    'seq_id':     seq_dir.name,
                    'frame_dir':  seq_dir,
                })

    return sequences

def _sample_frames(frame_dir, T, training=True):
    all_frames = sorted(frame_dir.glob('*.png'))
    N = len(all_frames)

    if N == 0:
        raise RuntimeError(f"No PNG frames found in {frame_dir}")

    if N >= T:
        if training:
            start = random.randint(0, N - T)
        else:
            start = (N - T) // 2
        selected = all_frames[start: start + T]
    else:
        selected = [all_frames[i % N] for i in range(T)]

    return selected

def _gallery_probe_split(test_ids, split_pct, seed=GALLERY_SEED):
    rng = random.Random(seed)
    ids = sorted(test_ids) 
    n_gallery = max(1, int(len(ids) * split_pct))
    gallery_ids = rng.sample(ids, n_gallery)
    gallery_ids = sorted(gallery_ids)
    probe_ids   = sorted(set(ids) - set(gallery_ids))
    return gallery_ids, probe_ids

class FVGBDataset(Dataset):
    def __init__(self, root, subject_ids, id_remap, gender_map,
                 T=30, image_size=224, augment=False):
        super().__init__()
        self.gender_map = gender_map
        self.id_remap   = id_remap
        self.T          = T
        self.image_size = image_size
        self.augment    = augment

        self.sequences = _collect_sequences(root, subject_ids)

        if len(self.sequences) == 0:
            raise RuntimeError(
                f"No sequences found for given subject_ids under {root}."
            )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq    = self.sequences[idx]
        sid    = seq['subject_id']
        
        frames = _sample_frames(seq['frame_dir'], self.T, training=self.augment)

        do_flip = self.augment and (random.random() < 0.5)

        tensors = []
        for fpath in frames:
            img = Image.open(fpath).convert('L')
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
            if do_flip:
                img = TF.hflip(img)
            tensors.append(TF.to_tensor(img)) 

        sequence_tensor = torch.stack(tensors, dim=0)

        return sequence_tensor, self.id_remap[sid], self.gender_map[sid]

class FVGBGalleryDataset(Dataset):
    def __init__(self, root, gallery_ids, gender_map,
                 gallery_sessions, T=30, image_size=224):

        super().__init__()
        self.gender_map = gender_map
        self.T          = T
        self.image_size = image_size

        self.sequences = _collect_sequences(
            root, gallery_ids,
            sessions=gallery_sessions,
            seq_ids=[GALLERY_SEQ],
        )

        if len(self.sequences) == 0:
            raise RuntimeError(
                f"No gallery sequences found. sessions={gallery_sessions} seq={GALLERY_SEQ}"
            )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq    = self.sequences[idx]
        sid    = seq['subject_id']
        frames = _sample_frames(seq['frame_dir'], self.T, training=False)
        tensors = []
        for fpath in frames:
            img = Image.open(fpath).convert('L')
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
            tensors.append(TF.to_tensor(img))
        return torch.stack(tensors, dim=0), sid, self.gender_map[sid]

class FVGBProbeDataset(Dataset):
    def __init__(self, root, probe_ids, gender_map,
                 probe_sessions_seqs, T=30, image_size=224):
        
        super().__init__()
        self.gender_map = gender_map
        self.T          = T
        self.image_size = image_size

        self.sequences = []
        for session, seq_ids in probe_sessions_seqs.items():
            seqs = _collect_sequences(
                root, probe_ids,
                sessions=[session],
                seq_ids=seq_ids,
            )
            self.sequences.extend(seqs)

        if len(self.sequences) == 0:
            raise RuntimeError(
                f"No probe sequences found for probe_sessions_seqs={probe_sessions_seqs}"
            )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq    = self.sequences[idx]
        sid    = seq['subject_id']
        frames = _sample_frames(seq['frame_dir'], self.T, training=False)
        tensors = []
        for fpath in frames:
            img = Image.open(fpath).convert('L')
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
            tensors.append(TF.to_tensor(img))
        return torch.stack(tensors, dim=0), sid, self.gender_map[sid]

def build_protocol_loaders(root, test_ids, gender_map, split_pct,
                           T=30, image_size=224, batch_size=16,
                           num_workers=4):
    gallery_ids, probe_ids = _gallery_probe_split(test_ids, split_pct)

    loaders = {}
    for protocol_name, protocol_cfg in PROTOCOLS.items():
        gallery_sessions    = protocol_cfg['gallery_sessions']
        probe_sessions_seqs = protocol_cfg['probe']

        gallery_ds = FVGBGalleryDataset(
            root, gallery_ids, gender_map,
            gallery_sessions=gallery_sessions,
            T=T, image_size=image_size,
        )
        probe_ds = FVGBProbeDataset(
            root, probe_ids, gender_map,
            probe_sessions_seqs=probe_sessions_seqs,
            T=T, image_size=image_size,
        )

        gallery_loader = DataLoader(
            gallery_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )
        probe_loader = DataLoader(
            probe_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

        loaders[protocol_name] = {
            'gallery':     gallery_loader,
            'probe':       probe_loader,
            'gallery_ids': gallery_ids,
            'probe_ids':   probe_ids,
        }

    return loaders

def build_fvgb_dataloaders(cfg):
    root        = cfg['dataset']['root']
    T           = cfg['dataset']['sequence_length']
    image_size  = cfg['dataset']['image_size'][0]   
    batch_size  = cfg['training']['batch_size']
    num_workers = cfg['training']['num_workers']
    val_frac    = cfg['dataset'].get('val_fraction', 0.1)

    train_ids  = _load_id_list(os.path.join(root, 'train_id_list.txt'))
    test_ids   = _load_id_list(os.path.join(root, 'test_id_list.txt'))
    gender_map = _load_gender_map(
        os.path.join(root, 'annotated_gender_information.csv')
    )

    n_val   = max(1, int(len(train_ids) * val_frac))
    val_ids = train_ids[-n_val:]
    tr_ids  = train_ids[:-n_val]

    id_remap = {sid: i for i, sid in enumerate(tr_ids)}
    for sid in val_ids:
        if sid not in id_remap:
            id_remap[sid] = len(id_remap)
    num_classes = len(tr_ids)

    train_ds = FVGBDataset(
        root, tr_ids, id_remap, gender_map,
        T=T, image_size=image_size, augment=True,
    )
    val_ds = FVGBDataset(
        root, val_ids, id_remap, gender_map,
        T=T, image_size=image_size, augment=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    proto_kwargs = dict(
        T=T, image_size=image_size,
        batch_size=batch_size, num_workers=num_workers,
    )
    protocols_1pct = build_protocol_loaders(
        root, test_ids, gender_map, split_pct=0.01, **proto_kwargs
    )
    protocols_5pct = build_protocol_loaders(
        root, test_ids, gender_map, split_pct=0.05, **proto_kwargs
    )

    return {
        'train':          train_loader,
        'val':            val_loader,
        'protocols_1pct': protocols_1pct,
        'protocols_5pct': protocols_5pct,
        'num_classes':    num_classes,
        'id_remap':       id_remap,
        'test_ids':       test_ids,
        'gender_map':     gender_map,
    }
