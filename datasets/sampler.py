import random
import collections
from torch.utils.data import Sampler


class PKSampler(Sampler):
    """
    Gender-balanced PK batch sampler.

    Each batch: P//2 male identities + P//2 female identities,
                K sequences each → batch_size = P * K.

    Args:
        dataset:    FVGBDataset (must have .sequences, .id_remap, .gender_map)
        P:          total identities per batch (must be even)
        K:          sequences per identity
        drop_last:  drop last incomplete batch
    """

    def __init__(self, dataset, P=8, K=4, drop_last=True):
        super().__init__()
        assert P % 2 == 0, f"P must be even for gender balance, got P={P}"
        self.P         = P
        self.K         = K
        self.drop_last = drop_last
        self.P_male    = P // 2
        self.P_female  = P // 2

        # Build index: {id_label -> [dataset_indices]}
        # and separate male/female label lists
        label_to_indices  = collections.defaultdict(list)
        label_to_gender   = {}

        for idx, seq in enumerate(dataset.sequences):
            sid      = seq['subject_id']
            id_label = dataset.id_remap[sid]
            gender   = dataset.gender_map[sid]
            label_to_indices[id_label].append(idx)
            label_to_gender[id_label] = gender

        # Keep only identities with >= K samples
        self.label_to_indices = {
            label: indices
            for label, indices in label_to_indices.items()
            if len(indices) >= K
        }

        # Split into male and female label pools
        self.male_labels   = sorted([
            l for l, g in label_to_gender.items()
            if g == 0 and l in self.label_to_indices
        ])
        self.female_labels = sorted([
            l for l, g in label_to_gender.items()
            if g == 1 and l in self.label_to_indices
        ])

        if len(self.male_labels) < self.P_male:
            raise RuntimeError(
                f"Need >= P/2={self.P_male} male identities with >= K={K} samples. "
                f"Only {len(self.male_labels)} qualify."
            )
        if len(self.female_labels) < self.P_female:
            raise RuntimeError(
                f"Need >= P/2={self.P_female} female identities with >= K={K} samples. "
                f"Only {len(self.female_labels)} qualify."
            )

        print(f"PKSampler ready: {len(self.male_labels)} male + "
              f"{len(self.female_labels)} female identities, "
              f"P={P} (balanced), K={K}, batch_size={P*K}")

    def __iter__(self):
        # Shuffle both pools independently each epoch
        male_pool   = self.male_labels.copy()
        female_pool = self.female_labels.copy()
        random.shuffle(male_pool)
        random.shuffle(female_pool)

        # Number of complete batches limited by the smaller pool
        n_batches = min(
            len(male_pool)   // self.P_male,
            len(female_pool) // self.P_female,
        )

        for i in range(n_batches):
            batch_male   = male_pool  [i * self.P_male   : (i+1) * self.P_male]
            batch_female = female_pool[i * self.P_female : (i+1) * self.P_female]
            batch_labels = batch_male + batch_female

            batch_indices = []
            for label in batch_labels:
                pool   = self.label_to_indices[label]
                chosen = random.choices(pool, k=self.K)
                batch_indices.extend(chosen)

            # Shuffle within batch so male/female aren't always in two halves
            random.shuffle(batch_indices)
            yield batch_indices

    def __len__(self):
        n = min(
            len(self.male_labels)   // self.P_male,
            len(self.female_labels) // self.P_female,
        )
        return n
