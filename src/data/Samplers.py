import numpy as np
from torch.utils.data import Sampler


class IdentityBalancedSampler(Sampler):
    """
    Samples batches such that each batch contains P unique people 
    and K samples per person.
    
    Args:
        data_source (Dataset): Your PyTorch dataset. Must have a .person_ids attribute
                               (or access to the list of IDs).
        batch_size (int): Total batch size (P * K).
        samples_per_person (int): Number of samples (K) per unique person in a batch.
    """
    def __init__(self, data_source, batch_size, samples_per_person=4):
        self.data_source = data_source
        self.batch_size = batch_size
        self.samples_per_person = samples_per_person
        self.num_identities_per_batch = batch_size // samples_per_person
        
        # 1. Group indices by Person ID
        self.indices_by_person = {}
        
        # We access the person_ids list directly from the dataset attributes
        # Ensure your ImagePairDataset class has 'self.person_ids'
        if not hasattr(data_source, 'person_ids'):
             raise AttributeError("Dataset must have 'person_ids' attribute for IdentityBalancedSampler")

        for idx, person_id in enumerate(data_source.person_ids):
            if person_id not in self.indices_by_person:
                self.indices_by_person[person_id] = []
            self.indices_by_person[person_id].append(idx)
            
        self.person_ids = list(self.indices_by_person.keys())
        
        # Number of batches per epoch
        self.num_batches = len(data_source) // batch_size

    def __iter__(self):
        # Create a shuffled list of people for this epoch.
        # We ensure the pool is large enough to cover the required number of batches.
        # Calculation: (Total Batches * People Per Batch) / Total Unique People
        multiplier = (self.num_batches * self.num_identities_per_batch) // len(self.person_ids) + 1
        people_pool = self.person_ids * multiplier
        np.random.shuffle(people_pool)
        
        iter_ptr = 0
        
        for _ in range(self.num_batches):
            batch_indices = []
            
            # Select P unique people for this batch
            selected_people = people_pool[iter_ptr : iter_ptr + self.num_identities_per_batch]
            iter_ptr += self.num_identities_per_batch
            
            for person_id in selected_people:
                # Select K random pair indices for this specific person
                available_indices = self.indices_by_person[person_id]
                
                # If a person has fewer than K pairs, replace=True allows duplicates to fill the gap.
                # If they have enough, replace=False ensures unique pairs.
                selected_indices = np.random.choice(
                    available_indices, 
                    self.samples_per_person, 
                    replace=(len(available_indices) < self.samples_per_person)
                )
                batch_indices.extend(selected_indices)

            for idx in batch_indices:
                yield idx

    def __len__(self):
        return len(self.data_source)
