import numpy as np
import torch
import scipy
from torch.utils.data import Dataset, Subset, DataLoader
import torch
import copy
from torchvision import datasets, transforms
from collections import Counter


class LocalDataset(Dataset):
    """
    because torch.dataloader need override __getitem__() to iterate by index
    this class is map the index to local dataloader into the whole dataloader
    """

    def __init__(self, dataset, Dict):
        self.dataset = dataset
        self.idxs = [int(i) for i in Dict]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        X, y = self.dataset[self.idxs[item]]
        return X, y


# Original
def LocalDataloaders(dataset, dict_users, batch_size, ShuffleorNot = True, BatchorNot = True, frac = 1):
    """
    dataset: the same dataset object
    dict_users: dictionary of index of each local model
    batch_size: batch size for each dataloader
    ShuffleorNot: Shuffle or Not
    BatchorNot: if False, the dataloader will give the full length of data instead of a batch, for testing
    """
    num_users = len(dict_users)
    loaders = []
    for i in range(num_users):
        num_data = len(dict_users[i])
        frac_num_data = int(frac*num_data)
        whole_range = range(num_data)
        frac_range = np.random.choice(whole_range, frac_num_data)
        frac_dict_users = [dict_users[i][j] for j in frac_range]
        if BatchorNot== True:
            loader = torch.utils.data.DataLoader(
                        LocalDataset(dataset,frac_dict_users),
                        batch_size=batch_size,
                        shuffle = ShuffleorNot,
                        num_workers=0,
                        drop_last=True)
        else:
            loader = torch.utils.data.DataLoader(
                        LocalDataset(dataset,frac_dict_users),
                        batch_size=len(LocalDataset(dataset,dict_users[i])),
                        shuffle = ShuffleorNot,
                        num_workers=0,
                        drop_last=True)
        loaders.append(loader)
    return loaders

# Test1
# def LocalDataloaders(
#     dataset,
#     dict_users,
#     batch_size,
#     ShuffleorNot=True,
#     BatchorNot=True,
#     frac=1,
#     rand_seed=42,
# ):
#     """
#     Modified LocalDataloaders (FedHKD-style):
#     - Applies reproducible subsampling using 'frac'
#     - Computes per-client class distributions
#     - Returns list of DataLoaders (train or test style)
#     """
#     num_users = len(dict_users)
#     loaders = []
#     client_class_distributions = []

#     if hasattr(dataset, "targets"):
#         y = np.array(dataset.targets)
#     elif hasattr(dataset, "labels"):
#         y = np.array(dataset.labels)
#     else:
#         raise ValueError("Dataset must have 'targets' or 'labels' attribute.")

#     num_classes = len(set(y))

#     for i in range(num_users):
#         np.random.seed(rand_seed + i)
#         num_data = len(dict_users[i])
#         frac_num_data = int(frac * num_data)
#         frac_indices = np.random.choice(num_data, frac_num_data, replace=False)
#         frac_dict_users = [dict_users[i][j] for j in frac_indices]

#         client_labels = [y[idx] for idx in frac_dict_users]
#         class_counts = Counter(client_labels)
#         distribution = {
#             cls: class_counts.get(cls, 0) / len(client_labels)
#             for cls in range(num_classes)
#         }
#         client_class_distributions.append(distribution)

#         g_train = torch.Generator().manual_seed(rand_seed + i)

#         loader = DataLoader(
#             Subset(dataset, frac_dict_users),
#             batch_size=batch_size if BatchorNot else len(frac_dict_users),
#             shuffle=ShuffleorNot,
#             generator=g_train,
#             num_workers=0,
#             drop_last=True,
#         )
#         loaders.append(loader)

#     print("Local dataloaders created successfully.")
#     for i, dist in enumerate(client_class_distributions):
#         print(f"Client {i} class distribution:")
#         for cls in range(num_classes):
#             print(f"  Class {cls}: {dist.get(cls, 0):.2f}")
#     print()

#     return loaders


# Original
def partition_data(n_users, alpha=0.5,rand_seed = 0, dataset = 'cifar10'):
    if dataset == 'CIFAR10':
        K = 10
        data_dir = '../data/cifar10/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                          transform=apply_transform)
        y_train = np.array(train_dataset.targets)
        y_test = np.array(test_dataset.targets)

    if dataset == 'CIFAR100':
        K = 100
        data_dir = '../data/cifar100/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.CIFAR100(data_dir, train=True, download=True,
                                       transform=apply_transform)
        test_dataset = datasets.CIFAR100(data_dir, train=False, download=True,
                                      transform=apply_transform)
        y_train = np.array(train_dataset.targets)
        y_test = np.array(test_dataset.targets)

    if dataset == 'EMNIST':
        K = 62
        data_dir = '../data/EMNIST/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5), (0.5))])
        train_dataset = datasets.EMNIST(data_dir, train=True, split = 'byclass', download=True,
                                       transform=apply_transform)
        test_dataset = datasets.EMNIST(data_dir, train=False, split = 'byclass', download=True,
                                      transform=apply_transform)
        y_train = np.array(train_dataset.targets)
        y_test = np.array(test_dataset.targets)
    if dataset == 'SVHN':
        K = 10
        data_dir = '../data/SVHN/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.SVHN(data_dir, split='train', download=True,
                                       transform=apply_transform)
        test_dataset = datasets.SVHN(data_dir, split='test', download=True,
                                      transform=apply_transform)
        y_train = np.array(train_dataset.labels)
        y_test = np.array(test_dataset.labels)

    min_size = 0
    N = len(train_dataset)
    N_test = len(test_dataset)
    net_dataidx_map = {}
    net_dataidx_map_test = {}
    np.random.seed(rand_seed)

    while min_size < 10:
        idx_batch = [[] for _ in range(n_users)]
        idx_batch_test = [[] for _ in range(n_users)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            idx_k_test = np.where(y_test == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, n_users))
            ## Balance
            proportions_train = np.array([p*(len(idx_j)<N/n_users) for p,idx_j in zip(proportions,idx_batch)])
            proportions_test = np.array([p*(len(idx_j)<N_test/n_users) for p,idx_j in zip(proportions,idx_batch_test)])
            proportions_train = proportions_train/proportions_train.sum()
            proportions_test = proportions_test/proportions_test.sum()
            proportions_train = (np.cumsum(proportions_train)*len(idx_k)).astype(int)[:-1]
            proportions_test = (np.cumsum(proportions_test)*len(idx_k_test)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions_train))]
            idx_batch_test = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch_test,np.split(idx_k_test,proportions_test))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(n_users):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]
        net_dataidx_map_test[j] = idx_batch_test[j]


    return (train_dataset, test_dataset,net_dataidx_map, net_dataidx_map_test)


# Test1
# def partition_data(n_users, alpha=0.5, rand_seed=0, dataset="CIFAR10"):
#     """
#     Modified partition_data (FedHKD-style):
#     - Uses Dirichlet sampling with balancing
#     - Returns train/test datasets + per-client index mappings
#     """
#     torch.manual_seed(rand_seed)
#     np.random.seed(rand_seed)

#     # Dataset selection
#     if dataset.upper() == "CIFAR10":
#         K = 10
#         data_dir = "./data/cifar10/"
#         transform = transforms.Compose(
#             [
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#             ]
#         )
#         train_dataset = datasets.CIFAR10(
#             data_dir, train=True, download=True, transform=transform
#         )
#         test_dataset = datasets.CIFAR10(
#             data_dir, train=False, download=True, transform=transform
#         )
#         y_train = np.array(train_dataset.targets)
#         y_test = np.array(test_dataset.targets)

#     elif dataset.upper() == "CIFAR100":
#         K = 100
#         data_dir = "./data/cifar100/"
#         transform = transforms.Compose(
#             [
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#             ]
#         )
#         train_dataset = datasets.CIFAR100(
#             data_dir, train=True, download=True, transform=transform
#         )
#         test_dataset = datasets.CIFAR100(
#             data_dir, train=False, download=True, transform=transform
#         )
#         y_train = np.array(train_dataset.targets)
#         y_test = np.array(test_dataset.targets)

#     else:
#         raise ValueError(f"Unsupported dataset: {dataset}")

#     N = len(train_dataset)
#     N_test = len(test_dataset)
#     min_size = 0
#     net_dataidx_map = {}
#     net_dataidx_map_test = {}

#     while min_size < 10:
#         idx_batch = [[] for _ in range(n_users)]
#         idx_batch_test = [[] for _ in range(n_users)]

#         for k in range(K):
#             idx_k = np.where(y_train == k)[0]
#             idx_k_test = np.where(y_test == k)[0]
#             np.random.shuffle(idx_k)
#             np.random.shuffle(idx_k_test)

#             proportions = np.random.dirichlet(np.repeat(alpha, n_users))
#             proportions_train = np.array(
#                 [
#                     p * (len(idx_j) < N / n_users)
#                     for p, idx_j in zip(proportions, idx_batch)
#                 ]
#             )
#             proportions_test = np.array(
#                 [
#                     p * (len(idx_j) < N_test / n_users)
#                     for p, idx_j in zip(proportions, idx_batch_test)
#                 ]
#             )

#             proportions_train /= proportions_train.sum()
#             proportions_test /= proportions_test.sum()

#             train_splits = (np.cumsum(proportions_train) * len(idx_k)).astype(int)[:-1]
#             test_splits = (np.cumsum(proportions_test) * len(idx_k_test)).astype(int)[
#                 :-1
#             ]

#             idx_batch = [
#                 idx_j + idx.tolist()
#                 for idx_j, idx in zip(idx_batch, np.split(idx_k, train_splits))
#             ]
#             idx_batch_test = [
#                 idx_j + idx.tolist()
#                 for idx_j, idx in zip(idx_batch_test, np.split(idx_k_test, test_splits))
#             ]

#         min_size = min(len(idx_j) for idx_j in idx_batch)

#     for j in range(n_users):
#         np.random.shuffle(idx_batch[j])
#         np.random.shuffle(idx_batch_test[j])
#         net_dataidx_map[j] = idx_batch[j]
#         net_dataidx_map_test[j] = idx_batch_test[j]

#     print(
#         f"Partitioning complete for {dataset} with {n_users} clients and alpha={alpha}."
#     )
#     return train_dataset, test_dataset, net_dataidx_map, net_dataidx_map_test


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}
    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    return net_cls_counts


def load_and_partition_data(num_clients, dataset, alpha, batch_size, frac, rand_seed=42):
    # Applying standard CIFAR-10 transform as in original function
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Setting seeds for reproducibility, matching original
    torch.manual_seed(rand_seed)
    np.random.seed(rand_seed)

    if dataset.upper() == "CIFAR10":
        num_classes = 10
        data_dir = "./data/cifar10/"
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        train_dataset = datasets.CIFAR10(
            data_dir, train=True, download=True, transform=transform
        )
        test_dataset = datasets.CIFAR10(
            data_dir, train=False, download=True, transform=transform
        )
        y_train = np.array(train_dataset.targets)
        y_test = np.array(test_dataset.targets)

    elif dataset.upper() == "CIFAR100":
        num_classes = 100
        data_dir = "./data/cifar100/"
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        train_dataset = datasets.CIFAR100(
            data_dir, train=True, download=True, transform=transform
        )
        test_dataset = datasets.CIFAR100(
            data_dir, train=False, download=True, transform=transform
        )
        y_train = np.array(train_dataset.targets)
        y_test = np.array(test_dataset.targets)


    N = len(train_dataset)  # Total training samples (50,000 for CIFAR-10)
    N_test = len(test_dataset)  # Total test samples (10,000 for CIFAR-10)

    # Initializing index maps for training and validation (test) sets, as in FedHKD
    net_dataidx_map = {}
    net_dataidx_map_test = {}

    # Partitioning using FedHKD's Dirichlet-based method with balancing
    min_size = 0
    while min_size < 10:  # Ensuring each client has at least 10 training samples
        idx_batch = [[] for _ in range(num_clients)]
        idx_batch_test = [[] for _ in range(num_clients)]
        for k in range(num_classes):
            idx_k = np.where(y_train == k)[0]
            idx_k_test = np.where(y_test == k)[0]
            np.random.shuffle(idx_k)  # Shuffling training indices
            np.random.shuffle(idx_k_test)  # Shuffling test indices
            # Sampling proportions from Dirichlet distribution
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            # Balancing: Zero out proportions for clients with enough samples
            proportions_train = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
            proportions_test = np.array([p * (len(idx_j) < N_test / num_clients) for p, idx_j in zip(proportions, idx_batch_test)])
            proportions_train = proportions_train / proportions_train.sum()  # Normalizing
            proportions_test = proportions_test / proportions_test.sum()  # Normalizing
            proportions_train = (np.cumsum(proportions_train) * len(idx_k)).astype(int)[:-1]
            proportions_test = (np.cumsum(proportions_test) * len(idx_k_test)).astype(int)[:-1]
            # Splitting indices across clients
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions_train))]
            idx_batch_test = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_test, np.split(idx_k_test, proportions_test))]
        min_size = min([len(idx_j) for idx_j in idx_batch])  # Checking minimum size

    # Assigning indices to clients
    for j in range(num_clients):
        np.random.shuffle(idx_batch[j])
        np.random.shuffle(idx_batch_test[j])
        net_dataidx_map[j] = idx_batch[j]
        net_dataidx_map_test[j] = idx_batch_test[j]

    # Creating client datasets and loaders, mimicking LocalDataloaders
    client_train_loaders = []
    client_val_loaders = []
    client_class_distributions = []

    for i in range(num_clients):
        # Setting seed for client-specific subsampling
        np.random.seed(rand_seed + i)

        # Subsampling training indices (frac=args.part)
        num_data = len(net_dataidx_map[i])
        frac_num_data = int(frac * num_data)
        frac_indices = np.random.choice(num_data, frac_num_data, replace=False)
        train_indices = [net_dataidx_map[i][j] for j in frac_indices]

        # Subsampling validation (test) indices (frac=2*args.part)
        num_data_test = len(net_dataidx_map_test[i])
        frac_num_data_test = int(min(2 * frac, 1.0) * num_data_test)  # Ensuring frac <= 1
        frac_indices_test = np.random.choice(num_data_test, frac_num_data_test, replace=False)
        val_indices = [net_dataidx_map_test[i][j] for j in frac_indices_test]

        # Calculating class distribution for training set
        client_labels = [y_train[idx] for idx in train_indices]
        class_counts = Counter(client_labels)
        distribution = {cls: class_counts.get(cls, 0) / len(client_labels) if len(client_labels) > 0 else 0 for cls in range(num_classes)}
        client_class_distributions.append(distribution)

        # Creating subset datasets
        client_train_dataset = Subset(full_dataset, train_indices)
        client_val_dataset = Subset(test_dataset, val_indices)

        # Creating data loaders with fixed seeds
        g_train = torch.Generator().manual_seed(rand_seed + i)
        g_val = torch.Generator().manual_seed(rand_seed + i + num_clients)

        train_loader = DataLoader(client_train_dataset, batch_size=batch_size,
                                 shuffle=True, generator=g_train, drop_last=True)
        val_loader = DataLoader(client_val_dataset, batch_size=batch_size,
                                shuffle=True, generator=g_val, drop_last=True)

        client_train_loaders.append(train_loader)
        client_val_loaders.append(val_loader)

    # Creating global test loader, matching FedHKD's global_loader_test
    g_test = torch.Generator().manual_seed(rand_seed + 2 * num_clients + 1)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                             generator=g_test, num_workers=2)

    # Printing class distributions, as in original
    print("Data partitioning complete.")
    for i, dist in enumerate(client_class_distributions):
        print(f"Client {i} class distribution:")
        for cls in range(num_classes):
            print(f"  Class {cls}: {dist.get(cls, 0):.2f}")

    # Resetting random seeds
    torch.manual_seed(rand_seed)
    np.random.seed(rand_seed)

    return client_train_loaders, client_val_loaders, test_loader, client_class_distributions
