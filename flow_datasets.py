import networkx as nx
import numpy as np
import os
import re
import torch
from torch_geometric.data import Dataset, Data
import torch.utils.data as torch_data

def read_edge_index(filehandle):
    edge_index=[[], []]
    edge_index[0] = [int(x) for x in filehandle.readline().split()]
    edge_index[1] = [int(x) for x in filehandle.readline().split()]
    edge_index = torch.tensor(edge_index)
    return edge_index

def folder_elements(folder_name):
    if not os.path.isdir(folder_name):
        return 0
    return (len([_ for _ in os.listdir(folder_name)]))

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    def tryint(s):
        try:
            return int(s)
        except:
            return s
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

class GraphDatasetBase(Dataset):
    def __init__(self, root, device='cuda', split='train', transform=None, pre_transform=None, less_wired=False, probp=1, probq=4):
        self.device = device
        self.less_wired = less_wired
        self.probp = probp
        self.probq = probq
        lw = '_less_wired' if self.less_wired else ''
        proot = ''
        if not(probp == 1 and probq == 4):
            proot = '_'+str(probp)+'_'+str(probq)
        root += lw
        root += proot
        print("ROOT", root)
        assert split in ['train', 'val'] or 'test' in split
        self.split = split
        # print("INIT FIN")
        super(GraphDatasetBase, self).__init__(root, transform, pre_transform)
        print("PROCESSED", self.processed_dir)

    @property
    def raw_file_names(self):
        dirname = self.raw_dir+'/'+self.split
        raw_file_names = [self.split+'/'+str(x)+'.txt' for x in range(1, folder_elements(dirname)+1)]
        return raw_file_names

    @property
    def processed_file_names(self):
        # Raw dir below is not a typo
        dirname = self.raw_dir + '/' + self.split
        processed_file_names = [self.split+'/'+str(x)+'.pt' for x in range(0, folder_elements(self.processed_dir+'/'+self.split))]
        return processed_file_names

    def __len__(self):
        if not os.path.isdir('./' + self.raw_dir+ '/' + self.split):
            self.download()
        if not os.path.isdir('./' + self.processed_dir+ '/' + self.split):
            self.process()
        return (len([_ for _ in os.listdir(self.processed_dir+'/'+self.split)]))

    def process_augmenting_iteration(self, file_handle, s, n):
            metadata = {}
            metadata['weights'] = torch.tensor([float(x) for x in file_handle.readline().split()])
            metadata['capacities'] = torch.tensor([float(x) for x in file_handle.readline().split()])

            edge_attr = torch.cat((metadata['weights'].view(-1, 1), metadata['capacities'].view(-1, 1)), dim=1)

            bf = torch.tensor([float(x) for x in file_handle.readline().split()]).unsqueeze(1)
            pred = torch.tensor([float(x) for x in file_handle.readline().split()]).unsqueeze(1)
            features = torch.cat((bf, pred), dim=1).unsqueeze(1)
            target_features = None

            for i in range(s+n+1):
                target_bf = torch.tensor([float(x) for x in file_handle.readline().split()]).unsqueeze(1)
                target_pred = torch.tensor([float(x) for x in file_handle.readline().split()]).unsqueeze(1)
                y = torch.cat((target_bf, target_pred), dim=1)
                if target_features is None:
                    target_features = y.unsqueeze(1)
                else:
                    target_features = torch.cat((target_features, y.unsqueeze(1)), dim=1)

                if i != s+n-1:
                    features = torch.cat((features, y.unsqueeze(1)), dim=1)
            return edge_attr, features, target_features, metadata

    def process_graph(self, file_handle):
        s, n = [int(x) for x in file_handle.readline().split()]
        edge_index=[[], []]
        edge_index[0] = [int(x) for x in file_handle.readline().split()]
        edge_index[1] = [int(x) for x in file_handle.readline().split()]
        edge_index = torch.tensor(edge_index)
        return s, n, edge_index

    def preload(self):
        self.preloaded_data = []
        for path in self.processed_paths:
            print(path)
            self.preloaded_data.append(torch.load(path).to(self.device))

    def download(self):
        if not os.path.isdir(self.root+'/raw'):
            if not self.less_wired:
                os.system('./gen.sh 8 8 all_iter')
            else:
                os.system('./gen.sh 8 8 all_iter_less_wired'+' '+str(self.probp)+' '+str(self.probq))

    def get(self, idx):
        if not os.path.isdir(os.path.join(self.processed_dir+'/'+self.split)):
            if not os.path.isdir(os.path.join(self.raw_dir+'/'+self.split)):
                self.download()
            self.process()
        
        data = torch.load(os.path.join(self.processed_dir+'/'+self.split, '{}.pt'.format(idx))).to(self.device)
        return data

class SingleIterationDataset(GraphDatasetBase):
    def process(self):
        cnt = 0
        dirname = self.processed_dir+'/'+self.split
        print("PRP", self.raw_paths)
        for raw_path in sorted(self.raw_paths, key=alphanum_key):
            with open(raw_path, "r") as raw_file:
                s, n, edge_index = self.process_graph(raw_file)

                while True:
                    edge_attr, features, target_features, metadata = self.process_augmenting_iteration(raw_file, s, n)
                    if edge_attr.nelement() == 0:
                        break
                    metadata['reachability'] = target_features[:, :, 1] != -1

                    data = Data(features, edge_index, edge_attr, y=target_features, **metadata)
                    if not os.path.isdir(os.path.join(dirname)):
                        os.mkdir(os.path.join(dirname))
                    torch.save(data, os.path.join(dirname, '{}.pt'.format(cnt)))

                    cnt += 1

class BFSSingleIterationDataset(GraphDatasetBase):

    def __init__(self, root, device='cuda', split='train', transform=None, pre_transform=None, less_wired=False, probp=1, probq=4):
        super(BFSSingleIterationDataset, self).__init__(root, device, split, transform, pre_transform, less_wired, probp, probq)


    def process(self):
        cnt = 0
        dirname = self.processed_dir+'/'+self.split
        for raw_path in sorted(self.raw_paths, key=alphanum_key):
            with open(raw_path, "r") as raw_file:
                s, n, edge_index = self.process_graph(raw_file)

                while True:
                    edge_attr, features, target_features, metadata = self.process_augmenting_iteration(raw_file, s, n)
                    if edge_attr.nelement() == 0:
                        break
                    edge_attr[:, 1] = (metadata["capacities"] > 0).float()

                    features = (features[:, :, 1] != -1).float()
                    target_features = (target_features[:, :, 1] != -1).float()

                    data = Data(features, edge_index, edge_attr, y=target_features, **metadata)
                    if not os.path.isdir(os.path.join(dirname)):
                        os.mkdir(os.path.join(dirname))
                    torch.save(data, os.path.join(dirname, '{}.pt'.format(cnt)))

                    cnt += 1

class GraphOnlyDataset(GraphDatasetBase):

    def process(self):
        cnt = 0
        dirname = self.processed_dir+'/'+self.split
        for raw_path in sorted(self.raw_paths, key=alphanum_key):
            with open(raw_path, "r") as raw_file:
                s, n, edge_index = self.process_graph(raw_file)
                edge_attr, features, target_features, metadata = self.process_augmenting_iteration(raw_file, s, n)
                data = Data(features, edge_index, edge_attr, y=target_features, **metadata)
                if not os.path.isdir(os.path.join(dirname)):
                    os.mkdir(os.path.join(dirname))
                torch.save(data, os.path.join(dirname, '{}.pt'.format(cnt)))
                cnt += 1

class GraphOnlyDatasetBFS(GraphDatasetBase):

    def process(self):
        cnt = 0
        dirname = self.processed_dir+'/'+self.split
        for raw_path in sorted(self.raw_paths, key=alphanum_key):
            with open(raw_path, "r") as raw_file:
                s, n, edge_index = self.process_graph(raw_file)
                edge_attr, features, target_features, metadata = self.process_augmenting_iteration(raw_file, s, n)
                edge_attr[:, 1] = (metadata["capacities"] > 0).float()
                features = (features[:, :, 1] != -1).float()
                target_features = (target_features[:, :, 1] != -1).float()
                data = Data(features, edge_index, edge_attr, y=target_features, **metadata)
                if not os.path.isdir(os.path.join(dirname)):
                    os.mkdir(os.path.join(dirname))
                torch.save(data, os.path.join(dirname, '{}.pt'.format(cnt)))
                cnt += 1


#######################
#  the defualt source node is NODE 0
INF = 255
def b_f(adj):
    size = adj.size(0)
    res = list()
    predecessors = list()
    t = torch.zeros(size).fill_(INF)
    t[0] = 0
    p = torch.arange(size)  # predecessor
    res.append(t)
    predecessors.append(p)

    while True:
        #        new_t = torch.min(torch.cat([t, adj[:,i], dim=1), dim=1)
        new_t = t.clone()
        new_p = p.clone()
        for i in range(size):
            for j in range(size):
                if adj[j, i] > 1e-5:
                    #                    new_t[i] = min(new_t[i], t[j] + adj[j,i])
                    if t[j] + adj[j, i] < new_t[i]:
                        new_t[i] = t[j] + adj[j, i]
                        new_p[i] = j
        #        new_t = torch.min(adj + t.view(-1, 1), dim=0).values
        #        print(new_t)
        if torch.sum(torch.abs(t - new_t)) < 1e-5:
            break
        t = new_t
        p = new_p
        res.append(t)
        predecessors.append(p)
    return torch.stack(res), torch.stack(predecessors)

def get_random_adjacency_matrix(size):
    trivial = False
    if trivial:
        threshold = 0.3
        adj = torch.rand(size, size)
        adj = (adj + adj.T) / 2
        adj[adj < threshold] = 0
        for i in range(size):
            adj[i, i] = 0
    else:
        p = 0.6
        g = nx.generators.random_graphs.fast_gnp_random_graph(size, p)
        adj = torch.tensor(nx.to_numpy_matrix(g)).float()
        for i in range(adj.shape[0]):
            for j in range(adj.shape[1]):
                if adj[i][j] > 0:
                    adj[i][j] = np.random.uniform(0.2, 1.0)
    return adj

class BellmanFordDataset(Dataset):
    def __init__(self, root, device='cuda', split='train', transform=None, pre_transform=None, *args, **kwargs):
        self.device = device
        assert split in ['train', 'val'] or 'test' in split
        self.split = split
        if not os.path.exists(root):
            os.mkdir(root)
            os.mkdir(os.path.join(root, "raw"))
            os.mkdir(os.path.join(root, "processed"))
        super(BellmanFordDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        dirname = os.path.join(self.raw_dir, self.split)
        raw_file_names = [os.path.join(self.split, str(x)+'.pt') for x in range(0, folder_elements(dirname))]
        return raw_file_names

    @property
    def processed_file_names(self):
        processed_file_names = [os.path.join(self.split, str(x)+'.pt') for x in range(0, folder_elements(os.path.join(self.processed_dir, self.split)))]
        return processed_file_names

    def __len__(self):
        if not os.path.isdir(os.path.join('./', self.raw_dir, self.split)):
            self.download()
        if not os.path.isdir(os.path.join('./', self.processed_dir, self.split)):
            self.process()
        return (len([_ for _ in os.listdir(os.path.join(self.processed_dir, self.split))]))

    def process(self):
        cnt = 0
        dirname = os.path.join(self.processed_dir, self.split)
        for raw_path in sorted(self.raw_paths, key=alphanum_key):
            data_dict = torch.load(raw_path)
            adj = data_dict["adj"]
            num_nodes = adj.shape[0]
            values = data_dict["values"]
            predecessors = data_dict["predecessors"]

            edge_index = [[], []]
            edge_attr = []
            for i in range(adj.shape[0]):
                for j in range(adj.shape[1]):
                    if adj[i][j] > 0:
                        edge_index[0].append(i)
                        edge_index[1].append(j)
                        edge_attr.append(adj[i][j])
            edge_index = torch.tensor(edge_index)
            edge_attr = torch.tensor(edge_attr)

            # like in the paper we concatenate so that the predecessor is first and then the values
            predecessors = predecessors.float().unsqueeze(2)
            values = values.unsqueeze(2)
            features = torch.cat((predecessors, values), dim=2)
            features = torch.transpose(features, 0, 1)
            # repeat the last iteration, so that all graphs have the same number of iterations
            last_row = features[:, -1, :].unsqueeze(1)
            last_row = last_row.repeat(1, num_nodes-1-features.shape[1], 1)
            features = torch.cat((features, last_row), dim=1)
            #print(features)
            #print(features.shape)

            target_features = features.clone().detach()[:, 1:, :]
            # repeat last row of target features
            last_row = target_features[:, -1, :].unsqueeze(1)
            target_features = torch.cat((target_features, last_row), dim=1)
            #print(target_features)
            #print(target_features.shape)

            data = Data(features, edge_index, edge_attr, y=target_features)
            if not os.path.exists(dirname):
                os.mkdir(dirname)
            torch.save(data, os.path.join(dirname, '{}.pt'.format(cnt)))
            cnt += 1

    def download(self):
        print("download")
        dirname = os.path.join(self.raw_dir, self.split)
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
            if self.split == "train":
                number_of_graphs = 100
                sizes = [20]
            elif self.split == "val":
                number_of_graphs = 5
                sizes = [20]
            elif self.split == "test":
                number_of_graphs = 5
                sizes = [20, 50, 100]
            elif self.split == "test_30":
                number_of_graphs = 20
                sizes = [30]
            elif self.split == "test_50":
                number_of_graphs = 20
                sizes = [50]
            elif self.split == "test_100":
                number_of_graphs = 20
                sizes = [100]
            threshold = 0.3

            idx = 0
            for size in sizes:
                for c in range(number_of_graphs):
                    adj = get_random_adjacency_matrix(size)
                    t, p = b_f(adj)
                    filename = os.path.join(dirname, f"{idx}.pt")
                    torch.save({"adj": adj, "values": t, "predecessors": p}, filename)
                    idx += 1

    def __getitem__(self, idx):
        if not os.path.isdir(os.path.join(self.processed_dir, self.split)):
            if not os.path.isdir(os.path.join(self.raw_dir, self.split)):
                self.download()
            self.process()
        
        data = torch.load(os.path.join(self.processed_dir, self.split, '{}.pt'.format(idx))).to(self.device)
        return data
            
if __name__ == '__main__':
    print("b_f")
    f = BellmanFordDataset("BellmanFord", split='train', device='cpu')
    print(f.root)
    print(f.processed_dir)
    print(f.raw_dir)
    # x  and y have shape (#nodes, #steps, #2), where the last dimension has 
    # the first element representing the predecessor and the second representing the value
    print(f[0].x, f[0].x.shape)
    print(f[0].y, f[0].y.shape)
    print(f[0].num_nodes)

    from torch_geometric.data import DataLoader
    dataloader = DataLoader(f, batch_size=4, shuffle=True, drop_last=False, num_workers=8)
    for i in dataloader:
        print(i)
