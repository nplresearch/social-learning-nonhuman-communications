from scipy.stats import chi2, entropy
import numpy as np
import pandas as pd
from utils import rolling_window

#####################
#INDEX

# - VLMC
# - MCM
# - preprocess_clicks

#####################


class VLMC:
    def __init__(self, n_params):
        self.n_params= n_params
        
        
    def fit(self, sequences,counts, K=None,tol=1e-10):
        self.sequences = sequences
        self.counts = counts
        self.lens = np.array([len(s) for s in sequences])
        self.max_depth=np.max(self.lens)
        self.K = K
        self.tol = tol
        
        if self.K == None:
            self.K = 0.5*chi2.ppf(0.95,self.n_params-1)
    
        index_list = np.arange(1,len(self.sequences)).tolist()  #avoid the root node
        mark = (self.lens<self.max_depth).nonzero()[0][-1]


        self.matrix=[]
        for child in index_list[mark::-1]: #start with biggest and go up
            father = self.get_context_parent(child)

            dist_child = self.get_dist(child)
            dist_father = self.get_dist(father)

            C = self.counts[child]*entropy(dist_child,dist_father+self.tol) #to avoid inf values of entropy

            if C>=self.K:
                self.matrix.append(child)
                
        if len(self.matrix)==0:
            self.A = np.array([[0]])
            self.nodes = [0]
            self.weights = [0]
        else:
            self.create_matrix()
        
        return self

    def create_matrix(self):
        #create adjacency matrix using paths
        self.path_lens = []
        paths = np.concatenate([self.get_path(leaf) for leaf in self.matrix])
        self.nodes,b = np.unique(paths,return_inverse = True)
        #nodes[b].reshape((len(paths), 2))
        self.A = np.zeros(shape=(b.max()+1,b.max()+1))
        for edge in b.reshape(len(paths),2):
            self.A[edge[0],edge[1]] = 1
            
        self.edges = self.nodes[b].reshape(len(paths),2)        
        self.pme,self.weights =0,[]
        for edge in self.edges:
            w = self.counts[edge[1]]/self.counts[edge[0]]
            self.pme+=self.counts[edge[1]]*np.log(w)
            self.weights.append(w)
            

    def get_dist(self,i):
        """
        input w
        outputs p(a|w) for all a \in A

        """
        idx = [j for j in (self.lens==len(self.sequences[i])+1).nonzero()[0] if self.sequences[j][:-1]==self.sequences[i]]
        mask = [self.sequences[j][-1] for j in idx]
        dist = np.zeros(self.n_params)
        dist[mask] = self.counts[idx]/self.counts[i]
        return dist/np.sum(dist)


    def get_context_children(self,i):
        children = [j for j in (self.lens==len(self.sequences[i])+1).nonzero()[0] if self.sequences[j][-len(self.sequences[i]):]==self.sequences[i]]
        return children

    def get_context_parent(self,i):
        if self.lens[i]==1:
            return 0
        parent = [j for j in (self.lens==len(self.sequences[i])-1).nonzero()[0] if self.sequences[j]==self.sequences[i][-len(self.sequences[j]):]]
        return parent[0]

    def get_path(self, i):
        path = [i]
        child = i
        for _ in range(len(self.sequences[child])-1):
            child = self.get_context_parent(child)
            path.append(child)

        path = [0]+path[::-1]
        self.path_lens.append(len(path)-1)
        return rolling_window(np.array(path),2)
    
    
    



def preprocess_clicks(clicks, whale='ATWOOD',horizon = 1.5,resolution = 0.015,
                      max_depth = 10,shuffle = False,last_bin=True):
    """
    Preprocess the dataset with clicks to be added to the VLMC.
    
    """
    
    #Separate clicks by tags
    tags = []
    for name, group in clicks[clicks['whale']==whale].groupby('tag'):    
        delta = (group['time']-group['time'].min()).values
        delta = delta[1:]-delta[:-1]
        tags.append(delta)

    codas = []
    for values in tags:
        if shuffle:
            np.random.shuffle(values)
        ids = np.argwhere(values>horizon).flatten()
        inner_codas=[]
        i=0
        for j in ids:
            inner_codas.append(values[i:j])
            i=j+1
        #remove empty blocks
        codas.extend([coda for coda in inner_codas if len(coda)>0])

    bins = np.arange(0,horizon+resolution,resolution)
    
    #add symbol for last bin
    if last_bin:
        raw_sequences = [np.r_[np.digitize(coda,bins),len(bins)-1] for coda in codas]
    else:
        raw_sequences = [np.digitize(coda,bins) for coda in codas]
    
    #create context sequences
    sequences, counts = [],[]
    for d in range(1,max_depth):
        u, c = np.unique(np.concatenate([rolling_window(sequence, d) for sequence in raw_sequences if len(sequence)>=d]),return_counts=True,axis=0)
        sequences.extend(u.tolist()); counts.extend(c)
    counts = np.array(counts)

    #(add root node)
    counts = np.r_[len(sequences),counts]
    sequences = [[]]+sequences
    
    return raw_sequences, sequences, counts, len(bins)



class MCM:
    def __init__(self, depth, whale='ATWOOD', horizon=1.5, resolution=0.02, shuffle =False):
        self.depth = depth
        self.whale = whale
        self.horizon = horizon
        self.resolution = resolution
        self.shuffle=shuffle
    
    def fit(self, clicks,entropy_toggle=True):
        
        #1. Preprocess clicks into intervals
        tags = []
        for name, group in clicks[clicks['whale']==self.whale].groupby('tag'):    
            delta = (group['time']-group['time'].min()).values
            delta = delta[1:]-delta[:-1]
            tags.append(delta)

        codas = []
        for values in tags:
            if self.shuffle:
                np.random.shuffle(values)
            ids = np.argwhere(values>self.horizon).flatten()
            inner_codas=[]
            i=0
            for j in ids:
                inner_codas.append(values[i:j])
                i=j+1
            #remove empty blocks
            codas.extend([coda for coda in inner_codas if len(coda)>0])
        
        #2. DISCRETIZE
        bins = np.arange(0,self.horizon+self.resolution,self.resolution)
        self.n_params = len(bins) 
        self.positions = [np.digitize(coda,bins) for coda in codas]
        
        # divide blocks into sets of h+1 clicks: 
        #Example h=2: [1,5,3,2] -> [1,5,3],[5,3,2]
        temp,skipped = [],0
        for position in self.positions: 
            if len(position)>=self.depth: #jumps over blocks with less clicks than memory
                temp.extend([list(a) for a in rolling_window(position,self.depth)])
                skipped+=1
        self.skipped = 1- (skipped/len(self.positions))
        temp = np.array(temp)
        
        #3. PROBABILITY ESTIMATION p(y|x)
        #Count ocurrences  (x,y)
        ux, idsx,csx =np.unique(temp,return_counts=True,return_inverse=True,axis=0)
        #Count ocurrences of (x)
        u, ids,cs = np.unique(ux[:,:-1],return_counts=True,return_inverse=True,axis=0)

        #arrange into a 2d array (N) where Column1: occurences of (x) and Column2: occurences of (x,y)
        ncs = np.array([np.sum((temp[:,:-1]==unu).all(axis=1)) for unu in u])
        self.nodes = ux
        self.N = np.c_[ncs[ids],csx]
        
        #adjacency_matrix A        
        self.F = len(temp) #total number of occurences (to get frequency to get entropy)
        self.A = np.zeros(shape=(len(u),self.n_params))
        if entropy_toggle:
            for i in range(len(ux)): #aij = H(i|j)
                self.A[ids[i],ux[i,-1]] = -(self.N[i,1]/self.F)*np.log2(self.N[i,1]/self.N[i,0])
        else:
            for i in range(len(ux)): #aij = p(i|j)
                self.A[ids[i],ux[i,-1]] = self.N[i,1]/self.N[i,0]
          
        return self
    
