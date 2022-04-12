import pandas as pd
import numpy as np
import unidecode

#####################
#INDEX

# - rolling_window
# - flatten
# - gen_even_slices
# - Compound Unique
# - (Subset unique) (To be added)

# - get_click_data
# - read_parse

#####################



def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def flatten(t):
    return [item for sublist in t for item in sublist]

def gen_even_slices(n, n_packs, *, n_samples=None):
    start = 0
    if n_packs < 1:
        raise ValueError("gen_even_slices got n_packs=%s, must be >=1" % n_packs)
    for pack_num in range(n_packs):
        this_n = n // n_packs
        if pack_num < n % n_packs:
            this_n += 1
        if this_n > 0:
            end = start + this_n
            if n_samples is not None:
                end = min(n_samples, end)
            yield slice(start, end, None)
            start = end


def compound_unique(raw_sequences):
    """
    Functions like np.unqiue but for list of different sized arrays
    """
    lens = np.array([len(word) for word in raw_sequences])
    uniques,counts = [],[]
    for d in np.unique(lens):
        u,c = np.unique([raw_sequences[j] for j in (lens==d).nonzero()[0]],return_counts=True,axis=0)
        uniques.extend(u.tolist()); counts.extend(c)
    counts = np.array(counts)
    return uniques, counts


def get_click_data(path='Data/all_codas.csv',time_column='TsTo'):
    df = pd.read_csv(path)
        #remove rows with no timestep info
    df = df[~df[time_column].isna()]
    data = df.iloc[:, np.r_[56,5:45]].values
    rows,cols = np.where(data!=0)
    data[:,1:]=data[:,1:]+data[:,0][:,None]

    clicks = np.empty(shape=(rows.shape[0],6))
    i=0
    name_labels = df['Name'].unique()
    tag_labels = df['Tag'].unique()

    for row,col in zip(rows,cols):
        clicks[i,:]=[df.index[row],data[row,col],
                     np.argwhere(name_labels==df[df.index==df.index[row]].Name.values[0]).flatten()[0],
                     df[df.index==df.index[row]].Bout.values[0],
                     np.argwhere(tag_labels==df[df.index==df.index[row]].Tag.values[0]).flatten()[0],
                     df[df.index==df.index[row]].Focal.values[0]]
        i+=1

    clicks = pd.DataFrame(clicks,
                          columns=['coda_index','time','whale','bout','tag','focal']).sort_values('time').reset_index(drop=True)

    #clicks.replace({"whale": dict(zip(np.arange(len(name_labels)), name_labels))},inplace=True)
    #clicks.replace({"tag": dict(zip(np.arange(len(tag_labels)), tag_labels))},inplace=True)

    clicks['whale'] = clicks['whale'].map(dict(zip(np.arange(len(name_labels)), name_labels)))
    clicks['tag'] = clicks['tag'].map(dict(zip(np.arange(len(tag_labels)), tag_labels)))
    return clicks

def read_parse(path_to_file):
    df = pd.read_fwf(path_to_file,sep='\n')
    df= df.replace({r'\.': '',',':'',':':'','-':' ','_':''},regex=True)
    if df.shape[1]>1:
        df = df[[df.columns[0]]]
    df.columns = ['text']
    df['text'] = df['text'].apply(unidecode.unidecode).str.lower()
    df['text'] = df['text'].str.replace('\d+', '',regex=True)
    df["text"] = df['text'].str.replace('[^\w\s]','',regex=True)
    
    raw_sequences = []
    for a in df.to_numpy():
        raw_sequences.extend(a[0].split())
    return raw_sequences