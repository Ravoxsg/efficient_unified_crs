import torch
import numpy as np
from utils import seed_everything


seed_everything(42)

trainval = torch.load("DATA/trainval_data_processed_full")
p = np.random.permutation(len(trainval))
print(p[:10])
trainval = [trainval[x] for x in p]
thresh = int(len(trainval) * 0.9)
train = trainval[:thresh]
val = trainval[thresh:]

torch.save(train, "DATA/train_data_processed_full")
torch.save(val, "DATA/val_data_processed_full")
