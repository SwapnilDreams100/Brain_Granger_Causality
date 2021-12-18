from scipy.io import loadmat
import numpy as np
import torch
import matplotlib.pyplot as plt
from models.clstm import cLSTM, train_model_accumulated_ista
import argparse
import random, os

parser = argparse.ArgumentParser()
parser.add_argument("-seed", "--seed", help = "0, 1")
parser.add_argument("-lam", "--lam", help = "0, 1")
parser.add_argument("-lr", "--lr", help = "0, 1")
parser.add_argument("-percent_var", "--percent_var", help = "0, 1")
parser.add_argument("-context", "--context", help = "0, 1")
parser.add_argument("-mbsize", "--mbsize", help = "0, 1")
parser.add_argument("-file", "--file", help = "0, 1")

args = parser.parse_args()

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(int(args.seed))

data = loadmat(args.file)
npdata = data['X']
npdata_changed_t = npdata.swapaxes(0,2)
device = torch.device('cuda')
X = torch.tensor(npdata_changed_t, dtype=torch.float32, device=torch.device('cpu'))
crnn = cLSTM(X.shape[-1], hidden=100).cuda(device=device)
train_loss_list = train_model_accumulated_ista(
    crnn, X, context=int(args.context), mbsize=int(args.mbsize), lam=float(args.lam), lam_ridge=1e-3, lr=float(args.lr), max_iter=20000,
    check_every=10, percent_var = int(args.percent_var))

GC_est = crnn.GC().cpu().data.numpy()
GC = np.array([[0, 0, 0, 0, 0],[1, 0, 0, 0, 0], [0, 0, 0, 0, 0],[1, 1, 0, 0, 0],[0, 0, 0, 0, 0]])
print('Estimated variable usage = %.2f%%' % (100 * np.mean(GC_est)))
print('Actual variable usage = %.2f%%' % (100 * np.mean(GC)))

fig, axarr = plt.subplots(1, 2, figsize=(10, 5))
axarr[0].imshow(GC, cmap='viridis')
axarr[0].set_title('GC actual')
axarr[0].set_ylabel('Affected series')
axarr[0].set_xlabel('Causal series')
axarr[0].set_xticks([])
axarr[0].set_yticks([])

axarr[1].imshow(GC_est, cmap='viridis', vmin=0, vmax=1, extent=(0, len(GC_est), len(GC_est), 0))
axarr[1].set_ylabel('Affected series')
axarr[1].set_xlabel('Causal series')
axarr[1].set_xticks([])
axarr[1].set_yticks([])

plt.show()
