import matplotlib.pyplot as plt
from matplotlib import cm
import os

def plot_feature(ax, feat):
    cax = ax.imshow(feat.T, interpolation='nearest', cmap=cm.coolwarm, origin='lower')
    ax.set_aspect(aspect=0.4)

def visualize_aug(feat, aug_feat, feat_len, step, d):
    os.makedirs(d , exist_ok=True) 

    # feat, aug_feat: [L x feat_dim]
    feat = feat[:feat_len, :]
    aug_feat = aug_feat[:feat_len, :]

    fig, (ax1, ax2) = plt.subplots(2, 1, tight_layout=True)
    plot_feature(ax1, feat)
    plot_feature(ax2, aug_feat)

    plt.tight_layout(pad=-1)
    plt.savefig(os.path.join(d, str(step)+'.pdf'))
    #plt.show()


