import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
from skimage.color import rgb2lab, deltaE_ciede2000
A='/mnt/user-data/uploads/LLIE_Color/assets/'; W='/tmp/claude-0/work/'
methods={'CWNet':'cwnet/low{}.png','HVI':'hvi/{}.png','Retinexformer':'retinexformer/{}.png','Diff-Retinex':'output/{}.png','ReDDiT':'reddit/{}.png'}
gtf={'00739':'gt_stair.npy','00751':'gt_pool2.npy','00752':'gt_pool3.npy'}
def load(m,i): return np.asarray(Image.open(A+methods[m].format(i)).convert('RGB')).astype(float)/255
for i,f in gtf.items():
    g=np.load(W+f)/255; Lg=rgb2lab(g); Cg=np.hypot(Lg[...,1],Lg[...,2])
    fig,ax=plt.subplots(3,6,figsize=(21,8.2))
    ax[0,0].imshow(g); ax[0,0].set_title('GT')
    ax[1,0].imshow(Cg,cmap='magma',vmin=0,vmax=50); ax[1,0].set_title('GT chroma C*')
    ax[2,0].axis('off')
    for k,m in enumerate(methods,1):
        o=load(m,i); Lo=rgb2lab(o); Co=np.hypot(Lo[...,1],Lo[...,2])
        dE=deltaE_ciede2000(Lg,Lo)
        X=np.c_[o.reshape(-1,3),np.ones(o.size//3)]; M=np.linalg.lstsq(X,g.reshape(-1,3),rcond=None)[0]
        oc=np.clip(X@M,0,1).reshape(g.shape); dEc=deltaE_ciede2000(Lg,rgb2lab(oc))
        ax[0,k].imshow(o); ax[0,k].set_title(f'{m}  dE00={dE.mean():.1f}')
        im1=ax[1,k].imshow(Co-Cg,cmap='RdBu',vmin=-30,vmax=30); ax[1,k].set_title('C_out-C_gt (red=greyer)')
        im2=ax[2,k].imshow(dEc,cmap='inferno',vmin=0,vmax=20); ax[2,k].set_title(f'after 3x3 fit: dE00={dEc.mean():.1f}')
    for a in ax.flat: a.set_xticks([]); a.set_yticks([])
    fig.colorbar(im1,ax=ax[1,:].tolist(),fraction=0.01); fig.colorbar(im2,ax=ax[2,:].tolist(),fraction=0.01)
    fig.suptitle(f'{i}: row2 = local chroma loss, row3 = error that a global colour matrix cannot explain',fontsize=13)
    fig.savefig(W+f'color_analysis_{i}.png',dpi=80,bbox_inches='tight'); plt.close(fig)
print('done')
