# 偏置校正是否去掉了"暗部颜色虚高": 校正前后训练集 (和测试集) 的 GT 色度 / 输入暗示色度, 按输入亮度分档
# 用法 (仓库根目录): python analysis/loss_data_ft/check_debias.py
import sys, os, numpy as np; sys.path.insert(0,os.path.dirname(__file__)); from common import *
names=sorted(os.listdir(f'{TR}/Low')); O=[]; raw=[]; cor=[]
for n in names:
    lo,g=L(f'{TR}/Low/{n}'),L(f'{TR}/Normal/{n}'); o=dark_offset(lo,g); O.append(o)
    raw.append(block_stats(lo,g)); cor.append(block_stats(debias(lo,o),g))
O=np.array(O)*12.92*255; np.save('analysis/loss_data_ft/train_offsets.npy',np.array(O)/12.92/255)
print(f'训练集 {len(names)} 张, 估出底座的 {int((np.abs(O).sum(1)>0).sum())} 张; 底座 (DN, 线性段折算) 四分位: '+' '.join(f'{c} {np.percentile(O[:,k],[25,75]).round(1)}' for k,c in enumerate('RGB')))
cat=lambda lst: {k:np.concatenate([d[k] for d in lst]) for k in lst[0]}
R,C=cat(raw),cat(cor)
for lo_,hi_ in [(0,3),(3,5),(5,8),(8,12),(12,20),(20,40)]:
    sr,sc=dark_sel(R,lo_,hi_),dark_sel(C,lo_,hi_)
    print(f'  训练 {lo_:>2}-{hi_:<2}DN  原始 {ratio(R,"gt",sr):.2f} ({sr.sum():6d} 块)   校正后 {ratio(C,"gt",sc):.2f} ({sc.sum():6d} 块)')
TO={}; traw=[]; tcor=[]
for i in IDS:
    lo,g=low(i),gt(i); TO[i]=dark_offset(lo,g)
    if i in ('00739','00740'): continue
    traw.append(block_stats(lo,g)); tcor.append(block_stats(debias(lo,TO[i]),g))
np.save('analysis/loss_data_ft/test_offsets.npy',TO)
R,C=cat(traw),cat(tcor)
for lo_,hi_ in [(3,5),(5,8)]:
    sr,sc=dark_sel(R,lo_,hi_),dark_sel(C,lo_,hi_)
    print(f'  测试(不含00739/40) {lo_}-{hi_}DN  原始 {ratio(R,"gt",sr):.2f} ({sr.sum()} 块)   校正后 {ratio(C,"gt",sc):.2f} ({sc.sum()} 块)')
print(f'  00739 底座 (DN): {(TO["00739"]*12.92*255).round(2)}')
