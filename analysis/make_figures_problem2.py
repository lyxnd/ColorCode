# 问题二 (同场景输出色调不一致) 的结果图
#   problem2_alignment.png: 同场景两张输入, 把较暗的一张按不同方式对齐到较亮的一张后的 Retinexformer 输出 (取自 run_alignment_cpu.py 的推理结果)
#   problem2_exposure.png : 同一张输入只改曝光 (x0.5 / x1 / x2) 的 Retinexformer 输出, 按亮度归一化后显示, 只看色调
# 输入图在显示时用一个标量 (线性光) 提亮到 GT 的平均亮度, 不改颜色.
# 用法 (仓库根目录, 先跑 analysis/run_alignment_cpu.py): python analysis/make_figures_problem2.py
import sys, os, csv, numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
from skimage.color import rgb2lab
sys.path.insert(0,'analysis/loss_data_ft'); from common import gt, low, lin, srgb, W, L, build_rf, run
torch.set_num_threads(4)
A='analysis/alignment_intervention'; OUT='analysis/figures'; os.makedirs(OUT,exist_ok=True)
F=lambda s: ImageFont.truetype('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',s)
CW,CH,LW,HH,TH=240,160,120,44,24   # 单元宽高, 左侧标签宽, 表头高, 单元下方文字高
Y=lambda x: (lin(np.clip(x,0,1))@W).mean()
def match(x,ref): return srgb(lin(np.clip(x,0,1))*Y(ref)/max(Y(x),1e-9))   # 标量提亮到 ref 的平均亮度, 色度不变
def cell(x): return Image.fromarray(np.round(np.clip(x,0,1)*255).astype(np.uint8)).resize((CW,CH),Image.LANCZOS)
def grid(title,header,rows,path,note):
    nc=len(header); W_=LW+nc*CW; H_=60+HH+len(rows)*(CH+TH)+40
    im=Image.new('RGB',(W_,H_),'white'); d=ImageDraw.Draw(im); d.text((12,14),title,fill='black',font=F(24))
    for c,h in enumerate(header): d.multiline_text((LW+c*CW+6,60),h,fill='black',font=F(15),spacing=2)
    for r,(lab,cells,caps) in enumerate(rows):
        y=60+HH+r*(CH+TH); d.multiline_text((8,y+CH//2-20),lab,fill='black',font=F(16),spacing=4)
        for c,(x,cap) in enumerate(zip(cells,caps)):
            im.paste(cell(x),(LW+c*CW,y)); d.text((LW+c*CW+6,y+CH+2),cap,fill=(180,30,30) if cap.startswith('!') else 'black',font=F(15))
    d.text((12,H_-32),note,fill=(90,90,90),font=F(15)); im.save(path); print('saved',path)

# ---- 图 1: 对齐干预
csvr={r['pair']:r for r in csv.DictReader(open(f'{A}/alignment_pairs_retinexformer.csv',encoding='utf-8-sig'))}
pairs=[('00751_to_00752',''),('00757_to_00758',''),('00742_to_00741',''),('00768_to_00767',''),('00778_to_00775','反例')]
V=[('original','不对齐'),('exposure','只对齐曝光'),('gain','逐通道增益'),('gainoffset','增益+底座')]
rows=[]
for p,tag in pairs:
    dk,bt=p.split('_to_'); r=csvr[p]; g=gt(bt)
    cells=[g,match(low(bt),g),match(low(dk),g),L(f'{A}/outputs_retinexformer/{bt}_original.png')]
    caps=['','','','']
    for v,_ in V:
        cells.append(L(f'{A}/outputs_retinexformer/{dk}_original.png' if v=='original' else f'{A}/outputs_retinexformer/{p}_{v}.png'))
        caps.append(f'色调差 {float(r[v+":output_tone_difference"]):.2f}')
    rows.append((f'{bt} / {dk}'+(f'\n({tag})' if tag else ''),cells,caps))
grid('问题二: 把较暗的同场景输入对齐到较亮的一张后, Retinexformer 两张输出的色调差',
     ['GT','较亮输入','较暗输入','较亮输入的输出']+[f'较暗输入的输出\n{n}' for _,n in V],rows,f'{OUT}/problem2_alignment.png',
     '色调差 = 该输出与"较亮输入的输出"的全图平均 a*b* 之差. 输入在显示时按 GT 平均亮度提亮 (只乘一个系数, 不改颜色). 最后一行是增益+底座没起作用的反例.')

# ---- 图 2: 只改曝光
rf=build_rf('Retinexformer/pretrained_weights/LOL_v2_real.pth').eval()
ab=lambda x: rgb2lab(np.clip(x,0,1))[...,1:].reshape(-1,2).mean(0)
cand=sorted({r['dark_id'] for r in csvr.values()}|{r['bright_id'] for r in csvr.values()}); K=[0.5,1,2]; res=[]
for i in cand:
    lo,g=low(i),gt(i); ll=lin(lo); cin=ab(match(lo,g))-ab(g)
    if np.hypot(*cin)<5: continue
    o={k:run(rf,srgb(ll*k).astype(np.float32)) for k in K}; res.append((np.hypot(*(ab(o[2])-ab(o[0.5]))),i,o,cin))
res.sort(key=lambda t:-t[0]); rows=[]
for sc,i,o,cin in res[:4]:
    g=gt(i); gab=ab(g); cells=[g,match(low(i),g)]+[match(o[k],g) for k in K]
    tr=lambda k: (ab(o[k])-gab)@cin/(cin@cin)
    caps=['',f'输入色偏 a*{cin[0]:+.1f} b*{cin[1]:+.1f}']+[f'a*{ab(o[k])[0]-gab[0]:+.1f} b*{ab(o[k])[1]-gab[1]:+.1f}  传递 {tr(k):.2f}' for k in K]
    rows.append((i,cells,caps))
grid('同一张输入只改曝光: Retinexformer 输出的色调随输入亮度变化',['GT','输入 (x1)','输出: 输入亮度 x0.5','输出: 输入亮度 x1','输出: 输入亮度 x2'],rows,f'{OUT}/problem2_exposure.png',
     '输出已按 GT 平均亮度归一化后显示, 只保留色调差异. a*b* 为相对 GT 的偏差; 传递 = 输入色偏被带进输出的比例 (0 = 全部去掉, 1 = 照搬). 选的是色调变化最大的 4 张.')
