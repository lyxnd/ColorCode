# 同一场景多次采集的输出一致性评估 (LOL-v2-Real 测试集, 不需要新数据)
# 依据: 测试集 100 张图按 GT 文件内容(md5)分组, 同一组 = 同一场景、同一 GT、不同的低光采集
# 对每个模型、组内每一对输出 (i, j) 计算:
#   输出互差ΔE00   两张输出逐像素 ΔE00 均值 (理想为 0)
#   全局色调差     两张输出全图平均 a*, b* 之差的模长;  |ΔL*| 为平均亮度差
#   局部互差       用 3x3+偏置 把输出 j 全局对齐到输出 i 后的 ΔE00 -> 全局部分占比 = 1 - 局部/原始
#   亮度比         两张输出线性光平均亮度之比 (>=1)
# 另算每张图对 GT 的 PSNR / ΔE00, 用于比较 "准确度" 和 "一致性"
# 用法 (在 LLIE_Color 目录下, 每次一个步骤, 中间结果存 analysis/consistency_cache/):
#   python analysis/consistency_eval.py model <模型名>      逐个模型计算, 模型名为 results_color_exp 下的子目录
#   python analysis/consistency_eval.py seed A A_seed200    同配置不同训练种子的参照
#   python analysis/consistency_eval.py report > analysis/consistency_results.txt
import os, sys, json, hashlib, itertools, numpy as np
from PIL import Image
from skimage.color import rgb2lab, deltaE_ciede2000
R = 'Retinexformer/results_color_exp'; G = 'assets/gt'; CACHE = 'analysis/consistency_cache'
models = ['orig', 'A', 'A_seed200', 'B', 'B_seed200', 'C', 'D']
names = sorted(os.listdir(G))
os.makedirs(CACHE, exist_ok=True)
rgb = lambda p: np.asarray(Image.open(p).convert('RGB')).astype(np.float64) / 255
lin = lambda x: np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
luminance = lambda x: float((lin(x) @ np.array([0.2126, 0.7152, 0.0722])).mean())

def make_groups():
    by_hash = {}
    for n in names:
        by_hash.setdefault(hashlib.md5(open(f'{G}/{n}', 'rb').read()).hexdigest(), []).append(n)
    return sorted(by_hash.values())

def align(src, dst):
    X = np.c_[src.reshape(-1, 3), np.ones(src.size // 3)]
    M = np.linalg.lstsq(X, dst.reshape(-1, 3), rcond=None)[0]
    return np.clip(X @ M, 0, 1).reshape(dst.shape)

def run_model(m):
    groups = make_groups(); pairs = [p for g in groups for p in itertools.combinations(g, 2)]
    img, lab_cache = {}, {}
    def load(n):
        if n not in lab_cache:
            lab_cache.clear()  # 同组内的图连续出现, 只缓存一组
        return n
    out_cache = {}
    for g in groups:
        out = {n: rgb(f'{R}/{m}/test/{n}') for n in g}
        lab = {n: rgb2lab(out[n]) for n in g}
        gt = rgb(f'{G}/{g[0]}'); gl = rgb2lab(gt)
        for n in g:
            img[n] = dict(psnr=float(10 * np.log10(1 / np.mean((out[n] - gt) ** 2))),
                          de_gt=float(deltaE_ciede2000(gl, lab[n]).mean()),
                          mean_lab=lab[n].reshape(-1, 3).mean(0).tolist(), lum=luminance(out[n]))
        for a, b in itertools.combinations(g, 2):
            de = float(deltaE_ciede2000(lab[a], lab[b]).mean())
            de_local = float(deltaE_ciede2000(lab[a], rgb2lab(align(out[b], out[a]))).mean())
            img.setdefault('_pairs', []).append(dict(a=a, b=b, de=de, de_local=de_local))
    json.dump(img, open(f'{CACHE}/{m}.json', 'w'))
    print(m, 'done', len(img['_pairs']), 'pairs')

def run_seed(x, y):
    v, t = [], []
    for n in names:
        lx, ly = rgb2lab(rgb(f'{R}/{x}/test/{n}')), rgb2lab(rgb(f'{R}/{y}/test/{n}'))
        v.append(float(deltaE_ciede2000(lx, ly).mean()))
        d = lx.reshape(-1, 3).mean(0) - ly.reshape(-1, 3).mean(0); t.append(float(np.hypot(d[1], d[2])))
    json.dump(dict(de=v, tone=t), open(f'{CACHE}/seed_{x}_{y}.json', 'w')); print('seed', x, y, 'done')

def report():
    groups = make_groups(); sizes = [len(g) for g in groups]
    res = {}
    for m in models:
        if not os.path.exists(f'{CACHE}/{m}.json'): continue
        img = json.load(open(f'{CACHE}/{m}.json')); pr = img.pop('_pairs')
        for p in pr:
            d = np.array(img[p['a']]['mean_lab']) - np.array(img[p['b']]['mean_lab'])
            la, lb = img[p['a']]['lum'], img[p['b']]['lum']
            p.update(dL=abs(d[0]), tone=float(np.hypot(d[1], d[2])), da=d[1], db=d[2],
                     lum_ratio=max(la, lb) / min(la, lb), de_gt=(img[p['a']]['de_gt'] + img[p['b']]['de_gt']) / 2)
        res[m] = (img, pr)
    npairs = len(next(iter(res.values()))[1])
    print(f'== 分组: {len(groups)} 个场景, 每组 {min(sizes)}-{max(sizes)} 张, 组大小分布 '
          + ', '.join(f'{k}张x{sizes.count(k)}' for k in sorted(set(sizes))) + f', 共 {npairs} 对')
    print('   组成员: ' + ' | '.join(' '.join(x[:-4] for x in g) for g in groups if len(g) > 2) + ' (其余为两张一组)')
    print(f'\n== 一致性 vs 准确度 (组内两两比较 {npairs} 对平均; 准确度为 100 张平均)')
    print(f'{"模型":10s} {"PSNR":>6s} {"对GT ΔE":>8s} | {"输出互差ΔE":>9s} {"全局色调差":>9s} {"|ΔL*|":>6s} {"亮度比":>6s} {"局部互差":>8s} {"全局占比":>7s} {"互差>5的对":>9s}')
    for m, (img, pr) in res.items():
        de = np.array([p['de'] for p in pr]); loc = np.array([p['de_local'] for p in pr])
        print(f'{m:10s} {np.mean([v["psnr"] for v in img.values()]):6.2f} {np.mean([v["de_gt"] for v in img.values()]):8.2f} | '
              f'{de.mean():9.2f} {np.mean([p["tone"] for p in pr]):9.2f} {np.mean([p["dL"] for p in pr]):6.2f} '
              f'{np.mean([p["lum_ratio"] for p in pr]):6.3f} {loc.mean():8.2f} {1 - loc.mean() / de.mean():7.0%} {np.mean(de > 5):9.0%}')
    pr = res['orig'][1]
    print('\n== orig: 互差里颜色和亮度哪个多 (按全图平均值比较)')
    print(f'  全局色调差 > |ΔL*| 的对: {np.mean([p["tone"] > p["dL"] for p in pr]):.0%};  '
          f'全局色调差>3 的对: {np.mean([p["tone"] > 3 for p in pr]):.0%};  |ΔL*|>3 的对: {np.mean([p["dL"] > 3 for p in pr]):.0%};  '
          f'色调差中位数 {np.median([p["tone"] for p in pr]):.2f}, |ΔL*|中位数 {np.median([p["dL"] for p in pr]):.2f}')
    print('\n== 训练随机性参照: 同一配置、不同训练种子, 同一张输入的输出之差 (100 张平均)')
    for x, y in [('A', 'A_seed200'), ('B', 'B_seed200')]:
        f = f'{CACHE}/seed_{x}_{y}.json'
        if os.path.exists(f):
            s = json.load(open(f)); print(f'  {x} vs {y}: ΔE00={np.mean(s["de"]):.2f}  全局色调差={np.mean(s["tone"]):.2f}')
    print('\n== orig 互差最大的 10 对 (Δ 为前者减后者; Δa*>0 偏品红, Δb*>0 偏黄)')
    for p in sorted(pr, key=lambda p: -p['de'])[:10]:
        print(f'  {p["a"][:-4]}/{p["b"][:-4]}  互差ΔE={p["de"]:5.2f} 局部={p["de_local"]:4.2f} 色调差={p["tone"]:4.2f} '
              f'(Δa*{p["da"]:+.1f}, Δb*{p["db"]:+.1f}) |ΔL*|={p["dL"]:4.1f} 亮度比={p["lum_ratio"]:.2f}  对GT ΔE均值={p["de_gt"]:5.2f}')
    print('\n== 一致性和准确度是否同一件事: 各对 "输出互差" 与 "两张对GT ΔE均值" 的相关系数')
    for m, (img, pr) in res.items():
        print(f'  {m:10s} r={np.corrcoef([p["de"] for p in pr], [p["de_gt"] for p in pr])[0, 1]:+.2f}')
    print('\n== 模型排名对比 (1=最好): 按PSNR / 按输出互差')
    ms = list(res)
    by_psnr = sorted(ms, key=lambda m: -np.mean([v['psnr'] for v in res[m][0].values()]))
    by_cons = sorted(ms, key=lambda m: np.mean([p['de'] for p in res[m][1]]))
    print('  按PSNR:  ' + ' > '.join(by_psnr)); print('  按一致性: ' + ' > '.join(by_cons))
    dup = [f'{p["a"][:-4]}/{p["b"][:-4]}' for p in res['orig'][1] if p['de'] < 0.05]
    print('\n== 疑似重复输入 (确定性模型下输出互差<0.05): ' + (' '.join(dup) or '无'))

if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'model': run_model(sys.argv[2])
    elif cmd == 'seed': run_seed(sys.argv[2], sys.argv[3])
    else: report()
