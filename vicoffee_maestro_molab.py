# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "torch",
#     "numpy",
#     "pandas",
#     "h5py",
#     "huggingface_hub",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # MAESTRO on ViCoffee · v1.1

    Masked-autoencoder pretraining (MAESTRO, Labatie et al. 2025, arXiv 2508.10894) on ViCoffee-S2 Sentinel-2 time series,
    ported from *MAESTRO on PASTIS v3.4*: ViT-tiny, 16×16 crops, patch 2, 16 temporal bins, LP / FT / SL with LP-FT, layer decay and EMA.
    Coffee vs not-coffee on 10 labelled sites, 2023–2025. Label efficiency under a fixed protocol: 5 folds from `splits.json`
    (train / val / test sites), 4 label ratios (5 %, 10 %, 50 %, 100 % of blocks), 3 seeds; mean ± std over 5 folds × 3 seeds.

    The encoder is pretrained once on the 120 unlabelled windows, which lie outside all 10 labelled sites, so one encoder serves every fold without test leakage.

    Run ① → ② → ③. molab keeps only files uploaded through the file browser, so download the session bundle (last section) before the session ends.

    **Changes in v1.1 (6 Oct 2026)**

    - `HF_DATASET_REPO` is preset to `Gr8-FPT-Capstone/ViCoffeeS2v1.1`. Only the token is left to enter.
    - ① reads the data straight from the zip in the dataset: it downloads the zip once and pulls the three files out of it. The extra paste-in cell is no longer needed.
    - ① also finds the three files if you drag them into molab's file browser, and then skips the download.
    - If the download fails, ① says why: the token was rejected, or it can't see the dataset (with the dataset names it can see).
    - A rejected token now says to create a new one and copy it from the pop-up, since Hugging Face shows it in full only once.
    - No backup to a second Hugging Face repo: results stay in the session bundle you download by hand.
    """)
    return


@app.cell
def _():
    VERSION = "1.1"
    import time
    SESSION_T0 = time.time()

    import copy, io, json, math, os, sys, zipfile
    from pathlib import Path

    MISSING = [pip for mod, pip in [("torch", "torch"), ("matplotlib", "matplotlib"), ("h5py", "h5py"),
                                    ("pandas", "pandas"), ("huggingface_hub", "huggingface-hub")]
               if __import__("importlib.util", fromlist=["util"]).find_spec(mod) is None]
    if MISSING:
        os.system(f"{sys.executable} -m pip install -q " + " ".join(MISSING))

    import h5py
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    WORK = Path.home() / "vicoffee_ssl"
    WORK.mkdir(parents=True, exist_ok=True)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if DEVICE == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    AMP_ON = DEVICE == "cuda"
    AMP_DTYPE = torch.bfloat16 if DEVICE == "cpu" or torch.cuda.is_bf16_supported() else torch.float16
    USE_SCALER = AMP_ON and AMP_DTYPE == torch.float16
    FUSED = DEVICE == "cuda"

    print("version:", VERSION)
    print("torch  :", torch.__version__)
    print("device :", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "(no GPU: the full protocol needs one)")
    print("work   :", WORK)
    return (
        AMP_DTYPE,
        AMP_ON,
        DEVICE,
        F,
        FUSED,
        Path,
        SESSION_T0,
        USE_SCALER,
        VERSION,
        WORK,
        copy,
        h5py,
        io,
        json,
        math,
        nn,
        np,
        os,
        pd,
        plt,
        time,
        torch,
        zipfile,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Settings
    The data source and the fixed protocol. Change the repo ids here; the rest matches the ViCoffee protocol and v3.4.
    """)
    return


@app.cell
def _(WORK):
    # --- data: private dataset in your Hugging Face organization ----------------------------
    HF_DATASET_REPO = "Gr8-FPT-Capstone/ViCoffeeS2v1.1"   # the part after huggingface.co/datasets/
    HF_REVISION = None                          # branch, tag or commit; None = main
    DATA_FILE_NAMES = ("s2_10m_2023_2025.h5", "labels_10m.h5", "splits.json")

    OUT = WORK / "runs_vicoffee_maestro"

    # --- protocol (same as vicoffee_maestro.py) ---------------------------------------------
    PRETRAINED_ENCODER = None          # path to a PASTIS mae_fold5_...pt to skip pretraining
    PRETRAIN_WITH_TRAIN_SITES = False  # True = also pretrain on the fold's train sites (one encoder per fold, as v3.4)
    MASK_CLOUDY_TOKENS = True
    SAVE_WEIGHTS = True                # best seed per (regime, fold, ratio)

    EPOCHS = dict(LP=100, FT=100, SL=100)
    MODEL = "tiny"
    CROP, PATCH, BINS = 16, 2, 16
    PRE = dict(epochs=150, bs=72, lr=3e-5, mask=0.75, p_space=0.25, p_time=0.25, p_block=0.5, block_frac=(0.25, 0.5), final_div=1e4)
    DN = dict(bs=48, lr=dict(LP=1e-5, FT=2e-5, SL=4e-5), final_div=dict(LP=1e4, FT=2.0, SL=2.0), layer_decay=0.75, lpft=True, ema=True,
              val_every=5, patience_frac=0.3, min_frac=0.5, min_delta=0.002)
    WD, WARM_FRAC = 0.01, 0.2
    return (
        BINS,
        CROP,
        DATA_FILE_NAMES,
        DN,
        EPOCHS,
        HF_DATASET_REPO,
        HF_REVISION,
        MASK_CLOUDY_TOKENS,
        MODEL,
        OUT,
        PATCH,
        PRE,
        PRETRAINED_ENCODER,
        PRETRAIN_WITH_TRAIN_SITES,
        SAVE_WEIGHTS,
        WARM_FRAC,
        WD,
    )


@app.cell
def _(mo):
    ui_folds = mo.ui.multiselect(["0", "1", "2", "3", "4"], value=["0", "1", "2", "3", "4"], label="test folds")
    ui_ratios = mo.ui.multiselect(["0.05", "0.1", "0.5", "1.0"], value=["0.05", "0.1", "0.5", "1.0"], label="label ratios")
    ui_seeds = mo.ui.dropdown(["1", "3"], value="3", label="seeds (1 = pilot)")
    ui_regimes = mo.ui.multiselect(["LP", "FT", "SL"], value=["LP", "FT", "SL"], label="regimes")
    mo.vstack([mo.md("**Run scope.** The full protocol is 5 folds × 4 ratios × 3 seeds × 3 regimes = 180 runs. "
                     "Split it across sessions by test fold if needed; finished runs are skipped."),
               mo.hstack([ui_folds, ui_ratios, ui_seeds, ui_regimes], justify="start")])
    return ui_folds, ui_ratios, ui_regimes, ui_seeds


@app.cell
def _(mo, ui_folds, ui_ratios, ui_regimes, ui_seeds):
    FOLDS = sorted(int(f) for f in ui_folds.value)
    RATIOS = sorted(ui_ratios.value, key=float)
    SEEDS = list(range(int(ui_seeds.value)))
    REGIMES = [m for m in ("LP", "FT", "SL") if m in ui_regimes.value]
    N_RUNS = len(FOLDS) * len(RATIOS) * len(SEEDS) * len(REGIMES)
    mo.md(f"**{N_RUNS} runs** · folds {FOLDS} · ratios {RATIOS} · seeds {SEEDS} · regimes {REGIMES}")
    return FOLDS, N_RUNS, RATIOS, REGIMES, SEEDS


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Library code
    Data, model, training. Collapse and move on.
    """)
    return


@app.cell
def _(
    AMP_DTYPE,
    AMP_ON,
    BINS,
    CROP,
    DEVICE,
    DN,
    EPOCHS,
    F,
    FUSED,
    MASK_CLOUDY_TOKENS,
    MODEL,
    OUT,
    PATCH,
    PRE,
    PRETRAIN_WITH_TRAIN_SITES,
    Path,
    SAVE_WEIGHTS,
    USE_SCALER,
    WARM_FRAC,
    WD,
    copy,
    h5py,
    json,
    math,
    nn,
    np,
    pd,
    time,
    torch,
):
    YEARS = ["2023", "2024", "2025"]
    CLEAR_SCL = (4, 5, 6, 7)
    N_BANDS, N_CLASSES, IGNORE = 10, 2, 255
    S2_GROUPS = ((0, 4), (4, 8), (8, 10))
    PRESETS = {"tiny": dict(dim=192, depth=12, heads=3, dec_dim=128, dec_depth=4, dec_heads=4),
               "small": dict(dim=384, depth=12, heads=6, dec_dim=256, dec_depth=4, dec_heads=8),
               "base": dict(dim=768, depth=12, heads=12, dec_dim=512, dec_depth=8, dec_heads=16)}
    STATE = {}  # filled by ①: STATE['ST'] = normalisation stats, STATE['EV'] = eval crops


    def make_bins(T):
        parts = np.array_split(np.arange(T), BINS)
        idx = np.full((BINS, max(len(p) for p in parts)), -1)
        for i, p in enumerate(parts):
            idx[i, :len(p)] = p
        return idx


    def load_data(files):
        f = h5py.File(files["s2_10m_2023_2025.h5"], "r")
        L = h5py.File(files["labels_10m.h5"], "r")
        dates = np.array([d.decode() if isinstance(d, bytes) else str(d) for d in f["dates"][:]])
        years = np.array([d[:4] for d in dates])
        doy = np.array([(np.datetime64(d[:10], "D") - np.datetime64(d[:4] + "-01-01", "D")).astype(int) + 1 for d in dates])
        tidx = {y: np.where(years == y)[0] for y in YEARS}
        sites = sorted(L.keys())
        data = {p: dict(X=f[f"labeled/{p}/X"][:], clear=np.isin(f[f"labeled/{p}/SCL"][:], CLEAR_SCL),
                        label=L[p]["label"][:], block=L[p]["block"][:]) for p in sites}
        splits = json.loads(Path(files["splits.json"]).read_text())
        return dict(f=f, sites=sites, doy=doy, tidx=tidx, bins={y: make_bins(len(t)) for y, t in tidx.items()}, data=data, splits=splits)


    def load_unlabeled(D):
        if "U" not in D:
            t0 = time.time()
            U = D["f"]["unlabeled"]
            D["U"] = dict(X=U["X"][:], clear=np.isin(U["SCL"][:], CLEAR_SCL))
            print(f"unlabeled windows loaded: {D['U']['X'].shape} in {time.time() - t0:.0f}s")
        return D["U"]


    def norm_stats(D, n_windows=40, seed=0):
        path = OUT / "norm_stats.json"
        if not path.exists():
            U = D["f"]["unlabeled"]
            rng = np.random.default_rng(seed)
            buf = [[] for _ in range(N_BANDS)]
            for k in np.sort(rng.choice(U["X"].shape[0], n_windows, replace=False)):
                x, c = U["X"][k][::4], np.isin(U["SCL"][k][::4], CLEAR_SCL)
                for b in range(N_BANDS):
                    v = x[:, b][c]
                    if v.size:
                        buf[b].append(rng.choice(v, min(20000, v.size), replace=False).astype("f4"))
            st = {q: [float(np.quantile(np.concatenate(buf[b]), a)) for b in range(N_BANDS)] for q, a in (("q05", 0.05), ("median", 0.5), ("q95", 0.95))}
            path.write_text(json.dumps(st, indent=1))
        st = json.loads(path.read_text())
        s = {k: np.asarray(v, "f4")[None, :, None, None] for k, v in st.items()}
        s["range"] = np.maximum(s["q95"] - s["q05"], 1e-6)
        return s


    def normalize(x, clear):
        ST = STATE["ST"]
        x = (np.clip(x.astype("f4"), ST["q05"], ST["q95"]) - ST["median"]) / ST["range"]
        return x * clear[:, None]


    def pick_dates(cnt, bins, rng):
        idx, ok = np.maximum(bins, 0), bins >= 0
        cand = ok & (cnt[idx] > 0)
        none = ~cand.any(1)
        cand[none] = ok[none]
        return idx[np.arange(len(bins)), (rng.random(bins.shape) * cand).argmax(1)]


    def pick_dates_eval(xn, cnt, bins):
        idx, ok = np.maximum(bins, 0), bins >= 0
        xb = np.where(ok[..., None], xn.reshape(len(xn), -1)[idx], np.nan)
        mad = np.abs(xb - np.nanmedian(xb, 1, keepdims=True)).mean(-1)
        c = np.where(ok, cnt[idx], -1)
        mad[(c < 0.95 * c.max(1, keepdims=True)) | ~ok] = np.inf
        return idx[np.arange(len(bins)), mad.argmin(1)]


    def sample_crop(X, clear, t_year, bins, r, c, rng):
        t = t_year[pick_dates(clear[t_year, r:r + CROP, c:c + CROP].sum((1, 2)), bins, rng)]
        cl = clear[t, r:r + CROP, c:c + CROP]
        return normalize(X[t, :, r:r + CROP, c:c + CROP], cl), cl, t


    def to_inputs(x, cl, doy):
        x = torch.from_numpy(np.ascontiguousarray(x)).to(DEVICE).float()
        cl = torch.from_numpy(np.ascontiguousarray(cl)).to(DEVICE)
        B, T, g = cl.shape[0], cl.shape[1], CROP // PATCH
        tok = cl.reshape(B, T, g, PATCH, g, PATCH)
        d = torch.from_numpy(np.asarray(doy, "f4")).to(DEVICE)
        tf = temporal_features(d, d - 1)
        if not MASK_CLOUDY_TOKENS:
            return x, tf, None, None
        return x, tf, ~tok.any(5).any(3).reshape(B, T * g * g), tok.all(5).all(3).reshape(B, T * g * g)


    def temporal_features(doy, days_since_ref):
        d = doy / 365.25
        r = (days_since_ref / 365.25).unsqueeze(-1).expand(*doy.shape, 4)
        return torch.cat([torch.stack([torch.sin(2 * math.pi * d), torch.cos(2 * math.pi * d), torch.sin(4 * math.pi * d), torch.cos(4 * math.pi * d)], -1), r], -1)


    def sincos_2d(dim, g):
        y, x = torch.meshgrid(torch.arange(g, dtype=torch.float32), torch.arange(g, dtype=torch.float32), indexing="ij")
        om = 1.0 / 10000 ** (torch.arange(dim // 4, dtype=torch.float32) / (dim // 4))
        return torch.cat([f(c.flatten()[:, None] * om[None]) for c in (y, x) for f in (torch.sin, torch.cos)], 1)


    def patchify(x, P):
        B, T, C, H, W = x.shape
        g = H // P
        x = x.reshape(B, T, C, g, P, g, P).permute(0, 1, 3, 5, 4, 6, 2)
        return x.reshape(B, T, g * g, P * P * C)


    def build_mask(B, T, S, ratio, p_space, p_time, p_block, block_frac, device):
        rnd = lambda *s: torch.rand(*s, device=device)
        m = (rnd(B, 1, S) < p_space) | (rnd(B, T, 1) < p_time)
        if p_block > 0:
            lo, hi = block_frac
            span = ((lo + (hi - lo) * rnd(B)) * T).round().clamp(1, T).long()
            start = (rnd(B) * (T - span + 1).float()).floor().long()
            t = torch.arange(T, device=device)[None]
            m = m | ((t >= start[:, None]) & (t < (start + span)[:, None]) & (rnd(B, 1) < p_block))[:, :, None]
        m = m.expand(B, T, S).reshape(B, T * S)
        order = (m.float() + 0.5 * rnd(B, T * S)).argsort(1, descending=True)
        n_mask = int(round(ratio * T * S))
        masked = torch.zeros(B, T * S, dtype=torch.bool, device=device)
        masked.scatter_(1, order[:, :n_mask], True)
        return masked, order[:, n_mask:]


    def group_norm_targets(t, C, groups=S2_GROUPS, eps=1e-6):
        s = t.shape
        t = t.reshape(*s[:-1], -1, C)
        out = torch.empty_like(t)
        for a, b in groups:
            g = t[..., a:b]
            out[..., a:b] = (g - g.mean(dim=(-2, -1), keepdim=True)) / (g.std(dim=(-2, -1), keepdim=True) + eps)
        return out.reshape(s)


    def safe_mask(kpm):
        return None if kpm is None else kpm & ~kpm.all(1, keepdim=True)


    class Block(nn.Module):
        def __init__(self, dim, heads, mlp=4.0):
            super().__init__()
            self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
            self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
            self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp)), nn.GELU(), nn.Linear(int(dim * mlp), dim))

        def forward(self, x, kpm=None):
            h = self.n1(x)
            x = x + self.attn(h, h, h, key_padding_mask=kpm, need_weights=False)[0]
            return x + self.mlp(self.n2(x))


    class Encoder(nn.Module):
        def __init__(self, in_ch=10, patch=2, crop=16, dim=192, depth=12, heads=3, **_):
            super().__init__()
            self.P, self.C, self.g, self.dim = patch, in_ch, crop // patch, dim
            self.embed = nn.Linear(patch * patch * in_ch, dim)
            self.register_buffer("pos", sincos_2d(dim - 8, self.g), persistent=False)
            self.te = nn.Linear(8, 8, bias=False)
            nn.init.eye_(self.te.weight)
            self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(depth))
            self.norm = nn.LayerNorm(dim)

        def tokens(self, x, tfeat):
            B, T = x.shape[:2]
            S = self.g * self.g
            pe = torch.cat([self.pos[None, None].expand(B, T, S, -1), self.te(tfeat)[:, :, None].expand(B, T, S, 8)], -1)
            return (self.embed(patchify(x, self.P)) + pe).reshape(B, T * S, self.dim)

        def forward(self, x, tfeat, keep=None, kpm=None):
            z = self.tokens(x, tfeat)
            if keep is not None:
                z = torch.gather(z, 1, keep[..., None].expand(-1, -1, self.dim))
                kpm = None if kpm is None else torch.gather(kpm, 1, keep)
            kpm = safe_mask(kpm)
            for b in self.blocks:
                z = b(z, kpm)
            return self.norm(z)


    class MAE(nn.Module):
        def __init__(self, in_ch=10, patch=2, crop=16, preset="tiny"):
            super().__init__()
            cfg = PRESETS[preset]
            self.encoder = Encoder(in_ch, patch, crop, cfg["dim"], cfg["depth"], cfg["heads"])
            dd = cfg["dec_dim"]
            self.dec_embed = nn.Linear(cfg["dim"], dd)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, dd))
            nn.init.normal_(self.mask_token, std=0.02)
            self.register_buffer("dpos", sincos_2d(dd - 8, crop // patch), persistent=False)
            self.dec_te = nn.Linear(8, 8, bias=False)
            nn.init.eye_(self.dec_te.weight)
            self.dec = nn.ModuleList(Block(dd, cfg["dec_heads"]) for _ in range(cfg["dec_depth"]))
            self.dec_norm = nn.LayerNorm(dd)
            self.pred = nn.Linear(dd, patch * patch * in_ch)

        def forward(self, x, tfeat, masked, keep, kpm=None):
            B, T = x.shape[:2]
            S = self.encoder.g ** 2
            z = self.dec_embed(self.encoder(x, tfeat, keep, kpm))
            full = self.mask_token.to(z.dtype).expand(B, T * S, -1).clone()
            full.scatter_(1, keep[..., None].expand(-1, -1, z.shape[-1]), z)
            pe = torch.cat([self.dpos[None, None].expand(B, T, S, -1), self.dec_te(tfeat)[:, :, None].expand(B, T, S, 8)], -1)
            h = full + pe.reshape(B, T * S, -1)
            dkpm = safe_mask(None if kpm is None else kpm & ~masked)
            for b in self.dec:
                h = b(h, dkpm)
            return self.pred(self.dec_norm(h))

        def loss(self, x, tfeat, masked, keep, kpm=None, target_ok=None):
            pred = self(x, tfeat, masked, keep, kpm)
            tgt = group_norm_targets(patchify(x, self.encoder.P).flatten(1, 2), self.encoder.C)
            err = (pred.float() - tgt.float()).abs().mean(-1)
            w = masked if target_ok is None else masked & target_ok
            return (err * w).sum() / w.sum().clamp(min=1)


    class Segmenter(nn.Module):
        def __init__(self, encoder, n_classes, freeze=False):
            super().__init__()
            self.encoder, self.freeze, self.K = encoder, freeze, n_classes
            d = encoder.dim
            self.query = nn.Parameter(torch.zeros(1, 1, d))
            nn.init.normal_(self.query, std=0.02)
            self.pn = nn.LayerNorm(d)
            self.pool = nn.MultiheadAttention(d, max(1, d // 64), batch_first=True)
            self.out = nn.Linear(d, n_classes * encoder.P ** 2)
            if freeze:
                for p in self.encoder.parameters():
                    p.requires_grad_(False)

        def train(self, mode=True):
            super().train(mode)
            if self.freeze:
                self.encoder.eval()
            return self

        def forward(self, x, tfeat, kpm=None):
            B, T = x.shape[:2]
            g, P, d = self.encoder.g, self.encoder.P, self.encoder.dim
            with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze):
                z = self.encoder(x, tfeat, kpm=kpm)
            z = self.pn(z.reshape(B, T, g * g, d).transpose(1, 2).reshape(B * g * g, T, d))
            pm = None if kpm is None else safe_mask(kpm.reshape(B, T, g * g).transpose(1, 2).reshape(B * g * g, T))
            pooled = self.pool(self.query.expand(B * g * g, 1, d), z, z, key_padding_mask=pm, need_weights=False)[0]
            lg = self.out(pooled.squeeze(1)).reshape(B, g, g, P, P, self.K)
            return lg.permute(0, 5, 1, 3, 2, 4).reshape(B, self.K, g * P, g * P)


    class EpochEMA:
        def __init__(self, model, n_epochs):
            self.alpha = 1.0 - 1.0 / max(1.0, 0.2 * n_epochs)
            self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

        @torch.no_grad()
        def update(self, model):
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    self.shadow[k].mul_(self.alpha).add_(v.detach(), alpha=1 - self.alpha)
                else:
                    self.shadow[k].copy_(v)


    def lr_lambda(total, warm_frac, final_div, start_div=25.0):
        w = max(1, int(total * warm_frac))

        def f(step):
            if step < w:
                return 1.0 / start_div + (1 - 1.0 / start_div) * step / w
            p = min(1.0, (step - w) / max(1, total - w))
            return 1.0 / final_div + (1 - 1.0 / final_div) * 0.5 * (1 + math.cos(math.pi * p))
        return f


    def param_groups(model, lr, decay):
        depth, groups = len(model.encoder.blocks), {}
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if n.startswith("encoder.blocks."):
                lid = int(n.split(".")[2]) + 1
            elif n.startswith("encoder.norm") or not n.startswith("encoder."):
                lid = depth + 1
            else:
                lid = 0
            groups.setdefault(decay ** (depth + 1 - lid), []).append(p)
        return [{"params": ps, "lr": lr * sc} for sc, ps in sorted(groups.items())]


    def step(opt, sch, scaler, loss):
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sch.step()


    def ckpt_path(fold):
        tag = f"unl+train_f{fold}" if PRETRAIN_WITH_TRAIN_SITES else "unl"
        m = "_cm" if MASK_CLOUDY_TOKENS else ""
        return OUT / "pretrain" / (f"mae_{tag}_{MODEL}_D{BINS}_c{CROP}p{PATCH}_m{int(PRE['mask'] * 100)}_s{int(PRE['p_space'] * 100)}"
                                   f"_t{int(PRE['p_time'] * 100)}_b{int(PRE['p_block'] * 100)}_e{PRE['epochs']}_bs{PRE['bs']}{m}.pt")


    def pretrain(D, fold):
        path = ckpt_path(fold)
        if path.exists():
            return path
        U = load_unlabeled(D)
        imgs = [(U["X"][k], U["clear"][k], y) for k in range(len(U["X"])) for y in YEARS]
        if PRETRAIN_WITH_TRAIN_SITES:
            imgs += [(D["data"][p]["X"], D["data"][p]["clear"], y) for p in D["splits"]["folds"][fold]["train"] for y in YEARS]
        rows = np.concatenate([np.full(max(1, X.shape[-2] * X.shape[-1] // CROP ** 2), i) for i, (X, _, _) in enumerate(imgs)])
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        mae = MAE(N_BANDS, PATCH, CROP, MODEL).to(DEVICE)
        bs = PRE["bs"]
        opt = torch.optim.AdamW(mae.parameters(), lr=PRE["lr"] * bs ** 0.5, betas=(0.9, 0.99), weight_decay=WD, fused=FUSED)
        spe = -(-len(rows) // bs)
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(spe * PRE["epochs"], WARM_FRAC, PRE["final_div"]))
        scaler = torch.amp.GradScaler(enabled=USE_SCALER)
        S, hist, t0 = (CROP // PATCH) ** 2, [], time.time()
        print(f"pretraining: {len(imgs)} image-years, {len(rows)} crops/epoch, {spe} steps/epoch, {PRE['epochs']} epochs")
        for ep in range(PRE["epochs"]):
            mae.train()
            er, tot = rng.permutation(rows), torch.zeros((), device=DEVICE)
            for a in range(0, len(er), bs):
                xs, cls, ds = [], [], []
                for i in er[a:a + bs]:
                    X, clear, y = imgs[i]
                    r, c = rng.integers(0, X.shape[-2] - CROP + 1), rng.integers(0, X.shape[-1] - CROP + 1)
                    x, cl, t = sample_crop(X, clear, D["tidx"][y], D["bins"][y], r, c, rng)
                    xs.append(x)
                    cls.append(cl)
                    ds.append(D["doy"][t])
                x, tf, kpm, full = to_inputs(np.stack(xs), np.stack(cls), np.stack(ds))
                masked, keep = build_mask(x.shape[0], BINS, S, PRE["mask"], PRE["p_space"], PRE["p_time"], PRE["p_block"], PRE["block_frac"], DEVICE)
                with torch.autocast(DEVICE, dtype=AMP_DTYPE, enabled=AMP_ON):
                    loss = mae.loss(x, tf, masked, keep, kpm, full)
                step(opt, sch, scaler, loss)
                tot += loss.detach()
            hist.append(float(tot) / spe)
            print(f"  pretrain epoch {ep + 1:3d}/{PRE['epochs']}  masked L1 {hist[-1]:.4f}  {(time.time() - t0) / 60:.1f} min")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"encoder": mae.encoder.state_dict(), "model": MODEL, "fold": fold, "history": hist, "epoch": PRE["epochs"], "bins": BINS,
                    "crop": CROP, "patch": PATCH, "pre": PRE, "with_train_sites": PRETRAIN_WITH_TRAIN_SITES, "norm_stats": {k: v.ravel().tolist() for k, v in STATE["ST"].items()},
                    "secs": time.time() - t0}, path)
        return path


    def eval_cache(D):
        cache = {}
        for p in D["sites"]:
            d = D["data"][p]
            H, W = d["label"].shape[1:]
            pos = [(r, c) for r in sorted(set(range(0, H - CROP + 1, CROP)) | {H - CROP}) for c in sorted(set(range(0, W - CROP + 1, CROP)) | {W - CROP})]
            for y in YEARS:
                ty, xs, cls, ds = D["tidx"][y], [], [], []
                for r, c in pos:
                    cl = d["clear"][ty, r:r + CROP, c:c + CROP]
                    xn = normalize(d["X"][ty, :, r:r + CROP, c:c + CROP], cl)
                    k = pick_dates_eval(xn, cl.sum((1, 2)), D["bins"][y])
                    xs.append(xn[k].astype(np.float16))
                    cls.append(cl[k])
                    ds.append(D["doy"][ty[k]])
                cache[(p, y)] = (np.stack(xs), np.stack(cls), np.stack(ds), pos)
        return cache


    def metrics_from_cm(cm):
        tn, fp, fn, tp = [float(v) for v in cm.ravel()]
        n = tn + fp + fn + tp
        iou1, iou0 = tp / max(tp + fp + fn, 1), tn / max(tn + fn + fp, 1)
        p1, r1 = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        p0, r0 = tn / max(tn + fn, 1), tn / max(tn + fp, 1)
        f1, f0 = 2 * p1 * r1 / max(p1 + r1, 1e-9), 2 * p0 * r0 / max(p0 + r0, 1e-9)
        oa = (tp + tn) / max(n, 1)
        pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / max(n * n, 1)
        return dict(mIoU=(iou0 + iou1) / 2, IoU_coffee=iou1, IoU_not=iou0, F1_coffee=f1, macro_F1=(f0 + f1) / 2, OA=oa,
                    kappa=(oa - pe) / max(1 - pe, 1e-9), precision_coffee=p1, recall_coffee=r1)


    @torch.no_grad()
    def evaluate(model, D, sites):
        model.eval()
        cm_all, cm_year = np.zeros((2, 2), np.int64), {y: np.zeros((2, 2), np.int64) for y in YEARS}
        for p in sites:
            d = D["data"][p]
            H, W = d["label"].shape[1:]
            for yi, y in enumerate(YEARS):
                x, cl, doy, pos = STATE["EV"][(p, y)]
                x, tf, kpm, _ = to_inputs(x, cl, doy)
                with torch.autocast(DEVICE, dtype=AMP_DTYPE, enabled=AMP_ON):
                    prob = model(x, tf, kpm).float().softmax(1).cpu().numpy()
                acc, cnt = np.zeros((N_CLASSES, H, W), "f4"), np.zeros((H, W), "f4")
                for k, (r, c) in enumerate(pos):
                    acc[:, r:r + CROP, c:c + CROP] += prob[k]
                    cnt[r:r + CROP, c:c + CROP] += 1
                pred, lab = (acc / cnt).argmax(0), d["label"][yi]
                m = lab != IGNORE
                cm = np.bincount(lab[m].astype(np.int64) * 2 + pred[m], minlength=4).reshape(2, 2)
                cm_all += cm
                cm_year[y] += cm
        res = metrics_from_cm(cm_all)
        res["mIoU_by_year"] = {y: metrics_from_cm(c)["mIoU"] for y, c in cm_year.items()}
        res["confusion"] = cm_all.tolist()
        return res


    def train_units(D, fold, ratio, seed):
        chosen = {}
        for s in D["splits"]["subsets"][str(fold)][str(seed)][ratio]:
            p, b = s.split(":")
            chosen.setdefault(p, []).append(int(b))
        units, labels = [], {}
        for p, blocks in chosen.items():
            d = D["data"][p]
            labels[p] = np.where(np.isin(d["block"], blocks)[None], d["label"], IGNORE)
            for b in blocks:
                rr, cc = np.where(d["block"] == b)
                for yi in range(len(YEARS)):
                    if (labels[p][yi, rr, cc] != IGNORE).any():
                        units.append((p, yi, rr.min(), rr.max(), cc.min(), cc.max()))
        return units, labels


    def train_batch(D, units, labels, rng):
        xs, cls, ds, ys = [], [], [], []
        for p, yi, r0, r1, c0, c1 in units:
            d, y = D["data"][p], YEARS[yi]
            H, W = d["label"].shape[1:]
            r = rng.integers(max(0, r1 - CROP + 1), min(r0, H - CROP) + 1)
            c = rng.integers(max(0, c1 - CROP + 1), min(c0, W - CROP) + 1)
            x, cl, t = sample_crop(d["X"], d["clear"], D["tidx"][y], D["bins"][y], r, c, rng)
            xs.append(x)
            cls.append(cl)
            ds.append(D["doy"][t])
            ys.append(labels[p][yi, r:r + CROP, c:c + CROP])
        x, tf, kpm, _ = to_inputs(np.stack(xs), np.stack(cls), np.stack(ds))
        return x, tf, kpm, torch.from_numpy(np.stack(ys).astype(np.int64)).to(DEVICE)


    def build_segmenter(mode, enc_path):
        enc = Encoder(N_BANDS, PATCH, CROP, **PRESETS[MODEL])
        if mode in ("LP", "FT"):
            enc.load_state_dict(torch.load(enc_path, map_location="cpu", weights_only=False)["encoder"])
        return Segmenter(enc, N_CLASSES, freeze=mode == "LP").to(DEVICE)


    def train_run(D, mode, fold, ratio, seed, enc_path):
        torch.manual_seed(seed)
        rng = np.random.default_rng(10000 + seed)
        units, labels = train_units(D, fold, ratio, seed)
        val_sites = D["splits"]["folds"][fold]["val"]
        model = build_segmenter(mode, enc_path)
        head_file = OUT / "lp_heads" / f"f{fold}_r{ratio}_s{seed}.pt"
        lpft = mode == "FT" and DN["lpft"] and head_file.exists()
        if lpft:
            model.load_state_dict(torch.load(head_file, map_location="cpu"), strict=False)
        n_ep, bs = EPOCHS[mode], min(DN["bs"], len(units))
        groups = param_groups(model, DN["lr"][mode] * bs ** 0.5, DN["layer_decay"] if mode == "FT" else 1.0)
        opt = torch.optim.AdamW(groups, betas=(0.9, 0.99), weight_decay=WD, fused=FUSED)
        spe = -(-len(units) // bs)
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(spe * n_ep, WARM_FRAC, DN["final_div"][mode]))
        scaler = torch.amp.GradScaler(enabled=USE_SCALER)
        ema = EpochEMA(model, n_ep) if DN["ema"] and mode != "LP" else None
        shadow = copy.deepcopy(model) if ema else None
        patience = max(DN["val_every"], int(math.ceil(DN["patience_frac"] * n_ep)))
        min_ep = int(DN["min_frac"] * n_ep)
        best, best_ref, best_state, best_ep, last_imp, curve = -1.0, -1.0, None, 0, 0, []
        for ep in range(n_ep):
            model.train()
            order, tot = rng.permutation(len(units)), torch.zeros((), device=DEVICE)
            for a in range(0, len(order), bs):
                x, tf, kpm, y = train_batch(D, [units[i] for i in order[a:a + bs]], labels, rng)
                with torch.autocast(DEVICE, dtype=AMP_DTYPE, enabled=AMP_ON):
                    loss = F.cross_entropy(model(x, tf, kpm).float(), y, ignore_index=IGNORE)
                step(opt, sch, scaler, loss)
                tot += loss.detach()
            ran = ep + 1
            if ema:
                ema.update(model)
            if ran % DN["val_every"] == 0 or ran == n_ep:
                target = model
                if ema:
                    shadow.load_state_dict(ema.shadow)
                    target = shadow
                v = evaluate(target, D, val_sites)["mIoU"]
                curve.append((ran, round(float(tot) / spe, 5), round(v, 5)))
                mark = ""
                if v > best:
                    best, best_ep, mark = v, ran, "  *"
                    best_state = {k: t.detach().cpu().clone() for k, t in target.state_dict().items()}
                if v > best_ref + DN["min_delta"]:
                    best_ref, last_imp = v, ran
                print(f"    epoch {ran:3d}  loss {float(tot) / spe:.4f}  val mIoU {v:.4f}{mark}")
                if ran >= min_ep and ran - last_imp >= patience:
                    break
        model.load_state_dict(best_state)
        if mode == "LP":
            head_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save({k: v for k, v in best_state.items() if not k.startswith("encoder.")}, head_file)
        info = dict(n_units=len(units), n_pixels=int(sum((l != IGNORE).sum() for l in labels.values())), batch=bs, steps_per_epoch=spe,
                    epochs_planned=n_ep, epochs_run=ran, best_epoch=best_ep, val_mIoU=best, lpft_init=lpft, curve=curve)
        return model, info


    def save_weights(model, mode, fold, ratio, seed, info, test):
        path = OUT / "weights" / f"{mode}_f{fold}_r{ratio}_best.pt"
        if path.exists() and torch.load(path, map_location="cpu", weights_only=False)["val_mIoU"] >= info["val_mIoU"]:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "regime": mode, "fold": fold, "ratio": ratio, "seed": seed,
                    "encoder_size": MODEL, "n_classes": N_CLASSES, "bins": BINS, "crop": CROP, "patch": PATCH, "val_mIoU": info["val_mIoU"],
                    "test": {k: v for k, v in test.items() if k != "confusion"}}, path)


    def run_one(D, mode, fold, ratio, seed, enc_path):
        path = OUT / "runs" / f"{mode}_f{fold}_r{ratio}_s{seed}.json"
        if path.exists():
            return json.loads(path.read_text())
        t0 = time.time()
        model, info = train_run(D, mode, fold, ratio, seed, enc_path)
        test = evaluate(model, D, D["splits"]["folds"][fold]["test"])
        if SAVE_WEIGHTS:
            save_weights(model, mode, fold, ratio, seed, info, test)
        res = dict(regime=mode, fold=fold, ratio=ratio, seed=seed, encoder=str(enc_path) if mode != "SL" else None, minutes=round((time.time() - t0) / 60, 2), **info, test=test)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(res, indent=1))
        return res


    def summarize(results):
        rows = [dict(regime=r["regime"], fold=r["fold"], ratio=r["ratio"], seed=r["seed"], epochs_run=r["epochs_run"], best_epoch=r["best_epoch"],
                     val_mIoU=r["val_mIoU"], **{k: v for k, v in r["test"].items() if k not in ("mIoU_by_year", "confusion")},
                     **{f"mIoU_{y}": v for y, v in r["test"]["mIoU_by_year"].items()}) for r in results]
        df = pd.DataFrame(rows)
        df.to_csv(OUT / "all_runs.csv", index=False)
        keys = ["mIoU", "IoU_coffee", "IoU_not", "F1_coffee", "macro_F1", "OA", "kappa"] + [f"mIoU_{y}" for y in YEARS]
        g = df.groupby(["ratio", "regime"])
        table = (100 * g[keys].mean()).round(1).astype(str) + " ± " + (100 * g[keys].std(ddof=0)).round(1).astype(str)
        table.insert(0, "runs", g.size())
        table.insert(1, "epochs_run", g["epochs_run"].median().astype(int))
        table.to_csv(OUT / "summary_mean_std.csv")
        return table

    return (
        IGNORE,
        STATE,
        YEARS,
        ckpt_path,
        eval_cache,
        load_data,
        norm_stats,
        pretrain,
        run_one,
        summarize,
        train_units,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data
    Downloads the three ViCoffee files (about 1.2 GB) from the private dataset, loads them into memory (about 3 GB RAM),
    computes normalisation statistics from the unlabelled windows and caches the evaluation crops.
    Token: add a molab secret `HF_TOKEN`, or paste it below for this session only.
    """)
    return


@app.cell
def _(mo):
    ui_hf_token = mo.ui.text(kind="password", label="Hugging Face token (leave empty if the HF_TOKEN secret is set)", full_width=True)
    ui_hf_token
    return (ui_hf_token,)


@app.cell
def _(mo):
    run_cache = mo.ui.run_button(label="① Download + load data")
    run_cache
    return (run_cache,)


@app.cell
def _(
    AMP_DTYPE,
    AMP_ON,
    BINS,
    CROP,
    DATA_FILE_NAMES,
    DEVICE,
    HF_DATASET_REPO,
    HF_REVISION,
    OUT,
    PATCH,
    Path,
    STATE,
    WORK,
    YEARS,
    eval_cache,
    load_data,
    mo,
    norm_stats,
    np,
    os,
    run_cache,
    ui_hf_token,
    zipfile,
):
    DATA_DIR = WORK / "data"
    HF_TOKEN = ui_hf_token.value.strip() or os.environ.get("HF_TOKEN") or None


    def find_data_files():
        # the HF download lands in DATA_DIR; files uploaded through molab's file browser sit next to the notebook
        _roots = [DATA_DIR] + [r for r in {Path(mo.notebook_dir() or Path.cwd()), Path.cwd()} if r.exists()]
        found = {}
        for _n in DATA_FILE_NAMES:
            for _root in _roots:
                _hits = sorted((h for h in _root.rglob(_n) if WORK not in h.parents or _root == DATA_DIR),
                               key=lambda p: len(p.parts)) if _root.exists() else []
                if _hits:
                    found[_n] = _hits[0]
                    break
        return found


    DATA_READY = len(find_data_files()) == len(DATA_FILE_NAMES)
    mo.stop(not run_cache.value and not DATA_READY, mo.md("*Press ① to download and load the data.*"))
    if not DATA_READY:
        mo.stop("/" not in HF_DATASET_REPO or HF_DATASET_REPO.startswith("YOUR-ORG"),
                mo.md("**Set `HF_DATASET_REPO` in Settings first.**"))
        mo.stop(HF_TOKEN is None, mo.md("**No token: add the `HF_TOKEN` secret or paste a token above.**"))
        from huggingface_hub import snapshot_download
        from huggingface_hub import HfApi as _HfApi
        from huggingface_hub.errors import HfHubHTTPError as _HfHubHTTPError
        try:
            with mo.status.spinner(title=f"downloading {HF_DATASET_REPO}"):
                snapshot_download(HF_DATASET_REPO, repo_type="dataset", revision=HF_REVISION, token=HF_TOKEN,
                                  local_dir=DATA_DIR, allow_patterns=[f"*{_n}" for _n in DATA_FILE_NAMES] + ["*.zip"])
        except _HfHubHTTPError as _err:
            _api = _HfApi(token=HF_TOKEN)
            try:
                _who = _api.whoami()["name"]
            except Exception:
                _who = None
            if _who is None:
                _why = ("**The token was rejected** (it is wrong, expired or cut short). Hugging Face shows a token in full only once, so create a new one "
                        "at huggingface.co/settings/tokens (or use *Invalidate and refresh* on the old one), copy it from the pop-up and paste it above. "
                        "If you saved it as a secret, replace the secret too and restart the notebook.")
            else:
                _org = HF_DATASET_REPO.split("/")[0]
                try:
                    _seen = [d.id for d in _api.list_datasets(author=_org)]
                except Exception:
                    _seen = []
                _why = (f"**Token OK (user `{_who}`), but `{HF_DATASET_REPO}` is not visible to it.** "
                        + (f"Datasets this token sees in `{_org}`: " + ", ".join(f"`{d}`" for d in _seen) + ". Copy the exact id into `HF_DATASET_REPO`."
                           if _seen else f"It sees no datasets in `{_org}`: give the token read access to that organization, or use a Read-type token."))
            mo.stop(True, mo.md(_why + f"\n\n`{type(_err).__name__}`: {str(_err).splitlines()[0]}"))
    # the dataset may hold the files inside a zip (as on the PC): pull them out, wherever they sit in it
    if len(find_data_files()) < len(DATA_FILE_NAMES):
        import shutil as _shutil
        for _zp in sorted(DATA_DIR.rglob("*.zip")):
            with zipfile.ZipFile(_zp) as _zz:
                for _m in _zz.infolist():
                    _name = _m.filename.replace("\\", "/").rstrip("/").split("/")[-1]
                    if _name in DATA_FILE_NAMES and _name not in find_data_files():
                        with mo.status.spinner(title=f"extracting {_name} from {_zp.name}"):
                            with _zz.open(_m) as _src, open(DATA_DIR / _name, "wb") as _dst:
                                _shutil.copyfileobj(_src, _dst, 16 * 2**20)
            _zp.unlink()
    DATA_FILES = find_data_files()
    _missing = [n for n in DATA_FILE_NAMES if n not in DATA_FILES]
    if _missing:
        raise FileNotFoundError(f"not in {HF_DATASET_REPO}: {_missing}")

    OUT.mkdir(parents=True, exist_ok=True)
    with mo.status.spinner(title="loading data, normalisation stats, eval crops"):
        D = load_data(DATA_FILES)
        STATE["ST"] = norm_stats(D)
        STATE["EV"] = eval_cache(D)
    mo.md("```\n" + "\n".join([
        *(f"{k:22s} {v.stat().st_size / 2**20:8.0f} MB" for k, v in DATA_FILES.items()),
        "",
        f"device {DEVICE} · amp {AMP_DTYPE if AMP_ON else 'fp32'}",
        f"sites {D['sites']}",
        f"dates per year {({y: len(t) for y, t in D['tidx'].items()})}",
        f"eval crops per site-year {len(STATE['EV'][(D['sites'][0], YEARS[0])][3])} · bins {BINS} · crop {CROP} · patch {PATCH}",
        "norm q05/median/q95 " + str([np.round(STATE["ST"][k].ravel(), 0).tolist() for k in ("q05", "median", "q95")]),
    ]) + "\n```")
    return (D,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Splits
    Each fold tests on 2 sites and validates on 2; label ratios pick nested sets of blocks inside the 6 train sites.
    """)
    return


@app.cell
def _(D, IGNORE, mo, train_units):
    _rows = ["| fold | test | val | train | blocks per ratio (seed 0) | labelled pixel-years at 100 % |", "|---|---|---|---|---|---:|"]
    for _f, _sp in enumerate(D["splits"]["folds"]):
        _nb = " / ".join(f"{len(D['splits']['subsets'][str(_f)]['0'][r])}" for r in ("0.05", "0.1", "0.5", "1.0"))
        _u, _l = train_units(D, _f, "1.0", 0)
        _rows.append(f"| {_f} | {', '.join(_sp['test'])} | {', '.join(_sp['val'])} | {', '.join(_sp['train'])} | {_nb} | "
                     f"{sum(int((v != IGNORE).sum()) for v in _l.values()):,} |")
    mo.md("\n".join(_rows))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Sites
    Coffee share among labelled pixels per year, and the share of clear (SCL 4–7) pixels over all dates.
    """)
    return


@app.cell
def _(D, IGNORE, YEARS, mo):
    _rows = ["| site | size | labelled px | coffee 2023 | coffee 2024 | coffee 2025 | clear |", "|---|---|---:|---:|---:|---:|---:|"]
    for _p in D["sites"]:
        _d = D["data"][_p]
        _lab = _d["label"]
        _share = [100 * float((_lab[i] == 1).sum()) / max(int((_lab[i] != IGNORE).sum()), 1) for i in range(len(YEARS))]
        _rows.append(f"| {_p} | {_lab.shape[1]}×{_lab.shape[2]} | {int((_lab[0] != IGNORE).sum()):,} | "
                     + " | ".join(f"{s:.0f} %" for s in _share) + f" | {100 * _d['clear'].mean():.0f} % |")
    mo.md("\n".join(_rows))
    return


@app.cell
def _(D, YEARS, np, plt):
    _fig, _ax = plt.subplots(figsize=(9, 3))
    for _y in YEARS:
        _t = D["tidx"][_y]
        _cf = np.mean([D["data"][p]["clear"][_t].mean((1, 2)) for p in D["sites"]], 0)
        _ax.plot(D["doy"][_t], 100 * _cf, ".-", label=_y)
    _ax.set_xlabel("day of year")
    _ax.set_ylabel("clear pixels (%)")
    _ax.set_title("Cloud: mean clear share per acquisition, 10 labelled sites")
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(D, IGNORE, YEARS, np, plt):
    _show = D["sites"][:4]
    _fig, _axs = plt.subplots(2, len(_show), figsize=(2.6 * len(_show), 5.2), squeeze=False)
    _yi = 1
    _t_year = D["tidx"][YEARS[_yi]]
    for _c, _p in enumerate(_show):
        _d = D["data"][_p]
        _tb = _t_year[int(np.argmax(_d["clear"][_t_year].mean((1, 2))))]
        _x = _d["X"][_tb, [2, 1, 0]].astype("f4").transpose(1, 2, 0)
        _axs[0, _c].imshow(np.clip(_x / max(np.percentile(_x, 98), 1e-6), 0, 1))
        _axs[0, _c].set_title(f"{_p} · DOY {D['doy'][_tb]}", fontsize=9)
        _lab = np.ma.masked_equal(_d["label"][_yi], IGNORE)
        _axs[1, _c].imshow(_lab, cmap="YlGn", vmin=0, vmax=1, interpolation="nearest")
        _axs[1, _c].set_title(f"label {YEARS[_yi]} (green = coffee)", fontsize=9)
    for _a in _axs.ravel():
        _a.axis("off")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pretraining
    MAE with 75 % masking (spatial tubes 25 %, random time steps 25 %, contiguous time blocks 50 %), 150 epochs, batch 72,
    on the 120 unlabelled windows × 3 years. Fully cloudy tokens and empty dates are ignored by attention and by the loss.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Continue from a previous session
    Drop the session bundle downloaded at the end of an earlier session, with the same settings, before pressing ② and ③.
    The encoder, LP heads and finished runs are restored and ③ carries on from the next run.
    """)
    return


@app.cell
def _(mo):
    ui_restore = mo.ui.file(filetypes=[".zip"], multiple=True, kind="area", max_size=2_000_000_000,
                            label="drop a session bundle or weights zip")
    ui_restore
    return (ui_restore,)


@app.cell
def _(OUT, Path, io, mo, ui_restore, zipfile):
    restore_notes = []
    OUT.mkdir(parents=True, exist_ok=True)


    def _extract(z, label):
        _n = 0
        for _zn in z.namelist():
            _rel = Path(_zn)
            if _zn.endswith("/") or _rel.is_absolute() or ".." in _rel.parts:
                continue
            _dst = OUT / _rel
            if _dst.exists() and _rel.parts[0] != "weights":
                continue
            _dst.parent.mkdir(parents=True, exist_ok=True)
            _dst.write_bytes(z.read(_zn))
            _n += 1
        restore_notes.append(f"- `{label}`: {_n} files restored")


    for _up in ui_restore.value:
        try:
            with zipfile.ZipFile(io.BytesIO(_up.contents)) as _z:
                _extract(_z, _up.name)
        except Exception as _err:
            restore_notes.append(f"- `{_up.name}`: could not read ({type(_err).__name__})")



    _on_disk = len(list((OUT / "runs").glob("*.json"))) if (OUT / "runs").exists() else 0
    mo.md("\n".join(restore_notes) + f"\n\n{_on_disk} finished runs on disk." if restore_notes else f"*Nothing restored.* {_on_disk} finished runs on disk.")
    return (restore_notes,)


@app.cell
def _(mo):
    run_pre = mo.ui.run_button(label="② Pretrain")
    run_pre
    return (run_pre,)


@app.cell
def _(
    D,
    FOLDS,
    PRETRAINED_ENCODER,
    Path,
    REGIMES,
    ckpt_path,
    mo,
    pretrain,
    restore_notes,
    run_pre,
    torch,
):
    _need_enc = any(m in REGIMES for m in ("LP", "FT"))
    _ready = (not _need_enc) or bool(PRETRAINED_ENCODER) or all(ckpt_path(f).exists() for f in FOLDS)
    _ = restore_notes  # restore first, so restored encoders count as ready
    mo.stop(not run_pre.value and not _ready, mo.md("*Press ② to pretrain.*"))
    ENC_PATHS, HISTORIES = {}, {}
    if _need_enc:
        for _f in FOLDS:
            ENC_PATHS[_f] = Path(PRETRAINED_ENCODER) if PRETRAINED_ENCODER else pretrain(D, _f)
            _st = torch.load(ENC_PATHS[_f], map_location="cpu", weights_only=False)
            HISTORIES[ENC_PATHS[_f].name] = _st.get("history", [])
    mo.md("\n".join(f"- fold {f}: `{p.name}`" for f, p in ENC_PATHS.items()) or "*SL only: no encoder needed.*")
    return ENC_PATHS, HISTORIES


@app.cell
def _(HISTORIES, mo, plt):
    mo.stop(not HISTORIES)
    _fig, _ax = plt.subplots(figsize=(6, 3))
    for _k, _h in HISTORIES.items():
        _ax.plot(range(1, len(_h) + 1), _h, label=_k[:40])
    _ax.set_xlabel("epoch")
    _ax.set_ylabel("masked L1 (group-normalised)")
    _ax.legend(fontsize=7)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Downstream
    LP (frozen encoder), FT (LP-FT, layer decay 0.75, EMA) and SL (same ViT from scratch), on identical label subsets per fold, ratio and seed.
    Validation every 5 epochs on the val sites; early stop after half the epochs when val mIoU stalls for 30 %; test with the best weights on every labelled pixel of the test sites.
    """)
    return


@app.cell
def _(FOLDS, N_RUNS, OUT, RATIOS, REGIMES, SEEDS, json, mo, np, restore_notes):
    _ = restore_notes
    _done = [(m, f, r, s) for f in FOLDS for r in RATIOS for s in SEEDS for m in REGIMES
             if (OUT / "runs" / f"{m}_f{f}_r{r}_s{s}.json").exists()]
    _mins = [json.loads((OUT / "runs" / f"{m}_f{f}_r{r}_s{s}.json").read_text())["minutes"] for m, f, r, s in _done]
    _eta = (N_RUNS - len(_done)) * (np.mean(_mins) if _mins else float("nan"))
    mo.md(f"**{len(_done)} / {N_RUNS} runs on disk.** "
          + (f"Mean {np.mean(_mins):.1f} min per finished run, so about {_eta / 60:.1f} h left." if _mins else "No timing yet: the first runs give an estimate.")
          + " molab wipes the session after 12 h or 90 min idle.")
    return


@app.cell
def _(mo):
    run_dn = mo.ui.run_button(label="③ Run downstream")
    run_dn
    return (run_dn,)


@app.cell
def _(
    D,
    ENC_PATHS,
    FOLDS,
    N_RUNS,
    OUT,
    RATIOS,
    REGIMES,
    SEEDS,
    SESSION_T0,
    mo,
    run_dn,
    run_one,
    summarize,
    time,
):
    _paths = {(m, f, r, s): OUT / "runs" / f"{m}_f{f}_r{r}_s{s}.json" for f in FOLDS for r in RATIOS for s in SEEDS for m in REGIMES}
    DN_READY = all(p.exists() for p in _paths.values())
    mo.stop(not run_dn.value and not DN_READY,
            mo.md(f"*{sum(p.exists() for p in _paths.values())}/{N_RUNS} runs on disk. Press ③ to "
                  + ("resume.*" if any(p.exists() for p in _paths.values()) else "start.*")))
    results = []
    with mo.status.progress_bar(total=N_RUNS, title="downstream", remove_on_exit=False) as _bar:
        for _f in FOLDS:
            for _r in RATIOS:
                for _s in SEEDS:
                    for _m in REGIMES:
                        _bar.update(increment=0, subtitle=f"{_m} · fold {_f} · ratio {_r} · seed {_s}")
                        _res = run_one(D, _m, _f, _r, _s, ENC_PATHS.get(_f))
                        results.append(_res)
                        print(f"{_m} f{_f} r{_r} s{_s}: test mIoU {_res['test']['mIoU']:.4f}  IoU coffee {_res['test']['IoU_coffee']:.4f}"
                              f"  | {_res['epochs_run']}/{_res['epochs_planned']} ep, best {_res['best_epoch']} | {_res['minutes']} min")
                        _bar.update()
            summarize(results)
    mo.md(f"**{len(results)} runs done.** Total session time {(time.time() - SESSION_T0) / 3600:.1f} h.")
    return (results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Results
    ### Summary
    Mean ± std over folds × seeds (%). Also saved as `summary_mean_std.csv` and `all_runs.csv`.
    """)
    return


@app.cell
def _(mo, results, summarize):
    summary = summarize(results)
    mo.ui.table(summary.reset_index(), selection=None, pagination=False)
    return


@app.cell
def _(RATIOS, REGIMES, np, plt, results):
    _fig, _axs = plt.subplots(1, 3, figsize=(14, 3.6))
    _styles = {"LP": ("-o", "#4c72b0"), "FT": ("-s", "#dd8452"), "SL": ("--^", "#55a868")}
    for _ax, _metric in zip(_axs, ["mIoU", "IoU_coffee", "macro_F1"]):
        for _m in REGIMES:
            _xs, _mu, _sd = [], [], []
            for _r in RATIOS:
                _v = [100 * x["test"][_metric] for x in results if x["regime"] == _m and x["ratio"] == _r]
                if _v:
                    _xs.append(100 * float(_r))
                    _mu.append(np.mean(_v))
                    _sd.append(np.std(_v))
            _ax.errorbar(_xs, _mu, yerr=_sd, fmt=_styles[_m][0], color=_styles[_m][1], capsize=3, label=_m)
        _ax.set_xscale("log")
        _ax.set_xticks([100 * float(r) for r in RATIOS], [f"{100 * float(r):g}" for r in RATIOS])
        _ax.set_xlabel("labelled blocks (%)")
        _ax.set_title(_metric)
        _ax.grid(alpha=0.3)
    _axs[0].legend()
    _fig.suptitle("Label efficiency on ViCoffee (test sites, mean ± std over folds × seeds)")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Per year and confusion matrix
    """)
    return


@app.cell
def _(RATIOS, mo):
    ui_res_ratio = mo.ui.dropdown(RATIOS, value=RATIOS[-1], label="label ratio")
    ui_res_ratio
    return (ui_res_ratio,)


@app.cell
def _(REGIMES, YEARS, mo, np, results, ui_res_ratio):
    _rows = ["| regime | runs | " + " | ".join(f"mIoU {y}" for y in YEARS) + " | coffee: TP / FN / FP / TN (% of labelled px) |",
             "|---|---:|" + "---:|" * len(YEARS) + "---|"]
    for _m in REGIMES:
        _sel = [x for x in results if x["regime"] == _m and x["ratio"] == ui_res_ratio.value]
        if not _sel:
            continue
        _years = [f"{100 * np.mean([x['test']['mIoU_by_year'][y] for x in _sel]):.1f}" for y in YEARS]
        _cm = np.sum([np.array(x["test"]["confusion"], dtype=float) for x in _sel], 0)
        _cm = 100 * _cm / _cm.sum()
        _rows.append(f"| {_m} | {len(_sel)} | " + " | ".join(_years) + f" | {_cm[1, 1]:.1f} / {_cm[1, 0]:.1f} / {_cm[0, 1]:.1f} / {_cm[0, 0]:.1f} |")
    mo.md("\n".join(_rows))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Download
    molab keeps only files uploaded through the file browser: everything this notebook writes is lost when the session ends
    (after 12 h, or 90 min idle). The **session bundle** holds the encoder, LP heads, finished runs and CSVs; drop it into
    *Continue from a previous session* next time. **Weights** holds the best FT / SL / LP model per fold and ratio.
    """)
    return


@app.cell
def _(mo):
    run_dl = mo.ui.run_button(label="prepare downloads")
    mo.vstack([mo.md("Builds the files from what is on disk now. If ③ is running, interrupt it first (finished runs are already saved)."), run_dl])
    return (run_dl,)


@app.cell
def _(OUT, VERSION, io, mo, run_dl, time, zipfile):
    mo.stop(not run_dl.value, mo.md("*Press **prepare downloads**.*"))


    def _zip(include):
        _buf = io.BytesIO()
        with zipfile.ZipFile(_buf, "w") as _z:
            for _p in sorted(OUT.rglob("*")):
                _rel = _p.relative_to(OUT)
                if _p.is_file() and include(_rel):
                    _z.write(_p, _rel.as_posix(), compress_type=zipfile.ZIP_STORED if _p.suffix == ".pt" else zipfile.ZIP_DEFLATED)
        return _buf.getvalue()


    _bundle = _zip(lambda r: r.parts[0] != "weights")
    _weights = _zip(lambda r: r.parts[0] == "weights")
    _stamp = time.strftime("%Y%m%d_%H%M")
    mo.vstack([
        mo.download(_bundle, filename=f"vicoffee_maestro_v{VERSION}_session_{_stamp}.zip", label=f"session bundle ({len(_bundle) / 2**20:.0f} MB)"),
        mo.download(_weights, filename=f"vicoffee_maestro_v{VERSION}_weights_{_stamp}.zip", label=f"weights ({len(_weights) / 2**20:.0f} MB)"),
        *([mo.download((OUT / "summary_mean_std.csv").read_bytes(), filename="summary_mean_std.csv", label="summary_mean_std.csv")]
          if (OUT / "summary_mean_std.csv").exists() else []),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reloading weights
    ```python
    st = torch.load("FT_f0_r1.0_best.pt", map_location="cpu", weights_only=False)
    enc = Encoder(N_BANDS, st["patch"], st["crop"], **PRESETS[st["encoder_size"]])
    m = Segmenter(enc, st["n_classes"], freeze=st["regime"] == "LP")
    m.load_state_dict(st["model"])
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
