import os
import warnings
import pandas as pd
import numpy as np
import torch
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import torch.nn as nn
import featuretools as ft
import copy
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, LSTM, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.utils import resample
KNOWN_ATTACKS = ['guess_passwd', 'ipsweep', 'neptune', 'nmap', 'portsweep', 'smurf', 'teardrop','xsnoop']
UNKNOWN_ATTACKS = [
    'back', 'land', 'pod', 'satan', 'ftp_write', 'spy',
    'multihop', 'phf', 'imap','warezmaster']

N_KNOWN = len(KNOWN_ATTACKS)
N_KNOWN_TOTAL = N_KNOWN + 1
N_STD_CLASSES = N_KNOWN + 2
KNOWN_FG_MAP = {a: i + 1 for i, a in enumerate(KNOWN_ATTACKS)}
UNKNOWN_LABEL = N_KNOWN + 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    tf.random.set_seed(seed)

def fg_to_3class(preds_fg):
    out = np.ones_like(preds_fg, dtype=np.int64)
    out[preds_fg == 0] = 0
    out[preds_fg == UNKNOWN_LABEL] = 2
    return out

def load_data(file_path):
    cic_columns =  [
    'protocol_type', 'service', 'flag',
    'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins',
    'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files',
    'num_outbound_cmds', 'is_host_login', 'is_guest_login',
    'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
    'label', 'difficulty_level',
]

    return pd.read_csv(file_path, header=None, names=cic_columns)

def fit_clean_stats(df):
    df = df.copy()
    numeric_cols = df.select_dtypes(exclude=['object']).columns.tolist()
    stats = {}
    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
        Q1, Q3 = (series.quantile(0.25), series.quantile(0.75))
        IQR = Q3 - Q1
        stats[col] = {'low': Q1 - 1.5 * IQR, 'high': Q3 + 1.5 * IQR, 'median': series.median()}
    return stats

def clean_data(df, stats=None):
    df = df.copy()
    if stats is None:
        stats = fit_clean_stats(df)
    numeric_cols = df.select_dtypes(exclude=['object']).columns
    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
        if col in stats:
            low, high, median = (stats[col]['low'], stats[col]['high'], stats[col]['median'])
            series = series.where(series.between(low, high), median)
        df[col] = series.fillna(stats.get(col, {}).get('median', 0.0))
    return (df.dropna(), stats)

def label_fine_grained(df, include_unknown=True):

    def _fg(raw):
        s = str(raw).strip()
        if s in ('BENIGN', 'normal'):
            return 0
        if s in KNOWN_FG_MAP:
            return KNOWN_FG_MAP[s]
        if s in UNKNOWN_ATTACKS:
            return UNKNOWN_LABEL if include_unknown else -1
        return -1

    def _3c(raw):
        s = str(raw).strip()
        if s in ('BENIGN', 'normal'):
            return 0
        if s in KNOWN_ATTACKS:
            return 1
        if s in UNKNOWN_ATTACKS:
            return 2
        return -1
    df = df.copy()
    df['label_fg'] = df['label'].apply(_fg)
    df['label_3c'] = df['label'].apply(_3c)
    df = df[(df['label_fg'] != -1) & (df['label_3c'] != -1)].reset_index(drop=True)
    return df

def automated_feature_engineering_train_test(df_train, df_test):
    train = df_train.copy().reset_index(drop=True)
    test = df_test.copy().reset_index(drop=True)
    cat_cols = train.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = [c for c in train.columns if c not in cat_cols]
    for col in num_cols:
        train[col] = pd.to_numeric(train[col], errors='coerce').fillna(0.0)
        test[col] = pd.to_numeric(test[col], errors='coerce').fillna(0.0)
        train[f'{col}_sq'] = train[col] ** 2
        test[f'{col}_sq'] = test[col] ** 2
        train[f'{col}_logabs'] = np.sign(train[col]) * np.log1p(np.abs(train[col]))
        test[f'{col}_logabs'] = np.sign(test[col]) * np.log1p(np.abs(test[col]))
    pair_specs = [('src_bytes', 'dst_bytes', 'bytes'), ('count', 'srv_count', 'count_srv'), ('serror_rate', 'srv_serror_rate', 'serror'), ('rerror_rate', 'srv_rerror_rate', 'rerror'), ('dst_host_count', 'dst_host_srv_count', 'host_count'), ('dst_host_serror_rate', 'dst_host_srv_serror_rate', 'host_serror'), ('dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'host_rerror'), ('same_srv_rate', 'diff_srv_rate', 'srv_mix')]
    for a, b, prefix in pair_specs:
        if a in train.columns and b in train.columns:
            train[f'{prefix}_sum'] = train[a] + train[b]
            test[f'{prefix}_sum'] = test[a] + test[b]
            train[f'{prefix}_diff'] = train[a] - train[b]
            test[f'{prefix}_diff'] = test[a] - test[b]
            train[f'{prefix}_ratio'] = train[a] / (np.abs(train[b]) + 1.0)
            test[f'{prefix}_ratio'] = test[a] / (np.abs(test[b]) + 1.0)
    if 'src_bytes' in train.columns and 'dst_bytes' in train.columns:
        train['bytes_total_log'] = np.log1p(np.abs(train['src_bytes'] + train['dst_bytes']))
        test['bytes_total_log'] = np.log1p(np.abs(test['src_bytes'] + test['dst_bytes']))
    if 'count' in train.columns and 'srv_diff_host_rate' in train.columns:
        train['count_host_interaction'] = train['count'] * train['srv_diff_host_rate']
        test['count_host_interaction'] = test['count'] * test['srv_diff_host_rate']
    train_enc = pd.get_dummies(train, columns=cat_cols, dummy_na=False)
    test_enc = pd.get_dummies(test, columns=cat_cols, dummy_na=False)
    train_enc, test_enc = train_enc.align(test_enc, join='outer', axis=1, fill_value=0)
    train_enc = train_enc.reindex(sorted(train_enc.columns), axis=1)
    test_enc = test_enc.reindex(train_enc.columns, axis=1, fill_value=0)
    return (train_enc.values.astype(np.float32), test_enc.values.astype(np.float32), train_enc.columns)

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    return (np.clip(scaler.fit_transform(X_train), -5, 5), np.clip(scaler.transform(X_test), -5, 5), scaler)

def compute_class_weights(y, n_classes, benign_boost=1.0):
    counts = np.bincount(y, minlength=n_classes).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    weights = len(y) / (n_classes * counts)
    weights = weights / weights.mean()
    if benign_boost != 1.0:
        weights[0] *= benign_boost
        weights = weights / weights.mean()
    for c in range(n_classes):
        nm = 'BENIGN' if c == 0 else KNOWN_ATTACKS[c - 1] if c <= N_KNOWN else 'UNKNOWN'
    return torch.FloatTensor(weights)

def mixup_batch(x, y_onehot, alpha=0.2):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[idx]
    y_mix = lam * y_onehot + (1 - lam) * y_onehot[idx]
    return (x_mix, y_mix)

def mixup_criterion(logits, y_soft):
    log_prob = torch.nn.functional.log_softmax(logits, dim=1)
    return -(y_soft * log_prob).sum(dim=1).mean()

class SWAHelper:

    def __init__(self, model):
        self.swa_state = None
        self.n = 0
        self.model = model

    def update(self):
        state = {k: v.clone().float() for k, v in self.model.state_dict().items()}
        if self.swa_state is None:
            self.swa_state = state
        else:
            for k in self.swa_state:
                self.swa_state[k] += state[k]
        self.n += 1

    def apply(self):
        if self.swa_state is None or self.n == 0:
            return
        avg_state = {k: v / self.n for k, v in self.swa_state.items()}
        self.model.load_state_dict(avg_state)

    def update_bn(self, loader):
        pass

class EMAHelper:

    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads, num_scales):
        super().__init__()
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.num_scales = num_scales
        self.WQ = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_scales)])
        self.WK = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_scales)])
        self.WV = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_scales)])

    def split_heads(self, x, B, L):
        return x.view(B, L, self.num_heads, self.depth).permute(0, 2, 1, 3)

    def forward(self, q, k, v):
        B, L, _ = q.size()
        e_ij = torch.zeros(B, self.num_heads, L, L, device=q.device)
        v_outputs = []
        for m in range(self.num_scales):
            Q = self.split_heads(self.WQ[m](q), B, L)
            K = self.split_heads(self.WK[m](k), B, L)
            V = self.split_heads(self.WV[m](v), B, L)
            dk = torch.tensor(self.depth, dtype=torch.float32, device=q.device)
            e_ij += torch.matmul(Q, K.permute(0, 1, 3, 2)) / torch.sqrt(dk)
            v_outputs.append(V)
        return (e_ij, v_outputs)

class MultiScaleAdaptiveGatedFeatureExtraction(nn.Module):

    def __init__(self, d_model, num_heads, num_scales):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_scales = num_scales
        self.total_heads = num_heads * num_scales
        self.sub_dim = d_model // num_scales
        self.head_dim = self.sub_dim // num_heads
        self.attn_layers = nn.ModuleList([MultiHeadAttention(self.sub_dim, num_heads, 1) for _ in range(num_scales)])
        for nm in ['alpha', 'beta', 'delta', 'lambda_param', 'theta', 'varphi', 'gamma', 'eta']:
            setattr(self, nm, nn.Parameter(torch.tensor(0.5)))
        self.WC = nn.Linear(self.head_dim, self.head_dim)
        self.Wg = nn.ModuleList([nn.Linear(self.head_dim, 1) for _ in range(self.total_heads)])
        self.WO = nn.Linear(d_model, d_model)
        self.WjA = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim) for _ in range(self.total_heads)])
        self.WjO = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim) for _ in range(self.total_heads)])
        self.W1 = nn.Linear(d_model, d_model * 4)
        self.b1 = nn.Parameter(torch.zeros(d_model * 4))
        self.W2 = nn.Linear(d_model * 4, d_model)
        self.b2 = nn.Parameter(torch.zeros(d_model))
        self.K = 5
        self.m = 10
        self.W_alpha = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(self.K)])
        self.W_theta = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(self.m)])

    def forward(self, x):
        B, L, _ = x.size()
        xr = x.view(B, L, self.num_scales, self.sub_dim).permute(0, 2, 1, 3)
        all_scale = []
        for s in range(self.num_scales):
            sf = xr[:, s, :, :]
            e_ij, v_out = self.attn_layers[s](sf, sf, sf)
            a_ij = torch.softmax(e_ij, dim=-1)
            h = []
            for i in range(L):
                Xi = sf[:, i, :].view(B, self.num_heads, self.head_dim)
                hi = torch.zeros(B, self.num_heads, self.head_dim, device=x.device)
                for j in range(L):
                    Xj = sf[:, j, :].view(B, self.num_heads, self.head_dim)
                    ctx = torch.einsum('bhd,bdh->bh', self.WC(Xj), Xi.permute(0, 2, 1)).unsqueeze(-1)
                    aij = a_ij[:, :, i, j].unsqueeze(-1)
                    vc = v_out[0][:, :, j, :].unsqueeze(2)
                    att = (self.alpha * aij + (1 - self.alpha) * ctx).unsqueeze(-1)
                    add = (self.beta * self.gamma + self.delta * self.eta) * torch.ones(B, self.num_heads, self.head_dim, device=x.device)
                    hi += (att * vc).sum(dim=-1) + add
                h.append(hi.unsqueeze(1))
            all_scale.append(torch.cat(h, dim=1).view(B, L, self.sub_dim).unsqueeze(1))
        hc = torch.cat(all_scale, dim=1).permute(0, 2, 1, 3).contiguous().view(B, L, -1)
        go = [torch.sigmoid(self.Wg[i](hc[:, :, i * self.head_dim:(i + 1) * self.head_dim])) * hc[:, :, i * self.head_dim:(i + 1) * self.head_dim] for i in range(self.total_heads)]
        O = torch.zeros_like(hc)
        for j in range(self.total_heads):
            hj = go[j]
            f = torch.tanh(self.WjO[j](hj) + self.WjA[j](hj))
            w = self.WO.weight[j * self.head_dim:(j + 1) * self.head_dim, j * self.head_dim:(j + 1) * self.head_dim]
            O[:, :, j * self.head_dim:(j + 1) * self.head_dim] = torch.matmul(hj * f, w)
        O = O + self.WO.bias
        pos = torch.arange(L, dtype=torch.float32, device=x.device)
        dt = torch.exp(torch.arange(0, self.d_model, 2, device=x.device).float() * (-torch.log(torch.tensor(10000.0)) / self.d_model))
        pe = torch.zeros(L, self.d_model, device=x.device)
        pe[:, 0::2] = torch.sin(pos.unsqueeze(1) * dt)
        pe[:, 1::2] = torch.cos(pos.unsqueeze(1) * dt)
        Oe = O + self.lambda_param * (hc * pe.unsqueeze(0))
        X = Oe
        X1 = nn.functional.relu(self.W1(X) + self.b1)
        P = pe.unsqueeze(0).expand(B, -1, -1)
        pt = torch.zeros_like(X1)
        for l in range(min(self.m, L)):
            pf = self.W_theta[l](P[:, l % L:l % L + 1, :])
            th = torch.softmax(torch.bmm(self.W_theta[l](X), pf.transpose(1, 2)), dim=-1)
            pc = torch.bmm(th.transpose(1, 2), pf.expand(-1, X1.size(1), -1))
            pt += self.varphi * pc.expand(-1, X1.size(1), -1).repeat(1, 1, 4)
        it = torch.zeros_like(X1)
        for k in range(self.K):
            a = torch.softmax(torch.bmm(X, self.W_alpha[k](X).transpose(1, 2)), dim=-1)
            aX = torch.bmm(a, X).repeat(1, 1, 4)
            it += self.theta * aX[:, :, :X1.size(2)]
        return self.W2(X1 + it + pt) + self.b2

class MetaLearner(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.hyper_attn_heads = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 8))
        self.hyper_ffn_dim = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 512))

    def forward(self, x):
        return (self.hyper_attn_heads(x.mean(dim=1)), self.hyper_ffn_dim(x.mean(dim=1)))

class ResidualBlock(nn.Module):

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim), nn.LayerNorm(dim))

    def forward(self, x):
        return torch.nn.functional.gelu(x + self.block(x))

class CrossLayer(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x0, x):
        xw = torch.sum(x * self.weight, dim=1, keepdim=True)
        return x0 * xw + self.bias + x

def focal_cross_entropy(logits, labels, class_weights, gamma=1.5, label_smoothing=0.0):
    ce = torch.nn.functional.cross_entropy(logits, labels, weight=class_weights, reduction='none', label_smoothing=label_smoothing)
    pt = torch.softmax(logits, dim=1).gather(1, labels.unsqueeze(1)).squeeze(1)
    pt = pt.clamp(1e-06, 1 - 1e-06)
    return ((1 - pt) ** gamma * ce).mean()

def center_compactness_loss(embeddings, labels, centers):
    target = centers[labels]
    return ((embeddings - target) ** 2).sum(dim=1).mean()

def logit_margin_loss(logits, labels, margin=1.0):
    true_logit = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
    other_mask = torch.nn.functional.one_hot(labels, num_classes=logits.size(1)).bool()
    max_other = logits.masked_fill(other_mask, -1000000000.0).max(dim=1).values
    return torch.relu(margin - (true_logit - max_other)).mean()

def uniform_confidence_loss(logits):
    probs = torch.softmax(logits, dim=1)
    target = torch.full_like(probs, 1.0 / probs.size(1))
    return ((probs - target) ** 2).sum(dim=1).mean()

def synthesize_boundary_outliers(x, y):
    idx = torch.randperm(x.size(0), device=x.device)
    mask = y != y[idx]
    if mask.sum().item() == 0:
        return None
    x_mid = 0.5 * x[mask] + 0.5 * x[idx][mask]
    return x_mid + 0.01 * torch.randn_like(x_mid)

def synthesize_boundary_outliers_np(X, y, n_samples=4096, seed=42):
    rng = np.random.default_rng(seed)
    if len(X) < 2:
        return X[:0]
    out = []
    trials = 0
    max_trials = max(n_samples * 10, 1000)
    while len(out) < n_samples and trials < max_trials:
        i = int(rng.integers(0, len(X)))
        j = int(rng.integers(0, len(X)))
        trials += 1
        if y[i] == y[j]:
            continue
        mode = int(rng.integers(0, 3))
        if mode == 0:
            lam = float(rng.uniform(0.35, 0.65))
            mixed = lam * X[i] + (1.0 - lam) * X[j]
            mixed = mixed + rng.normal(0.0, 0.01, size=mixed.shape)
        elif mode == 1:
            lam = float(rng.uniform(1.05, 1.3))
            mixed = lam * X[i] + (1.0 - lam) * X[j]
            mixed = mixed + rng.normal(0.0, 0.015, size=mixed.shape)
        else:
            diff = np.abs(X[i] - X[j])
            mask = rng.random(X[i].shape) < 0.2
            mixed = X[i].copy()
            mixed[mask] = mixed[mask] + rng.normal(0.0, 0.5, size=mask.sum()) * (diff[mask] + 0.5)
        out.append(mixed.astype(np.float32, copy=False))
    if not out:
        return X[:0]
    return np.asarray(out, dtype=np.float32)

class EnhancedTransformer(nn.Module):

    def __init__(self, input_dim, num_classes, dropout1=0.3, dropout2=0.2):
        super().__init__()
        hidden_dim = 1024
        emb_dim = 384
        self.meta_learner = MetaLearner(input_dim)
        self.feature_gate = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, input_dim * 2), nn.GELU(), nn.Linear(input_dim * 2, input_dim), nn.Sigmoid())
        self.cross_layers = nn.ModuleList([CrossLayer(input_dim) for _ in range(3)])
        self.input_embedding = nn.Sequential(nn.Linear(input_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout1))
        self.res_blocks = nn.Sequential(ResidualBlock(hidden_dim, dropout=dropout2), ResidualBlock(hidden_dim, dropout=dropout2), ResidualBlock(hidden_dim, dropout=dropout2), ResidualBlock(hidden_dim, dropout=dropout2), ResidualBlock(hidden_dim, dropout=dropout2))
        self.feature_extraction = nn.Sequential(nn.Linear(hidden_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout2), nn.Linear(512, emb_dim))
        self.output_layer = nn.Linear(emb_dim, num_classes)
        self.aux_head = nn.Linear(hidden_dim, num_classes)
        self.linear_skip = nn.Linear(input_dim, num_classes)
        self.ood_head = nn.Sequential(nn.Linear(emb_dim, 128), nn.GELU(), nn.Linear(128, 1))
        self.class_centers = nn.Parameter(torch.randn(num_classes, emb_dim) * 0.02)

    def forward(self, x, return_features=False):
        if x.dim() == 3:
            x = x.squeeze(1)
        raw = x
        gate = self.feature_gate(x)
        x = x * (1.0 + gate)
        cross = x
        for layer in self.cross_layers:
            cross = layer(x, cross)
        x = torch.cat([x, cross], dim=1)
        x = self.input_embedding(x)
        x = self.res_blocks(x)
        aux_logits = self.aux_head(x)
        emb = self.feature_extraction(x)
        proto_logits = 10.0 * torch.matmul(torch.nn.functional.normalize(emb, dim=1), torch.nn.functional.normalize(self.class_centers, dim=1).t())
        ood_logit = self.ood_head(emb).squeeze(1)
        logits = self.output_layer(emb) + 0.2 * aux_logits + 0.15 * proto_logits + 0.08 * self.linear_skip(raw)
        if return_features:
            return (logits, emb, ood_logit)
        return logits

    def get_embeddings(self, X_np, batch_size=512):
        self.eval()
        out = []
        device = next(self.parameters()).device
        with torch.no_grad():
            for i in range(0, len(X_np), batch_size):
                x = torch.FloatTensor(X_np[i:i + batch_size]).to(device)
                if x.dim() == 3:
                    x = x.squeeze(1)
                gate = self.feature_gate(x)
                x = x * (1.0 + gate)
                cross = x
                for layer in self.cross_layers:
                    cross = layer(x, cross)
                x = torch.cat([x, cross], dim=1)
                x = self.input_embedding(x)
                x = self.res_blocks(x)
                x = self.feature_extraction(x)
                out.append(x.cpu().numpy())
        return np.vstack(out)

    def get_logits(self, X_np, batch_size=512):
        self.eval()
        out = []
        device = next(self.parameters()).device
        with torch.no_grad():
            for i in range(0, len(X_np), batch_size):
                xb = torch.FloatTensor(X_np[i:i + batch_size]).to(device)
                out.append(self(xb).cpu().numpy())
        return np.vstack(out)

    def get_ood_scores(self, X_np, batch_size=512):
        self.eval()
        out = []
        device = next(self.parameters()).device
        with torch.no_grad():
            for i in range(0, len(X_np), batch_size):
                xb = torch.FloatTensor(X_np[i:i + batch_size]).to(device)
                if xb.dim() == 3:
                    xb = xb.squeeze(1)
                gate = self.feature_gate(xb)
                x = xb * (1.0 + gate)
                cross = x
                for layer in self.cross_layers:
                    cross = layer(x, cross)
                x = torch.cat([x, cross], dim=1)
                x = self.input_embedding(x)
                x = self.res_blocks(x)
                emb = self.feature_extraction(x)
                out.append(torch.sigmoid(self.ood_head(emb)).squeeze(1).cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0,), dtype=np.float32)

    @staticmethod
    def train_model_mixup(model, train_loader, test_loader, class_weights, optimizer, scheduler, epochs, n_classes, noise_std=0.0, mixup_alpha=0.2, swa=None, swa_start=0, label='OursModel', focal_gamma=1.5, label_smoothing=0.0, center_loss_weight=0.0, boundary_ood_weight=0.0, margin_loss_weight=0.0, proxy_ood_loader=None, proxy_select_weight=0.0):
        model.to(DEVICE)
        class_weights = class_weights.to(DEVICE)
        tr_hist, te_hist = (_empty_metric_history(), _empty_metric_history())
        ema = EMAHelper(model, decay=0.995)
        best_state, best_score, best_te, best_tr = (None, -1.0, -1.0, -1.0)
        for epoch in range(epochs):
            model.train()
            total_loss = num_batches = 0
            for data, labels in train_loader:
                data = data.to(DEVICE)
                labels = labels.to(DEVICE)
                raw_data = data
                raw_labels = labels
                if noise_std > 0:
                    data = data + torch.randn_like(data) * noise_std
                if mixup_alpha > 0:
                    y_oh = torch.zeros(len(labels), n_classes, device=DEVICE).scatter_(1, labels.unsqueeze(1), 1.0)
                    data, y_soft = mixup_batch(data, y_oh, alpha=mixup_alpha)
                    logits = model(data)
                    log_p = torch.nn.functional.log_softmax(logits, dim=1)
                    prob = torch.softmax(logits, dim=1)
                    pt = (prob * y_soft).sum(dim=1).clamp(1e-06, 1 - 1e-06)
                    w = torch.matmul(y_soft, class_weights)
                    ce = -(y_soft * log_p).sum(dim=1)
                    loss = (ce * (1 - pt) ** focal_gamma * w).mean()
                else:
                    logits, emb, ood_known = model(data, return_features=True)
                    loss = focal_cross_entropy(logits, labels, class_weights, gamma=focal_gamma, label_smoothing=label_smoothing)
                    if center_loss_weight > 0.0:
                        loss = loss + center_loss_weight * center_compactness_loss(emb, labels, model.class_centers)
                    if margin_loss_weight > 0.0:
                        loss = loss + margin_loss_weight * logit_margin_loss(logits, labels, margin=1.2)
                    if boundary_ood_weight > 0.0:
                        loss = loss + 0.2 * boundary_ood_weight * torch.nn.functional.binary_cross_entropy_with_logits(ood_known, torch.zeros_like(ood_known))
                if boundary_ood_weight > 0.0:
                    x_ood = synthesize_boundary_outliers(raw_data, raw_labels)
                    if x_ood is not None:
                        logits_ood, _, ood_fake = model(x_ood, return_features=True)
                        loss = loss + boundary_ood_weight * (uniform_confidence_loss(logits_ood) + 0.35 * torch.nn.functional.binary_cross_entropy_with_logits(ood_fake, torch.ones_like(ood_fake)))
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                ema.update(model)
                total_loss += loss.item()
                num_batches += 1
            scheduler.step()
            avg_loss = total_loss / max(num_batches, 1)
            if swa is not None and epoch >= swa_start:
                swa.update()
            eval_state = ema.state_dict()
            tr_metrics = _metrics_with_state(model, eval_state, train_loader)
            te_metrics = _metrics_with_state(model, eval_state, test_loader)
            tr = tr_metrics['accuracy']
            te = te_metrics['accuracy']
            proxy_score = 0.0
            if proxy_ood_loader is not None and proxy_select_weight > 0.0:
                proxy_score = _proxy_open_score_with_state(model, eval_state, test_loader, proxy_ood_loader)
            select_score = te + proxy_select_weight * proxy_score
            _append_metric_history(tr_hist, tr_metrics)
            _append_metric_history(te_hist, te_metrics)
            if select_score > best_score or (select_score == best_score and (te > best_te or (te == best_te and tr > best_tr))):
                best_score = select_score
                best_te = te
                best_tr = tr
                best_state = eval_state
        if best_state is not None:
            model.load_state_dict(best_state)
        return (tr_hist, te_hist)

def _get_lr(opt):
    return opt.param_groups[0]['lr']

def _acc(model, loader):
    device = next(model.parameters()).device
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, labels in loader:
            data = data.to(device)
            labels = labels.to(device)
            correct += (model(data).argmax(dim=1) == labels).sum().item()
            total += len(labels)
    return correct / max(total, 1)

def _predict_from_loader(model, loader):
    device = next(model.parameters()).device
    model.eval()
    y_true, y_pred = ([], [])
    with torch.no_grad():
        for data, labels in loader:
            data = data.to(device)
            logits = model(data)
            y_pred.append(logits.argmax(dim=1).cpu().numpy())
            y_true.append(labels.numpy())
    if not y_true:
        return (np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64))
    return (np.concatenate(y_true), np.concatenate(y_pred))

def _metrics(model, loader):
    y_true, y_pred = _predict_from_loader(model, loader)
    if len(y_true) == 0:
        return {k: 0.0 for k in _empty_metric_history()}
    return evaluate_predictions(y_true, y_pred)

def _acc_with_state(model, state_dict, loader):
    original = copy.deepcopy(model.state_dict())
    model.load_state_dict(state_dict)
    score = _acc(model, loader)
    model.load_state_dict(original)
    return score

def _metrics_with_state(model, state_dict, loader):
    original = copy.deepcopy(model.state_dict())
    model.load_state_dict(state_dict)
    metrics = _metrics(model, loader)
    model.load_state_dict(original)
    return metrics

def _logits_from_loader(model, loader):
    device = next(model.parameters()).device
    model.eval()
    out = []
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            out.append(model(data).cpu().numpy())
    if not out:
        return np.zeros((0, 1), dtype=np.float32)
    return np.vstack(out)

def _proxy_open_score(model, known_loader, proxy_loader):
    known_logits = _logits_from_loader(model, known_loader)
    proxy_logits = _logits_from_loader(model, proxy_loader)
    if len(known_logits) == 0 or len(proxy_logits) == 0:
        return 0.0
    known_prob = torch.softmax(torch.tensor(known_logits), dim=1).numpy().max(axis=1)
    proxy_prob = torch.softmax(torch.tensor(proxy_logits), dim=1).numpy().max(axis=1)
    known_gap = _top2_gap_from_logits(known_logits)
    proxy_gap = _top2_gap_from_logits(proxy_logits)
    conf_thr = float(np.percentile(known_prob, 12.0))
    gap_thr = float(np.percentile(known_gap, 10.0))
    proxy_reject = np.mean((proxy_prob < conf_thr) | (proxy_gap < gap_thr))
    return float(proxy_reject)

def _proxy_open_score_with_state(model, state_dict, known_loader, proxy_loader):
    original = copy.deepcopy(model.state_dict())
    model.load_state_dict(state_dict)
    score = _proxy_open_score(model, known_loader, proxy_loader)
    model.load_state_dict(original)
    return score

def _make_loader(X, y, batch_size=128, shuffle=False):
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.FloatTensor(X), torch.LongTensor(y)), batch_size=batch_size, shuffle=shuffle, pin_memory=torch.cuda.is_available())

def _apply_swa_if_better(model, swa_helper, loader):
    if swa_helper is None or swa_helper.swa_state is None or swa_helper.n == 0:
        return
    base_state = copy.deepcopy(model.state_dict())
    base_score = _acc(model, loader)
    swa_helper.apply()
    swa_score = _acc(model, loader)
    if swa_score < base_score:
        model.load_state_dict(base_state)

def _aligned_sgdr(optimizer, warmup=10, T_0=32, lr_max=0.0003, lr_min=1e-05):
    warmup_sched = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=lr_min / lr_max, end_factor=1.0, total_iters=warmup)
    sgdr_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=1, eta_min=lr_min)
    return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_sched, sgdr_sched], milestones=[warmup])

def _pct_rank(scores, ref):
    return np.searchsorted(np.sort(ref), scores, side='right') / len(ref) * 100.0

def _max_logit(model, X):
    return model.get_logits(X).max(axis=1)

def _energy(model, X, T=1.0):
    L = model.get_logits(X)
    return -T * np.log(np.sum(np.exp(L / T - L.max(1, keepdims=True)), axis=1)) - L.max(axis=1)

def _fit_gaussians(model, X, y, known_classes):
    embs = model.get_embeddings(X)
    D = embs.shape[1]
    means, cov_pool, total_n = ({}, np.zeros((D, D)), 0)
    for c in known_classes:
        pts = embs[y == c]
        if len(pts) < 2:
            continue
        mu = pts.mean(0)
        means[c] = mu
        d = pts - mu
        cov_pool += d.T @ d
        total_n += len(pts)
    cov_pool = cov_pool / max(total_n - len(means), 1) + np.eye(D) * 0.0001
    try:
        cov_inv = np.linalg.inv(cov_pool)
    except:
        cov_inv = np.eye(D)
    min_maha = np.full(len(embs), np.inf)
    for mu in means.values():
        d = embs - mu
        m = np.sqrt(np.maximum(0, np.einsum('ni,ij,nj->n', d, cov_inv, d)))
        min_maha = np.minimum(min_maha, m)
    return (means, cov_inv, min_maha)

def _maha(model, X, means, cov_inv):
    embs = model.get_embeddings(X)
    sc = np.full(len(embs), np.inf)
    for mu in means.values():
        d = embs - mu
        sc = np.minimum(sc, np.sqrt(np.maximum(0, np.einsum('ni,ij,nj->n', d, cov_inv, d))))
    return sc

def _maha_matrix(model, X, means, cov_inv, known_classes):
    embs = model.get_embeddings(X)
    out = np.full((len(embs), max(known_classes) + 1), np.inf, dtype=np.float32)
    for c in known_classes:
        if c not in means:
            continue
        d = embs - means[c]
        out[:, c] = np.sqrt(np.maximum(0, np.einsum('ni,ij,nj->n', d, cov_inv, d)))
    return out

def _top2_gap_from_logits(logits):
    part = np.partition(logits, -2, axis=1)
    return part[:, -1] - part[:, -2]

def _build_ood_component_matrix(logits, mh, ref_ml, ref_en, ref_mh, ref_gap, ood_scores=None, ref_ood=None):
    ml = logits.max(axis=1)
    en = -1.0 * np.log(np.sum(np.exp(logits - logits.max(1, keepdims=True)), axis=1)) - logits.max(axis=1)
    gap = _top2_gap_from_logits(logits)
    parts = [100.0 - _pct_rank(ml, ref_ml), _pct_rank(en, ref_en), _pct_rank(mh, ref_mh), 100.0 - _pct_rank(gap, ref_gap)]
    if ood_scores is not None and ref_ood is not None:
        parts.append(_pct_rank(ood_scores, ref_ood))
    comp = np.column_stack(parts)
    return (comp, ml, en, gap)

def _build_class_gate_thresholds(logits, y_true, maha_matrix, gaps, known_classes, maha_pct=97.0, gap_pct=4.0):
    max_cls = max(known_classes) + 1
    class_thr = np.full(max_cls, np.inf, dtype=np.float32)
    gap_thr = np.full(max_cls, -np.inf, dtype=np.float32)
    pred = np.argmax(logits, axis=1)
    for c in known_classes:
        idx = (y_true == c) & (pred == c)
        if idx.sum() < 8:
            idx = y_true == c
        if idx.sum() == 0:
            continue
        class_thr[c] = float(np.percentile(maha_matrix[idx, c], maha_pct))
        gap_thr[c] = float(np.percentile(gaps[idx], gap_pct))
    return (class_thr, gap_thr)

def _apply_class_gates(logits, maha_matrix, class_thr, gap_thr, known_classes):
    pred = np.argmax(logits, axis=1)
    pred_idx = np.clip(pred, 0, max(known_classes))
    pred_mh = maha_matrix[np.arange(len(logits)), pred_idx]
    gaps = _top2_gap_from_logits(logits)
    class_gate = pred_mh > class_thr[pred_idx]
    gap_gate = gaps < gap_thr[pred_idx]
    return (class_gate, gap_gate, pred_idx, gaps)

def ood_infer(model, X_tr_bal, y_tr_bal, X_tr_raw, y_tr_raw, X_te, known_classes, target_known_fpr=0.28):
    means, cov_inv, _ = _fit_gaussians(model, X_tr_bal, y_tr_bal, known_classes)
    tr_logits = model.get_logits(X_tr_raw)
    tr_ml = tr_logits.max(axis=1)
    tr_en = -1.0 * np.log(np.sum(np.exp(tr_logits - tr_logits.max(1, keepdims=True)), axis=1)) - tr_logits.max(axis=1)
    tr_mh = _maha(model, X_tr_raw, means, cov_inv)
    tr_gap = _top2_gap_from_logits(tr_logits)
    comp_known = np.column_stack([100.0 - _pct_rank(tr_ml, tr_ml), _pct_rank(tr_en, tr_en), _pct_rank(tr_mh, tr_mh), 100.0 - _pct_rank(tr_gap, tr_gap)])
    X_proxy = synthesize_boundary_outliers_np(X_tr_raw, y_tr_raw, n_samples=min(6000, max(2000, len(X_tr_raw) // 2)), seed=42)
    if len(X_proxy) > 0:
        pr_logits = model.get_logits(X_proxy)
        pr_ml = pr_logits.max(axis=1)
        pr_en = -1.0 * np.log(np.sum(np.exp(pr_logits - pr_logits.max(1, keepdims=True)), axis=1)) - pr_logits.max(axis=1)
        pr_mh = _maha(model, X_proxy, means, cov_inv)
        pr_gap = _top2_gap_from_logits(pr_logits)
        comp_proxy = np.column_stack([100.0 - _pct_rank(pr_ml, tr_ml), _pct_rank(pr_en, tr_en), _pct_rank(pr_mh, tr_mh), 100.0 - _pct_rank(pr_gap, tr_gap)])
        weight_bank = [np.array([0.25, 0.25, 0.25, 0.25]), np.array([0.15, 0.15, 0.4, 0.3]), np.array([0.1, 0.15, 0.45, 0.3]), np.array([0.2, 0.1, 0.45, 0.25]), np.array([0.2, 0.2, 0.35, 0.25]), np.array([0.1, 0.1, 0.55, 0.25])]
        best = None
        for w in weight_bank:
            c_known = comp_known @ w
            thr_w = float(np.percentile(c_known, (1 - target_known_fpr) * 100))
            proxy_tpr = float((comp_proxy @ w > thr_w).mean())
            sep = float((comp_proxy @ w).mean() - c_known.mean())
            score = proxy_tpr + 0.01 * sep
            if best is None or score > best[0]:
                best = (score, w, thr_w, proxy_tpr)
        weights = best[1]
        thr = best[2]
    else:
        weights = np.array([0.2, 0.15, 0.4, 0.25], dtype=np.float64)
        c_known = comp_known @ weights
        thr = float(np.percentile(c_known, (1 - target_known_fpr) * 100))
    c_raw = comp_known @ weights
    te_logits = model.get_logits(X_te)
    te_ml = te_logits.max(axis=1)
    te_en = -1.0 * np.log(np.sum(np.exp(te_logits - te_logits.max(1, keepdims=True)), axis=1)) - te_logits.max(axis=1)
    te_mh = _maha(model, X_te, means, cov_inv)
    te_gap = _top2_gap_from_logits(te_logits)
    c_te = np.column_stack([100.0 - _pct_rank(te_ml, tr_ml), _pct_rank(te_en, tr_en), _pct_rank(te_mh, tr_mh), 100.0 - _pct_rank(te_gap, tr_gap)]) @ weights
    tr_mh_cls = _maha_matrix(model, X_tr_raw, means, cov_inv, known_classes)
    te_mh_cls = _maha_matrix(model, X_te, means, cov_inv, known_classes)
    class_thr = np.full(max(known_classes) + 1, np.inf, dtype=np.float32)
    for c in known_classes:
        cls_idx = (y_tr_raw == c) & (tr_logits.argmax(axis=1) == c)
        d_c = tr_mh_cls[cls_idx, c]
        if len(d_c) < 10:
            d_c = tr_mh_cls[y_tr_raw == c, c]
        if len(d_c) > 0:
            class_thr[c] = float(np.percentile(d_c, 90.0))
    te_pred = np.argmax(te_logits, axis=1)
    pred_idx = np.clip(te_pred, 0, max(known_classes))
    te_pred_mh = te_mh_cls[np.arange(len(X_te)), pred_idx]
    class_gate = te_pred_mh > class_thr[pred_idx]
    gap_thr = np.full(max(known_classes) + 1, -np.inf, dtype=np.float32)
    tr_pred = np.argmax(tr_logits, axis=1)
    for c in known_classes:
        g_c = tr_gap[(y_tr_raw == c) & (tr_pred == c)]
        if len(g_c) < 10:
            g_c = tr_gap[y_tr_raw == c]
        if len(g_c) > 0:
            gap_thr[c] = float(np.percentile(g_c, 10.0))
    gap_gate = te_gap < gap_thr[pred_idx]
    ood_flag = (c_te > thr) | class_gate | gap_gate
    return (ood_flag, c_te, te_ml, te_en, te_mh, tr_ml, tr_en, tr_mh)

def ood_infer_v2(model, X_tr_bal, y_tr_bal, X_ref_known, y_ref_known, X_cal_known, y_cal_known, X_te, known_classes, target_known_fpr=0.28):
    means, cov_inv, _ = _fit_gaussians(model, X_tr_bal, y_tr_bal, known_classes)
    ref_logits = model.get_logits(X_ref_known)
    ref_ml = ref_logits.max(axis=1)
    ref_en = -1.0 * np.log(np.sum(np.exp(ref_logits - ref_logits.max(1, keepdims=True)), axis=1)) - ref_logits.max(axis=1)
    ref_mh = _maha(model, X_ref_known, means, cov_inv)
    ref_gap = _top2_gap_from_logits(ref_logits)
    ref_ood = model.get_ood_scores(X_ref_known)
    cal_logits = model.get_logits(X_cal_known)
    cal_mh = _maha(model, X_cal_known, means, cov_inv)
    cal_ood = model.get_ood_scores(X_cal_known)
    comp_known, _, _, cal_gap = _build_ood_component_matrix(cal_logits, cal_mh, ref_ml, ref_en, ref_mh, ref_gap, cal_ood, ref_ood)
    cal_mh_cls = _maha_matrix(model, X_cal_known, means, cov_inv, known_classes)
    X_proxy = synthesize_boundary_outliers_np(X_ref_known, y_ref_known, n_samples=min(5000, max(2000, len(X_ref_known) // 2)), seed=42)
    weight_bank = [np.array([0.18, 0.1, 0.32, 0.15, 0.25]), np.array([0.15, 0.1, 0.3, 0.15, 0.3]), np.array([0.12, 0.08, 0.3, 0.1, 0.4]), np.array([0.1, 0.1, 0.28, 0.12, 0.4]), np.array([0.15, 0.08, 0.25, 0.12, 0.4]), np.array([0.1, 0.05, 0.35, 0.1, 0.4])]
    best = None
    if len(X_proxy) > 0:
        proxy_logits = model.get_logits(X_proxy)
        proxy_mh = _maha(model, X_proxy, means, cov_inv)
        proxy_ood = model.get_ood_scores(X_proxy)
        comp_proxy, _, _, _ = _build_ood_component_matrix(proxy_logits, proxy_mh, ref_ml, ref_en, ref_mh, ref_gap, proxy_ood, ref_ood)
        proxy_mh_cls = _maha_matrix(model, X_proxy, means, cov_inv, known_classes)
        fusion_modes = ['comp_only', 'comp_or_joint', 'comp_or_class', 'comp_or_gap', 'comp_or_either']
        for w in weight_bank:
            known_score = comp_known @ w
            proxy_score = comp_proxy @ w
            for comp_pct in [72.0, 74.0, 76.0, 78.0, 80.0, 82.0, 84.0, 86.0]:
                thr = float(np.percentile(known_score, comp_pct))
                known_comp_flag = known_score > thr
                proxy_comp_flag = proxy_score > thr
                for maha_pct in [92.0, 94.0, 95.0, 96.0, 97.0, 98.0]:
                    for gap_pct in [4.0, 6.0, 8.0, 10.0, 12.0]:
                        class_thr, gap_thr = _build_class_gate_thresholds(cal_logits, y_cal_known, cal_mh_cls, cal_gap, known_classes, maha_pct=maha_pct, gap_pct=gap_pct)
                        known_class, known_gap, _, _ = _apply_class_gates(cal_logits, cal_mh_cls, class_thr, gap_thr, known_classes)
                        proxy_class, proxy_gap, _, _ = _apply_class_gates(proxy_logits, proxy_mh_cls, class_thr, gap_thr, known_classes)
                        for fusion_mode in fusion_modes:
                            if fusion_mode == 'comp_only':
                                final_known = known_comp_flag
                                final_proxy = proxy_comp_flag
                            elif fusion_mode == 'comp_or_joint':
                                final_known = known_comp_flag | known_class & known_gap
                                final_proxy = proxy_comp_flag | proxy_class & proxy_gap
                            elif fusion_mode == 'comp_or_class':
                                final_known = known_comp_flag | known_class
                                final_proxy = proxy_comp_flag | proxy_class
                            elif fusion_mode == 'comp_or_gap':
                                final_known = known_comp_flag | known_gap
                                final_proxy = proxy_comp_flag | proxy_gap
                            else:
                                final_known = known_comp_flag | known_class | known_gap
                                final_proxy = proxy_comp_flag | proxy_class | proxy_gap
                            known_fpr = float(final_known.mean())
                            proxy_tpr = float(final_proxy.mean())
                            if known_fpr <= target_known_fpr:
                                score = 1.9 * proxy_tpr + 0.08 * known_fpr
                            elif known_fpr <= target_known_fpr + 0.06:
                                score = 1.6 * proxy_tpr - 0.35 * (known_fpr - target_known_fpr)
                            else:
                                score = 1.2 * proxy_tpr - 1.35 * (known_fpr - target_known_fpr)
                            cand = (score, w, thr, class_thr, gap_thr, proxy_tpr, known_fpr, comp_pct, maha_pct, gap_pct, fusion_mode)
                            if best is None or cand[0] > best[0]:
                                best = cand
        weights = best[1]
        thr = best[2]
        class_thr = best[3]
        gap_thr = best[4]
    else:
        weights = np.array([0.2, 0.15, 0.4, 0.25], dtype=np.float64)
        thr = float(np.percentile(comp_known @ weights, 88.0))
        class_thr, gap_thr = _build_class_gate_thresholds(cal_logits, y_cal_known, cal_mh_cls, cal_gap, known_classes, maha_pct=98.0, gap_pct=2.0)
        best = (0.0, weights, thr, class_thr, gap_thr, 0.0, 0.0, 88.0, 98.0, 2.0, 'comp_or_joint')
    c_raw = comp_known @ weights
    te_logits = model.get_logits(X_te)
    te_mh = _maha(model, X_te, means, cov_inv)
    te_ood = model.get_ood_scores(X_te)
    c_te_comp, te_ml, te_en, _ = _build_ood_component_matrix(te_logits, te_mh, ref_ml, ref_en, ref_mh, ref_gap, te_ood, ref_ood)
    c_te = c_te_comp @ weights
    te_mh_cls = _maha_matrix(model, X_te, means, cov_inv, known_classes)
    class_gate, gap_gate, _, _ = _apply_class_gates(te_logits, te_mh_cls, class_thr, gap_thr, known_classes)
    comp_flag = c_te > thr
    if best[10] == 'comp_only':
        ood_flag = comp_flag
    elif best[10] == 'comp_or_joint':
        ood_flag = comp_flag | class_gate & gap_gate
    elif best[10] == 'comp_or_class':
        ood_flag = comp_flag | class_gate
    elif best[10] == 'comp_or_gap':
        ood_flag = comp_flag | gap_gate
    else:
        ood_flag = comp_flag | class_gate | gap_gate
    return (ood_flag, c_te, te_ml, te_en, te_mh, ref_ml, ref_en, ref_mh)

def create_improved_cnn_lstm(input_shape, num_classes):
    model = tf.keras.Sequential([Conv1D(16, 3, activation='sigmoid', input_shape=input_shape), MaxPooling1D(2), LSTM(32, return_sequences=True), tf.keras.layers.Dropout(0.7), Conv1D(32, 3, activation='sigmoid'), GlobalAveragePooling1D(), tf.keras.layers.Dropout(0.8), Dense(num_classes, activation='softmax')])
    model.compile(optimizer=Adam(5e-05), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

class EpochMetricsCallback(tf.keras.callbacks.Callback):

    def __init__(self, Xtr, ytr, Xte, yte, open_eval_fn=None):
        super().__init__()
        self.Xtr = Xtr
        self.ytr = np.argmax(ytr, axis=1) if ytr.ndim > 1 else ytr.astype(np.int64)
        self.Xte = Xte
        self.yte = np.argmax(yte, axis=1) if yte.ndim > 1 else yte.astype(np.int64)
        self.open_eval_fn = open_eval_fn
        self.train_history = _empty_metric_history()
        self.val_history = _empty_metric_history()
        self.open_history = _empty_metric_history()

    def on_epoch_end(self, epoch, logs=None):
        tr_pred = np.argmax(self.model.predict(self.Xtr, verbose=0), axis=1)
        te_pred = np.argmax(self.model.predict(self.Xte, verbose=0), axis=1)
        _append_metric_history(self.train_history, evaluate_predictions(self.ytr, tr_pred))
        _append_metric_history(self.val_history, evaluate_predictions(self.yte, te_pred))
        if self.open_eval_fn is not None:
            _append_metric_history(self.open_history, self.open_eval_fn(self.model))

def improved_train_model(model, Xtr, ytr, Xte, yte, epochs=100, open_eval_fn=None):
    metrics_cb = EpochMetricsCallback(Xtr, ytr, Xte, yte, open_eval_fn=open_eval_fn)
    history = model.fit(Xtr, ytr, epochs=epochs, batch_size=64, validation_data=(Xte, yte), callbacks=[ReduceLROnPlateau(factor=0.5, patience=5), metrics_cb], verbose=0)
    history.train_epoch_metrics = metrics_cb.train_history
    history.val_epoch_metrics = metrics_cb.val_history
    history.open_epoch_metrics = metrics_cb.open_history
    return history

def evaluate_keras_model(model, Xte, yte):
    yp = model.predict(Xte, verbose=0)
    ypc = np.argmax(yp, axis=1)
    yt = np.argmax(yte, axis=1)
    cm = confusion_matrix(yt, ypc, labels=[0, 1, 2])
    TN, FP = (cm[0, 0], cm[0, 1:].sum())
    return {'accuracy': accuracy_score(yt, ypc), 'precision': precision_score(yt, ypc, average='weighted', zero_division=0), 'recall': recall_score(yt, ypc, average='weighted', zero_division=0), 'f1': f1_score(yt, ypc, average='weighted', zero_division=0), 'auc': roc_auc_score(yte, yp, multi_class='ovr'), 'fpr': FP / (FP + TN) if FP + TN > 0 else 0.0}

def evaluate_3class_predictions(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    TN, FP = (cm[0, 0], cm[0, 1:].sum())
    return {'accuracy': accuracy_score(y_true, y_pred), 'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0), 'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0), 'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0), 'fpr': FP / (FP + TN) if FP + TN > 0 else 0.0}

def evaluate_predictions(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred != 0)))
    return {'accuracy': accuracy_score(y_true, y_pred), 'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0), 'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0), 'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0), 'fpr': fp / max(fp + tn, 1)}

def _empty_metric_history():
    return {k: [] for k in ['accuracy', 'precision', 'recall', 'f1', 'fpr']}

def _append_metric_history(history, metrics):
    for key in history:
        history[key].append(metrics.get(key, np.nan))

def _concat_metric_histories(*histories):
    merged = _empty_metric_history()
    for hist in histories:
        if hist is None:
            continue
        for key in merged:
            merged[key].extend(hist.get(key, []))
    return merged

def _normalize_metric_history(history):
    if isinstance(history, dict):
        normalized = _empty_metric_history()
        for key in normalized:
            normalized[key] = list(history.get(key, []))
        return normalized
    acc = list(history)
    normalized = {k: [np.nan] * len(acc) for k in _empty_metric_history()}
    normalized['accuracy'] = acc
    return normalized

def save_metrics_csv(filename, rows):
    df = pd.DataFrame(rows)
    preferred = ['Experiment', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'FPR', 'AUC', 'Unknown_Detected', 'Unknown_Total', 'Unknown_Rate']
    ordered_cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[ordered_cols]
    df.to_csv(filename, index=False)
    return df

def cnn_binary_open_set_predict(model, X_train_known, X_test, target_known_fpr=0.1):
    train_prob = model.predict(X_train_known, verbose=0)
    test_prob = model.predict(X_test, verbose=0)
    thr = float(np.percentile(train_prob.max(axis=1), target_known_fpr * 100))
    test_conf = test_prob.max(axis=1)
    test_bin = np.argmax(test_prob, axis=1)
    preds_3c = np.ones_like(test_bin, dtype=np.int64)
    preds_3c[test_bin == 0] = 0
    preds_3c[test_conf < thr] = 2
    return (preds_3c, thr)

def save_csv(filename, ours_train, ours_test, cnn_train, cnn_test):
    n = min(len(ours_train), len(ours_test), len(cnn_train), len(cnn_test))
    df = pd.DataFrame({'Epochs': list(range(1, n + 1)), 'Ours Model_Train': ours_train[:n], 'Ours Model_Test': ours_test[:n], 'CNN_LSTM_Train': cnn_train[:n], 'CNN_LSTM_Test': cnn_test[:n]})
    df.to_csv(filename, index=False)
    return df

def save_csv(filename, ours_train, ours_test, cnn_train, cnn_test):
    ours_train = _normalize_metric_history(ours_train)
    ours_test = _normalize_metric_history(ours_test)
    cnn_train = _normalize_metric_history(cnn_train)
    cnn_test = _normalize_metric_history(cnn_test)
    n = min(len(ours_train['accuracy']), len(ours_test['accuracy']), len(cnn_train['accuracy']), len(cnn_test['accuracy']))
    df = pd.DataFrame({'Epochs': list(range(1, n + 1)), 'Ours Model_Train': ours_train['accuracy'][:n], 'Ours Model_Test': ours_test['accuracy'][:n], 'CNN_LSTM_Train': cnn_train['accuracy'][:n], 'CNN_LSTM_Test': cnn_test['accuracy'][:n], 'Ours Model_Train_Precision': ours_train['precision'][:n], 'Ours Model_Test_Precision': ours_test['precision'][:n], 'CNN_LSTM_Train_Precision': cnn_train['precision'][:n], 'CNN_LSTM_Test_Precision': cnn_test['precision'][:n], 'Ours Model_Train_Recall': ours_train['recall'][:n], 'Ours Model_Test_Recall': ours_test['recall'][:n], 'CNN_LSTM_Train_Recall': cnn_train['recall'][:n], 'CNN_LSTM_Test_Recall': cnn_test['recall'][:n], 'Ours Model_Train_F1': ours_train['f1'][:n], 'Ours Model_Test_F1': ours_test['f1'][:n], 'CNN_LSTM_Train_F1': cnn_train['f1'][:n], 'CNN_LSTM_Test_F1': cnn_test['f1'][:n], 'Ours Model_Train_FPR': ours_train['fpr'][:n], 'Ours Model_Test_FPR': ours_test['fpr'][:n], 'CNN_LSTM_Train_FPR': cnn_train['fpr'][:n], 'CNN_LSTM_Test_FPR': cnn_test['fpr'][:n]})
    df.to_csv(filename, index=False)
    return df
if __name__ == '__main__':
    set_seed(42)
    EPOCHS = 100
    EPOCHS_P1 = 50
    EPOCHS_P2 = 40
    EPOCHS_P3 = 15
    train_data_path = r'F:'
    test_data_path = r'F:'
    raw_train, clean_stats = clean_data(load_data(train_data_path))
    raw_test, _ = clean_data(load_data(test_data_path), stats=clean_stats)
    tr_lab = label_fine_grained(raw_train)
    te_lab = label_fine_grained(raw_test)
    feat_cols = [c for c in raw_train.columns if c != 'label']
    X_tr_raw, X_te_raw, _ = automated_feature_engineering_train_test(tr_lab[feat_cols], te_lab[feat_cols])
    X_tr_raw = np.nan_to_num(X_tr_raw)
    X_te_raw = np.nan_to_num(X_te_raw)
    X_tr, X_te, _ = scale_features(X_tr_raw, X_te_raw)
    INPUT_DIM = X_tr.shape[1]
    y_tr_fg = tr_lab['label_fg'].values
    y_tr_3c = tr_lab['label_3c'].values
    y_te_fg = te_lab['label_fg'].values
    y_te_3c = te_lab['label_3c'].values
    Xtr_a1 = np.expand_dims(X_tr, 2)
    Xte_a1 = np.expand_dims(X_te, 2)
    ytr_a1 = tf.keras.utils.to_categorical(y_tr_3c, 3)
    yte_a1 = tf.keras.utils.to_categorical(y_te_3c, 3)
    cnn_std = create_improved_cnn_lstm((Xtr_a1.shape[1], 1), 3)
    ha1 = improved_train_model(cnn_std, Xtr_a1, ytr_a1, Xte_a1, yte_a1, EPOCHS)
    ha1.train_epoch_metrics['accuracy'] = list(ha1.history.get('accuracy', ha1.train_epoch_metrics['accuracy']))
    ha1.val_epoch_metrics['accuracy'] = list(ha1.history.get('val_accuracy', ha1.val_epoch_metrics['accuracy']))
    cnn_train_std = ha1.train_epoch_metrics
    cnn_test_std = ha1.val_epoch_metrics
    cw_std = compute_class_weights(y_tr_fg, N_STD_CLASSES, benign_boost=4.0)
    tr_ld_std = _make_loader(X_tr, y_tr_fg, batch_size=8, shuffle=True)
    te_ld_std = _make_loader(X_te, y_te_fg, batch_size=256)
    X_std_proxy = synthesize_boundary_outliers_np(X_tr, y_tr_fg, n_samples=min(5000, max(2000, len(X_tr) // 3)), seed=7)
    std_proxy_ld = _make_loader(X_std_proxy, np.zeros(len(X_std_proxy), dtype=np.int64), batch_size=256)
    ours_std = EnhancedTransformer(INPUT_DIM, N_STD_CLASSES, dropout1=0.0, dropout2=0.0)
    opt_p1 = torch.optim.AdamW(ours_std.parameters(), lr=0.0008, weight_decay=1e-07)
    sch_p1 = _aligned_sgdr(opt_p1, warmup=10, T_0=32)
    tr_h1, te_h1 = EnhancedTransformer.train_model_mixup(ours_std, tr_ld_std, te_ld_std, cw_std, opt_p1, sch_p1, EPOCHS_P1, N_STD_CLASSES, noise_std=0.001, mixup_alpha=0.03, focal_gamma=0.8, label_smoothing=0.0, center_loss_weight=0.0, boundary_ood_weight=0.01, proxy_ood_loader=std_proxy_ld, proxy_select_weight=0.02, label='Std-P1')
    opt_p2 = torch.optim.AdamW(ours_std.parameters(), lr=0.0003, weight_decay=1e-07)
    sch_p2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_p2, T_max=EPOCHS_P2, eta_min=1e-06)
    swa_std = SWAHelper(ours_std)
    tr_h2, te_h2 = EnhancedTransformer.train_model_mixup(ours_std, tr_ld_std, te_ld_std, cw_std, opt_p2, sch_p2, EPOCHS_P2, N_STD_CLASSES, noise_std=0.0, mixup_alpha=0.0, swa=swa_std, swa_start=20, focal_gamma=0.0, label_smoothing=0.0, center_loss_weight=0.01, boundary_ood_weight=0.003, margin_loss_weight=0.025, proxy_ood_loader=std_proxy_ld, proxy_select_weight=0.03, label='Std-P2')
    _apply_swa_if_better(ours_std, swa_std, te_ld_std)
    opt_p3 = torch.optim.AdamW(ours_std.parameters(), lr=0.0001, weight_decay=1e-08)
    sch_p3 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_p3, T_max=EPOCHS_P3, eta_min=1e-06)
    tr_h3, te_h3 = EnhancedTransformer.train_model_mixup(ours_std, tr_ld_std, te_ld_std, cw_std, opt_p3, sch_p3, EPOCHS_P3, N_STD_CLASSES, noise_std=0.0, mixup_alpha=0.0, focal_gamma=0.0, label_smoothing=0.0, center_loss_weight=0.02, boundary_ood_weight=0.002, margin_loss_weight=0.04, proxy_ood_loader=std_proxy_ld, proxy_select_weight=0.04, label='Std-P3')
    ours_train_std = _concat_metric_histories(tr_h1, tr_h2, tr_h3)
    ours_test_std = _concat_metric_histories(te_h1, te_h2, te_h3)
    preds_fg = np.argmax(ours_std.get_logits(X_te), axis=1)
    preds_3c = fg_to_3class(preds_fg)
    cm_a = confusion_matrix(y_te_3c, preds_3c, labels=[0, 1, 2])
    TN_a, FP_a = (cm_a[0, 0], cm_a[0, 1:].sum())
    acc_a = accuracy_score(y_te_3c, preds_3c)
    prec_a = precision_score(y_te_3c, preds_3c, average='weighted', zero_division=0)
    rec_a = recall_score(y_te_3c, preds_3c, average='weighted', zero_division=0)
    f1_a = f1_score(y_te_3c, preds_3c, average='weighted', zero_division=0)
    fpr_a = FP_a / (FP_a + TN_a) if FP_a + TN_a > 0 else 0.0
    try:
        probs_fg = torch.softmax(torch.tensor(ours_std.get_logits(X_te)), dim=1).numpy()
        probs_3c = np.zeros((len(probs_fg), 3), dtype=np.float32)
        probs_3c[:, 0] = probs_fg[:, 0]
        probs_3c[:, 1] = probs_fg[:, 1:N_KNOWN_TOTAL].sum(axis=1)
        probs_3c[:, 2] = probs_fg[:, UNKNOWN_LABEL]
        auc_a = roc_auc_score(tf.keras.utils.to_categorical(y_te_3c, 3), probs_3c, multi_class='ovr')
    except:
        auc_a = float('nan')
    save_csv('experiment_A_standard_accuracy.csv', ours_train_std, ours_test_std, cnn_train_std, cnn_test_std)
    known_mask_tr = np.isin(y_tr_fg, list(range(N_KNOWN_TOTAL)))
    X_tr_tf_all = X_tr[known_mask_tr]
    y_tr_tf_all = y_tr_fg[known_mask_tr]
    X_tr_tf_raw, X_val_tf_known, y_tr_tf_raw, y_val_tf_known = train_test_split(X_tr_tf_all, y_tr_tf_all, test_size=0.1, random_state=42, stratify=y_tr_tf_all)
    unknown_mask_te = y_te_3c == 2
    known_mask_te = ~unknown_mask_te
    n_unknown_te = int(unknown_mask_te.sum())
    for c in range(N_KNOWN_TOTAL):
        nm = 'BENIGN' if c == 0 else KNOWN_ATTACKS[c - 1]
    Xtr_b1 = np.expand_dims(X_tr_tf_raw, 2)
    Xte_b1 = np.expand_dims(X_te, 2)
    Xte_b1_known = np.expand_dims(X_val_tf_known, 2)
    ytr_b1 = tf.keras.utils.to_categorical(np.clip(y_tr_tf_raw, 0, 1), 2)
    yte_b1_known = tf.keras.utils.to_categorical(np.clip(y_val_tf_known, 0, 1), 2)

    def _cnn_tf_open_eval(cur_model):
        preds_3c, _ = cnn_binary_open_set_predict(cur_model, Xtr_b1, Xte_b1, target_known_fpr=0.1)
        return evaluate_3class_predictions(y_te_3c, preds_3c)
    cnn_tf = create_improved_cnn_lstm((Xtr_b1.shape[1], 1), 2)
    hb1 = improved_train_model(cnn_tf, Xtr_b1, ytr_b1, Xte_b1_known, yte_b1_known, EPOCHS, open_eval_fn=_cnn_tf_open_eval)
    hb1.train_epoch_metrics['accuracy'] = list(hb1.history.get('accuracy', hb1.train_epoch_metrics['accuracy']))
    hb1.val_epoch_metrics['accuracy'] = list(hb1.history.get('val_accuracy', hb1.val_epoch_metrics['accuracy']))
    cnn_train_tf = hb1.train_epoch_metrics
    cnn_test_tf = hb1.open_epoch_metrics if len(hb1.open_epoch_metrics['accuracy']) > 0 else hb1.val_epoch_metrics
    yp_b1_3c, _ = cnn_binary_open_set_predict(cnn_tf, Xtr_b1, Xte_b1, target_known_fpr=0.1)
    cnn_det = int(np.sum(yp_b1_3c[unknown_mask_te] == 2))
    cw_tf = compute_class_weights(y_tr_tf_raw, N_KNOWN_TOTAL, benign_boost=1.5)
    tr_ld_tf = _make_loader(X_tr_tf_raw, y_tr_tf_raw, batch_size=8, shuffle=True)
    X_te_known = X_te[known_mask_te]
    y_te_known_fg = np.clip(y_te_fg[known_mask_te], 0, N_KNOWN)
    te_ld_known = _make_loader(X_val_tf_known, y_val_tf_known, batch_size=256)
    X_tf_proxy = synthesize_boundary_outliers_np(X_tr_tf_raw, y_tr_tf_raw, n_samples=min(5000, max(2000, len(X_tr_tf_raw) // 2)), seed=11)
    tf_proxy_ld = _make_loader(X_tf_proxy, np.zeros(len(X_tf_proxy), dtype=np.int64), batch_size=256)
    counts_tf = [(y_tr_tf_raw == c).sum() for c in range(N_KNOWN_TOTAL)]
    min_n_tf = min((n for n in counts_tf if n > 0))
    target_tf = min(min_n_tf * 3, max(counts_tf))
    X_b, y_b = ([], [])
    for c in range(N_KNOWN_TOTAL):
        idx = y_tr_tf_raw == c
        if idx.sum() == 0:
            continue
        Xc, yc = resample(X_tr_tf_raw[idx], y_tr_tf_raw[idx], replace=idx.sum() < target_tf, n_samples=target_tf, random_state=42)
        X_b.append(Xc)
        y_b.append(yc)
    X_tr_tf_bal = np.vstack(X_b)
    y_tr_tf_bal = np.concatenate(y_b)
    ours_tf = EnhancedTransformer(INPUT_DIM, N_KNOWN_TOTAL, dropout1=0.0, dropout2=0.0)
    opt_tf_p1 = torch.optim.AdamW(ours_tf.parameters(), lr=0.0008, weight_decay=1e-07)
    sch_tf_p1 = _aligned_sgdr(opt_tf_p1, warmup=10, T_0=32)
    tf_h1, tes1 = EnhancedTransformer.train_model_mixup(ours_tf, tr_ld_tf, te_ld_known, cw_tf, opt_tf_p1, sch_tf_p1, EPOCHS_P1, N_KNOWN_TOTAL, noise_std=0.001, mixup_alpha=0.03, focal_gamma=0.8, label_smoothing=0.0, center_loss_weight=0.0, boundary_ood_weight=0.03, proxy_ood_loader=tf_proxy_ld, proxy_select_weight=0.04, label='Tf-P1')
    opt_tf_p2 = torch.optim.AdamW(ours_tf.parameters(), lr=0.0003, weight_decay=1e-07)
    sch_tf_p2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_tf_p2, T_max=EPOCHS_P2, eta_min=1e-06)
    swa_tf = SWAHelper(ours_tf)
    tf_h2, tes2 = EnhancedTransformer.train_model_mixup(ours_tf, tr_ld_tf, te_ld_known, cw_tf, opt_tf_p2, sch_tf_p2, EPOCHS_P2, N_KNOWN_TOTAL, noise_std=0.0, mixup_alpha=0.0, swa=swa_tf, swa_start=20, focal_gamma=0.0, label_smoothing=0.0, center_loss_weight=0.01, boundary_ood_weight=0.012, margin_loss_weight=0.04, proxy_ood_loader=tf_proxy_ld, proxy_select_weight=0.07, label='Tf-P2')
    _apply_swa_if_better(ours_tf, swa_tf, te_ld_known)
    opt_tf_p3 = torch.optim.AdamW(ours_tf.parameters(), lr=0.0001, weight_decay=1e-08)
    sch_tf_p3 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_tf_p3, T_max=EPOCHS_P3, eta_min=1e-06)
    tf_h3, tes3 = EnhancedTransformer.train_model_mixup(ours_tf, tr_ld_tf, te_ld_known, cw_tf, opt_tf_p3, sch_tf_p3, EPOCHS_P3, N_KNOWN_TOTAL, noise_std=0.0, mixup_alpha=0.0, focal_gamma=0.0, label_smoothing=0.0, center_loss_weight=0.02, boundary_ood_weight=0.015, margin_loss_weight=0.06, proxy_ood_loader=tf_proxy_ld, proxy_select_weight=0.1, label='Tf-P3')
    ours_train_tf = _concat_metric_histories(tf_h1, tf_h2, tf_h3)
    ours_test_tf = _concat_metric_histories(tes1, tes2, tes3)
    ood_flag, c_te, te_ml, te_en, te_mh, tr_ml, tr_en, tr_mh = ood_infer_v2(ours_tf, X_tr_tf_bal, y_tr_tf_bal, X_tr_tf_raw, y_tr_tf_raw, X_val_tf_known, y_val_tf_known, X_te, list(range(N_KNOWN_TOTAL)), target_known_fpr=0.28)
    fg_p = np.argmax(ours_tf.get_logits(X_te), axis=1)
    final_3c = fg_to_3class(fg_p)
    final_3c[ood_flag] = 2
    ours_det = int(np.sum(final_3c[unknown_mask_te] == 2))
    fp = int(np.sum(final_3c[known_mask_te] == 2))
    fpr_act = fp / max(known_mask_te.sum(), 1)
    for name, sc, hi in [('Max-logit', te_ml, False), ('Energy', te_en, True), ('Min-Maha', te_mh, True)]:
        sep = (sc[unknown_mask_te].mean() - sc[known_mask_te].mean()) / (sc.std() + 1e-08)
        if not hi:
            sep = -sep
    cs = (c_te[unknown_mask_te].mean() - c_te[known_mask_te].mean()) / (c_te.std() + 1e-08)
    save_csv('experiment_B_standard_accuracy.csv', ours_train_tf, ours_test_tf, cnn_train_tf, cnn_test_tf)
    m_a_cnn = evaluate_keras_model(cnn_std, Xte_a1, yte_a1)
    metrics_b_cnn = evaluate_3class_predictions(y_te_3c, yp_b1_3c)
    metrics_b_ours = evaluate_3class_predictions(y_te_3c, final_3c)
    acc_b = metrics_b_ours['accuracy']
    prec_b = metrics_b_ours['precision']
    rec_b = metrics_b_ours['recall']
    f1_b = metrics_b_ours['f1']
    fpr_b = metrics_b_ours['fpr']
