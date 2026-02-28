# ==========================================
# 1. Biblioteki Standardowe i Konfiguracja
# ==========================================
import os
import csv
import copy
import json
import datetime
import multiprocessing
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5pickle
from tqdm import tqdm

# ==========================================
# 2. Scikit-Learn
# ==========================================
from sklearn.metrics import (
    roc_curve, roc_auc_score, 
    accuracy_score, matthews_corrcoef,
    silhouette_score
)

# ==========================================
# 3. PyTorch & TorchData
# ==========================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Function
from torch.cuda.amp import GradScaler, autocast
import torchdata.datapipes as dp
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService

# ==========================================
# 4. Lokalne Narzędzia
# ==========================================
from loader_utils import (
    label_override, 
    load_label_override, 
    load_metadata_database, 
    stratified_sample, 
    read_folds_override, 
    get_eids_for_folds
)

# Ścieżki
data_pth = "/dmj/fizmed/mpoziomska/ELMIKO/neuroscreening-fuw/data/elmiko/processed_all_MIL_800"
model_pth = "/dmj/fizmed/jpelczar/od_martyny/minet/models/minet_raw_fold_6"
csv_path = 'used_label_database.csv'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"🔧 Using device: {device}")

# --- LISTA WSZYSTKICH SZPITALI (DLA MAPOWANIA) ---
ALLOWED_HOSPITALS = [
    "ZOZLO", "KATMOJPRZ", "SZC", "TOR", "OST", 
    "LUMICE", "CMD", "SRK", "AKS", "PRZ", 
    "KIEG", "OTW", "MKW", "PUS", "LUX_A", 
    "KUD", "ARCHDAM", "MOR", "KAL", "B2K", 
    "SLU", "SL2", "STG1", "CHE", "KLU", 
    "GAK", "WLU", "Z04O", "TER_L", "PIO"
]

# --- LISTA SZPITALI DLA TEJ MASZYNY (CZĘŚĆ 2/2) ---
MY_TARGET_LIST = [
    "KUD", "ARCHDAM", "MOR", "KAL", "B2K", 
    "SLU", "SL2", "STG1", "CHE", "KLU", 
    "GAK", "WLU", "Z04O", "TER_L", "PIO"
]

print(f"🚀 Uruchamiam pętlę (Multi-task NAIVE - od zera) dla szpitali: {MY_TARGET_LIST}")

# ==========================================
# DEFINICJE KLAS (POZA PĘTLĄ)
# ==========================================

def collate_pad(batch):
    X, y, eid = zip(*batch)
    y = torch.tensor(y)
    X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True)
    return [X, y, eid]

class Loader:
    def __init__(self, data_path, eids, label_override='classification_latest',
                 override_non_mil=False, minet_subsampling_n=None, num_workers=None):
        self._num_workers = num_workers
        self.minet_subsampling_n = minet_subsampling_n
        self._data_file = h5pickle.File(os.path.join(data_path, 'features', 'data.hdf5'), 'r')
        self.labels_database = load_label_override(label_override, data_path)
        self.metadata_database = load_metadata_database(data_path)
        manager = multiprocessing.Manager()
        self.additional_metadata = manager.dict()
        
        if override_non_mil: self._loader_type = 'none'
        else: self._loader_type = 'MIL'

        self._data_path = data_path
        self._eids = eids

    def construct_data_pipe(self, batch_size, pad):
        pipe = dp.map.SequenceWrapper(self._eids)
        pipe = pipe.shuffle()
        pipe = pipe.sharding_filter()
        pipe = pipe.map(self.loader_mapping_func)
        if self._loader_type == "none": pipe = pipe.unbatch()
        pipe = pipe.batch(batch_size=batch_size, drop_last=True)
        if (self._loader_type == 'MIL') and pad: pipe = pipe.collate(collate_pad)
        else: pipe = pipe.collate()
        return pipe

    def loader_mapping_func(self, eid):
        try:
            cls = 1 - label_override([eid], self.labels_database)[0]
            eid_key = str(eid)
            if eid_key in self._data_file['metadata'].attrs:
                meta_str = self._data_file['metadata'].attrs[eid_key]
                additional_metadata = json.loads(meta_str)
            else:
                additional_metadata = {}

            cls_torch = torch.tensor(int(cls), dtype=torch.int64)
            
            if eid_key in self._data_file['features']:
                data_h5 = np.array(self._data_file['features'][eid_key])
            else:
                return (torch.zeros(10, 19, 600), cls_torch, eid)

            if data_h5.ndim == 3 and data_h5.shape[2] == 19:
                data_h5 = data_h5.transpose(0, 2, 1)
            
            frames_n = data_h5.shape[0]
            frame_types = additional_metadata.get('events_list', ["None"] * frames_n)
            frame_timings = additional_metadata.get('event_timewindows', [[0, 0]] * frames_n)

            if self.minet_subsampling_n is not None and frames_n > self.minet_subsampling_n:
                data_h5, frame_types, frame_timings = stratified_sample(self.minet_subsampling_n, data_h5, frame_timings, frame_types)
            
            data = torch.tensor(data_h5, dtype=torch.float32)
            
            if self._loader_type == "none":
                return [(data[i], cls_torch, eid) for i in range(len(data))]
            else:
                return (data, cls_torch, eid)
        except Exception:
             return (torch.zeros(10, 19, 600), torch.tensor(0), eid)
    
    def get_batched_loader(self, batch_size, pad=True):
        pipe = self.construct_data_pipe(batch_size, pad)
        if self._num_workers is not None:
            num_workers = self._num_workers
        else:
            num_workers = min([6, multiprocessing.cpu_count() - 1])
        mp_rs = MultiProcessingReadingService(num_workers=num_workers)
        return DataLoader2(pipe, reading_service=mp_rs)

# =================================================================
# ZWYKŁA PROPAGACJA (MULTI-TASK, BEZ DANN)
# =================================================================
class NormalLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # NORMALNA PROPAGACJA - Model uczy się domeny
        return grad_output, None

class MinetMultiTask(nn.Module):
    def __init__(self, original_model, feature_dim=288, num_domains=39):
        super(MinetMultiTask, self).__init__()
        self.backbone = original_model
        
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_domains)
        )

    def forward(self, x, time_stamps=None, alpha=1.0, part=False):
        out = self.backbone(x, time_stamps=time_stamps, part=True)
        
        if isinstance(out, tuple): features = out[0]
        else: features = out

        if features.dim() == 3: 
            features = torch.mean(features, dim=1)

        if part: return features, None

        class_output = self.backbone.classifier(features)
        
        normal_features = NormalLayerF.apply(features, alpha)
        domain_output = self.domain_classifier(normal_features)

        return class_output, domain_output

# ==========================================
# RESET WAG (Do treningu NAIVE)
# ==========================================
def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        m.reset_parameters()


# ==========================================
# GŁÓWNA PĘTLA SZPITALI
# ==========================================

print("📂 Ładowanie metadanych (raz na start)...")
df_meta = pd.read_csv(csv_path, sep='|', low_memory=False)
df_meta['examination_id'] = df_meta['examination_id'].astype(str).str.strip()
df_meta['institution_id'] = df_meta['institution_id'].astype(str).str.strip()

df_meta = df_meta[df_meta['institution_id'].isin(ALLOWED_HOSPITALS)].copy()

all_hospitals_sorted = sorted(df_meta['institution_id'].unique().tolist())
hospital_to_id = {name: i for i, name in enumerate(all_hospitals_sorted)}
eid_to_hosp_name = pd.Series(df_meta.institution_id.values, index=df_meta.examination_id).to_dict()

def clean_eid_str(eid_raw):
    if isinstance(eid_raw, (list, tuple, np.ndarray)): 
        if len(eid_raw) > 0: eid_raw = eid_raw[0]
    if isinstance(eid_raw, bytes): return eid_raw.decode('utf-8')
    return str(eid_raw).strip()

def get_domain_labels(eids):
    labels = []
    flat_eids = eids if isinstance(eids, (list, tuple)) else [eids]
    for e in flat_eids:
        e_clean = clean_eid_str(e)
        h_name = eid_to_hosp_name.get(e_clean, None)
        label_id = hospital_to_id.get(h_name, 0)
        labels.append(label_id)
    return torch.tensor(labels, dtype=torch.long).to(device)

# --- START PĘTLI ITERUJĄCEJ PRZEZ SZPITALE ---
for current_target in MY_TARGET_LIST:
    
    print("\n" + "="*60)
    print(f"🏥 START PRZETWARZANIA SZPITALA: {current_target}")
    print("="*60)

    # 1. PRZYGOTOWANIE KATALOGU
    TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    EXP_ROOT = "experiments_multitask_naive" # <--- Nowa nazwa folderu dla startu od zera
    EXP_DIR = f"{EXP_ROOT}/{current_target}_{TIMESTAMP}"
    os.makedirs(EXP_DIR, exist_ok=True)

    TRAIN_LOG_FILE = os.path.join(EXP_DIR, "training_log.csv")
    FINAL_RESULTS_FILE = os.path.join(EXP_DIR, "final_results.csv")
    MODEL_SAVE_PATH = os.path.join(EXP_DIR, "best_model.pt")

    with open(TRAIN_LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_Loss', 'Train_Cls_Loss', 'Train_Dom_Loss', 'Val_Target_AUC', 'Alpha'])

    with open(FINAL_RESULTS_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric_Name', 'Value', 'Description'])

    def log_final_metric(name, value, description=""):
        with open(FINAL_RESULTS_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, value, description])

    # 2. PRZYGOTOWANIE DANYCH (LOSO)
    folds, fold_override = read_folds_override(data_pth, None)
    all_eids_raw = get_eids_for_folds(fold_override, [1, 2, 3, 4, 5, 6]) 

    source_pool = []
    target_pool = []
    valid_eids_set = set(df_meta['examination_id'].values)

    for eid in all_eids_raw:
        eid_clean = clean_eid_str(eid)
        if eid_clean in valid_eids_set:
            h_name = eid_to_hosp_name.get(eid_clean, "Unknown")
            if h_name == current_target:
                target_pool.append(eid)
            else:
                source_pool.append(eid)

    if not target_pool:
        print(f"⚠️ Pominiecie {current_target}: Brak danych.")
        continue

    random.seed(42)
    random.shuffle(target_pool)
    split_idx = int(len(target_pool) * 0.5)
    if split_idx == 0: split_idx = 1

    target_train_eids = target_pool[:split_idx]
    target_eval_eids = target_pool[split_idx:]

    print(f"   Source: {len(source_pool)} | Target Train: {len(target_train_eids)} | Target Eval: {len(target_eval_eids)}")

    source_loader = Loader(data_pth, source_pool, minet_subsampling_n=4, num_workers=4).get_batched_loader(32, pad=True)
    target_loader = Loader(data_pth, target_train_eids, minet_subsampling_n=4, num_workers=4).get_batched_loader(32, pad=True)
    eval_loader = Loader(data_pth, target_eval_eids, minet_subsampling_n=4, num_workers=2).get_batched_loader(32, pad=True)
    test_loader_full = Loader(data_pth, target_eval_eids, minet_subsampling_n=None, num_workers=2).get_batched_loader(1, pad=True)

    # 3. INICJALIZACJA MODELU (OD ZERA)
    print("\n🧠 Ładowanie modelu i RESET WAG (Naive Multi-Task)...")
    raw_backbone = torch.load(model_pth, map_location=device)
    
    # Niszczymy stare wagi!
    raw_backbone.apply(weight_reset)
    
    if hasattr(raw_backbone, 'n_chans'): raw_backbone.n_chans = 19

    multitask_model = MinetMultiTask(
        raw_backbone, 
        feature_dim=288, 
        num_domains=len(all_hospitals_sorted) 
    ).to(device)

    # Optimizer (Wszystko odblokowane, bo wagi są losowe)
    backbone_head_params = list(multitask_model.backbone.classifier.parameters())
    head_ids = list(map(id, backbone_head_params))
    backbone_base_params = filter(lambda p: id(p) not in head_ids, multitask_model.backbone.parameters())

    optimizer = optim.AdamW([
        {'params': backbone_base_params, 'lr': 1e-3}, 
        {'params': backbone_head_params, 'lr': 1e-3},
        {'params': multitask_model.domain_classifier.parameters(), 'lr': 1e-3}
    ], weight_decay=1e-4)

    loss_class_fn = nn.BCEWithLogitsLoss()
    loss_domain_fn = nn.CrossEntropyLoss()
    scaler = GradScaler()

    # 4. TRENING (50 EPOK)
    EPOCHS = 50
    STEPS = 500
    WARMUP_EPOCHS = 10 
    MAX_ALPHA = 1.0 
    
    best_auc = 0.0
    best_model_wts = copy.deepcopy(multitask_model.state_dict())

    def infinite_iterator(loader):
        while True:
            for batch in loader: yield batch
    iter_target = infinite_iterator(target_loader)

    print("🚀 Start Treningu Od Zera (50 Epok)...")
    for epoch in range(EPOCHS):
        multitask_model.train()
        iter_source = iter(source_loader)
        
        total_loss, total_cls, total_dom = 0, 0, 0
        steps = 0

        pbar = tqdm(range(STEPS), desc=f"Epoka {epoch+1}/{EPOCHS}", leave=False)
        
        for i in pbar:
            # Utrzymujemy wyliczanie alphy dla logów i wagi domeny, choć nie odwraca ona gradientu
            if epoch < WARMUP_EPOCHS:
                alpha = 0.0
            else:
                total_phase2_steps = (EPOCHS - WARMUP_EPOCHS) * STEPS
                current_phase2_step = (i + ((epoch - WARMUP_EPOCHS) * STEPS))
                ratio = current_phase2_step / total_phase2_steps
                alpha = min(ratio * MAX_ALPHA, MAX_ALPHA)
                
            try:
                batch_s = next(iter_source)
                s_X, s_y, s_eid = batch_s[0], batch_s[1], batch_s[2]
                batch_t = next(iter_target)
                t_X, t_y, t_eid = batch_t[0], batch_t[1], batch_t[2]
            except StopIteration:
                break

            s_X = s_X.to(device, dtype=torch.float32)
            s_y = s_y.to(device, dtype=torch.float32).view(-1, 1)
            t_X = t_X.to(device, dtype=torch.float32)
            
            s_ts = torch.zeros((s_X.shape[0], s_X.shape[1]), dtype=torch.long, device=device)
            t_ts = torch.zeros((t_X.shape[0], t_X.shape[1]), dtype=torch.long, device=device)

            d_label_s = get_domain_labels(s_eid)
            d_label_t = get_domain_labels(t_eid)
            
            optimizer.zero_grad()

            with autocast():
                c_pred_s, d_pred_s = multitask_model(s_X, time_stamps=s_ts, alpha=alpha)
                loss_cls = loss_class_fn(c_pred_s, s_y)
                loss_dom_s = loss_domain_fn(d_pred_s, d_label_s)

                _, d_pred_t = multitask_model(t_X, time_stamps=t_ts, alpha=alpha)
                loss_dom_t = loss_domain_fn(d_pred_t, d_label_t)

                loss_dom_total = loss_dom_s + loss_dom_t
                loss = (5.0 * loss_cls) + loss_dom_total
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer) 
            torch.nn.utils.clip_grad_norm_(multitask_model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            total_cls += loss_cls.item()
            total_dom += loss_dom_total.item()
            steps += 1
            
            pbar.set_postfix({'Cls': f"{loss_cls.item():.2f}", 'Dom': f"{loss_dom_total.item():.2f}"})

        # Walidacja
        multitask_model.eval()
        y_true, y_prob = [], []
        with torch.no_grad():
            for data in eval_loader:
                if len(data) == 3: X, y, _ = data; ts = None
                else: X, y, _, ts = data
                X = X.to(device, dtype=torch.float32)
                ts = torch.zeros((X.shape[0], X.shape[1]), dtype=torch.long, device=device) if ts is None else ts.to(device)
                
                logits, _ = multitask_model(X, time_stamps=ts, alpha=0.0)
                probs = torch.sigmoid(logits)
                y_true.extend(y.tolist())
                y_prob.extend(probs.cpu().tolist())
        
        try:
            val_auc = roc_auc_score(y_true, y_prob)
        except:
            val_auc = 0.5

        # Zapis logów
        with open(TRAIN_LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, total_loss/steps, total_cls/steps, total_dom/steps, val_auc, alpha])

        if val_auc > best_auc:
            best_auc = val_auc
            best_model_wts = copy.deepcopy(multitask_model.state_dict())
            torch.save(multitask_model.state_dict(), MODEL_SAVE_PATH)

    # 5. EWALUACJE KOŃCOWE
    def run_evaluation(model, loader, device, prefix=""):
        model.eval()
        y_true, y_prob = [], []
        with torch.no_grad():
            for data in tqdm(loader, desc=f"Eval {prefix}", leave=False):
                if len(data) == 3: X, y, eid = data; ts = None
                else: X, y, eid = data[0], data[1], data[2]; ts = None
                
                if X.shape[1] > 300: X = X[:, :300]
                X = X.to(device, dtype=torch.float32)
                ts = torch.zeros((1, X.shape[1]), dtype=torch.long, device=device)
                
                try:
                    logits, _ = model(X, time_stamps=ts, alpha=0.0)
                    prob = torch.sigmoid(logits).item()
                    y_true.append(y.item())
                    y_prob.append(prob)
                except: pass

        y_true = np.array(y_true)
        y_prob = np.array(y_prob)
        
        if len(np.unique(y_true)) > 1:
            auc_val = roc_auc_score(y_true, y_prob)
            fpr, tpr, thresh = roc_curve(y_true, y_prob)
            best_t = thresh[np.argmax(tpr - fpr)]
            y_pred = (y_prob >= best_t).astype(int)
            mcc = matthews_corrcoef(y_true, y_pred)
            acc = accuracy_score(y_true, y_pred)
            
            print(f"   👉 {prefix} Results: AUC={auc_val:.4f} | MCC={mcc:.4f} | Acc={acc:.4f}")
            log_final_metric(f"{prefix}Target_Diagnosis_AUC", auc_val, "Target Eval")
            log_final_metric(f"{prefix}Target_Diagnosis_MCC", mcc, "Target Eval")
            log_final_metric(f"{prefix}Target_Diagnosis_Acc", acc, "Target Eval")
            return auc_val
        else:
            print("   ⚠️ Brak klas w teście.")
            return 0.0

    def evaluate_metrics_comprehensive(model, loader, device, prefix=""):
        model.eval()
        feats, d_true, d_pred = [], [], []
        with torch.no_grad():
            for i, data in enumerate(loader):
                if i > 200: break
                if len(data) == 4: X, y, eid, ts = data
                else: X, y, eid = data; ts = None
                if X.shape[1] > 300: X = X[:, :300]
                X = X.to(device, dtype=torch.float32)
                ts = torch.zeros((X.shape[0], X.shape[1]), dtype=torch.long, device=device)
                
                f, _ = model.backbone(X, time_stamps=ts, part=True)
                if len(f.shape) == 3: f = torch.mean(f, dim=1)
                
                _, d_logits = model(X, time_stamps=ts, alpha=0.0)
                d_p = torch.argmax(d_logits, dim=1).cpu().numpy()
                d_t = get_domain_labels(eid).cpu().numpy()
                
                feats.append(f.cpu().numpy())
                d_true.extend(d_t)
                d_pred.extend(d_p)
                
        feats = np.concatenate(feats, axis=0)
        d_true = np.array(d_true)
        d_pred = np.array(d_pred)
        
        if len(np.unique(d_true)) > 1:
            mcc = matthews_corrcoef(d_true, d_pred)
            acc = np.mean(d_true == d_pred)
            log_final_metric(f"{prefix}Domain_MCC", mcc)
            log_final_metric(f"{prefix}Domain_Acc", acc)
            
            if len(feats) > 2000:
                idx = np.random.choice(len(feats), 2000, replace=False)
                feats = feats[idx]
                d_true = d_true[idx]
            
            sil = silhouette_score(feats, d_true)
            log_final_metric(f"{prefix}Silhouette_Domain", sil)

    # 1. Final Model
    run_evaluation(multitask_model, test_loader_full, device, prefix="Final_")
    evaluate_metrics_comprehensive(multitask_model, source_loader, device, prefix="Final_")

    # 2. Best Model
    if best_model_wts is not None:
        print(f"✅ Przywracanie najlepszych wag (AUC={best_auc:.4f})...")
        multitask_model.load_state_dict(best_model_wts)
        run_evaluation(multitask_model, test_loader_full, device, prefix="Best_")
        evaluate_metrics_comprehensive(multitask_model, source_loader, device, prefix="Best_")

    print(f"✅ Zakończono dla {current_target}")

print("\n🎉 ZAKOŃCZONO PĘTLĘ 1/2!")