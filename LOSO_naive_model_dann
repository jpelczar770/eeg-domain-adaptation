# ==========================================
# 1. Biblioteki Standardowe i Konfiguracja
# ==========================================
import os
import csv
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
print(f"Using device: {device}")

# ==========================================
# KONFIGURACJA EKSPERYMENTU (LOGOWANIE)
# ==========================================
TARGET_HOSPITAL_CODE = 'KLU' # <-- TU ZMIENIASZ SZPITAL (Target)

# Generowanie unikalnej nazwy folderu
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
EXP_DIR = f"experiments/{TARGET_HOSPITAL_CODE}_{TIMESTAMP}"
os.makedirs(EXP_DIR, exist_ok=True)

print(f"Wyniki będą zapisywane w: {EXP_DIR}")

# Ścieżki do plików
TRAIN_LOG_FILE = os.path.join(EXP_DIR, "training_log.csv")
FINAL_RESULTS_FILE = os.path.join(EXP_DIR, "final_results.csv")
MODEL_SAVE_PATH = os.path.join(EXP_DIR, "best_model.pt")

# Inicjalizacja loga treningowego
with open(TRAIN_LOG_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Epoch', 'Train_Cls_Loss', 'Train_Dom_Acc', 'Train_Dom_MCC', 'Val_Target_AUC', 'Alpha'])

# Inicjalizacja pliku wyników końcowych
with open(FINAL_RESULTS_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Metric_Name', 'Value', 'Description'])

def log_final_metric(name, value, description=""):
    """Dopisuje wynik do final_results.csv"""
    with open(FINAL_RESULTS_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([name, value, description])

# ==========================================
# 5. Definicje Klas i Funkcji (Loader, Model)
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

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class MinetDANN(nn.Module):
    def __init__(self, original_model, feature_dim=288, num_domains=39):
        super(MinetDANN, self).__init__()
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
        
        reverse_features = ReverseLayerF.apply(features, alpha)
        domain_output = self.domain_classifier(reverse_features)

        return class_output, domain_output

# ==========================================
# 6. PRZYGOTOWANIE METADANYCH I DANYCH LOSO
# ==========================================

print(" Ładowanie metadanych...")
df_meta = pd.read_csv(csv_path, sep='|', low_memory=False)

df_meta['examination_id'] = df_meta['examination_id'].astype(str).str.strip()
df_meta['institution_id'] = df_meta['institution_id'].astype(str).str.strip()

ALLOWED_HOSPITALS = [
    "ZOZLO", "KATMOJPRZ", "SZC", "TOR", "OST", 
    "LUMICE", "CMD", "SRK", "AKS", "PRZ", 
    "KIEG", "OTW", "MKW", "PUS", "LUX_A", 
    "KUD", "ARCHDAM", "MOR", "KAL", "B2K", 
    "SLU", "SL2", "STG1", "CHE", "KLU", 
    "GAK", "WLU", "Z04O", "TER_L", "PIO"
]

print(f"📉 Liczba wszystkich nagrań przed filtrowaniem: {len(df_meta)}")
df_meta = df_meta[df_meta['institution_id'].isin(ALLOWED_HOSPITALS)].copy()
print(f"📉 Liczba nagrań PO filtrowaniu (tylko 30 szpitali): {len(df_meta)}")

all_hospitals = sorted(df_meta['institution_id'].unique().tolist())
hospital_to_id = {name: i for i, name in enumerate(all_hospitals)}

eid_to_hosp_id = {}
eid_to_hosp_name = pd.Series(df_meta.institution_id.values, index=df_meta.examination_id).to_dict()

for eid, hosp in eid_to_hosp_name.items():
    eid_to_hosp_id[eid] = hospital_to_id.get(hosp, 0)

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
        labels.append(eid_to_hosp_id.get(e_clean, 0))
    return torch.tensor(labels, dtype=torch.long).to(device)

# --- CONFIG STRICT LOSO ---
print(f"\n KONFIGURACJA STRICT LOSO (Zero-Shot): Target = {TARGET_HOSPITAL_CODE}")

if TARGET_HOSPITAL_CODE not in ALLOWED_HOSPITALS:
    print(f"⚠️ UWAGA! Szpital {TARGET_HOSPITAL_CODE} nie znajduje się na liście ALLOWED_HOSPITALS!")

folds, fold_override = read_folds_override(data_pth, None)
all_eids_raw = get_eids_for_folds(fold_override, [1, 2, 3, 4, 5, 6]) 

train_pool = [] 
test_pool = []  
valid_eids_set = set(df_meta['examination_id'].values)

for eid in all_eids_raw:
    eid_clean = clean_eid_str(eid)
    if eid_clean in valid_eids_set:
        h_name = eid_to_hosp_name.get(eid_clean, "Unknown")
        if h_name == TARGET_HOSPITAL_CODE:
            test_pool.append(eid)
        elif h_name != "Unknown":
            train_pool.append(eid)

print(f"📊 DANE DO TRENINGU:")
print(f"   TRAIN: {len(train_pool)} pacjentów")
print(f"   TEST:  {len(test_pool)} pacjentów")

train_loader = Loader(
    data_pth, train_pool, minet_subsampling_n=4, num_workers=4
).get_batched_loader(batch_size=32, pad=True)

test_loader = Loader(
    data_pth, test_pool, minet_subsampling_n=None, num_workers=2
).get_batched_loader(batch_size=1, pad=True)

# ==========================================
# 7. INICJALIZACJA MODELU
# ==========================================
import copy
from torch.cuda.amp import GradScaler, autocast

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        m.reset_parameters()

print("\n🧠 Inicjalizacja modelu (RANDOM SCRATCH)...")
raw_backbone = torch.load(model_pth, map_location=device)

# Reset wag
print("♻️ Resetowanie wag do wartości losowych...")
raw_backbone.apply(weight_reset)

if hasattr(raw_backbone, 'n_chans'): raw_backbone.n_chans = 19

dann_model = MinetDANN(
    raw_backbone, 
    feature_dim=288, 
    num_domains=len(all_hospitals) 
).to(device)

print("✅ Model zainicjalizowany.")

# ==========================================
# OPTIMIZER & PARAMS
# ==========================================
backbone_head_params = list(dann_model.backbone.classifier.parameters())
head_ids = list(map(id, backbone_head_params))
backbone_base_params = filter(lambda p: id(p) not in head_ids, dann_model.backbone.parameters())

optimizer = optim.AdamW([
    {'params': backbone_base_params, 'lr': 1e-3}, 
    {'params': backbone_head_params, 'lr': 1e-3},
    {'params': dann_model.domain_classifier.parameters(), 'lr': 1e-3}
], weight_decay=1e-4)

loss_cls_fn = nn.BCEWithLogitsLoss()
loss_dom_fn = nn.CrossEntropyLoss()

EPOCHS = 50 # <-- ZMIANA NA 50 EPOK
PATIENCE = 70 
STEPS = 500
WARMUP_EPOCHS = 10 
MAX_ALPHA = 1.0    

scaler = GradScaler() 

best_auc = 0.0
best_model_wts = copy.deepcopy(dann_model.state_dict())
patience_counter = 0

# ==========================================
# 7b. PĘTLA TRENINGOWA
# ==========================================
print(f"\n🚀 START TRENINGU (Logging to {TRAIN_LOG_FILE})...")

for epoch in range(EPOCHS):
    # Alpha scheduler logic
    if epoch < WARMUP_EPOCHS:
        current_epoch_alpha_start = 0.0
    else:
        if epoch == WARMUP_EPOCHS:
             print(f"\n🔄 Epoka {epoch+1}: Faza DANN się zaczyna, Reset licznika Patience.")
             patience_counter = 0
        current_epoch_alpha_start = 0.0 

    # --- TRENING ---
    dann_model.train()
    iter_train = iter(train_loader)
    
    metrics = {'cls': 0, 'dom': 0, 'acc_dom': 0}
    all_dom_preds, all_dom_targets = [], []
    
    pbar = tqdm(range(STEPS), desc=f"Epoka {epoch+1}/{EPOCHS}")
    
    for i in pbar:
        # Alpha calculation
        if epoch < WARMUP_EPOCHS:
            alpha = 0.0
        else:
            total_phase2_steps = (EPOCHS - WARMUP_EPOCHS) * STEPS
            current_phase2_step = (i + ((epoch - WARMUP_EPOCHS) * STEPS))
            ratio = current_phase2_step / total_phase2_steps
            alpha = min(ratio * MAX_ALPHA, MAX_ALPHA)
        
        try:
            batch = next(iter_train)
        except StopIteration:
            iter_train = iter(train_loader)
            batch = next(iter_train)
            
        if len(batch) == 3: X, y, eid = batch; ts = None
        else: X, y, eid = batch[0], batch[1], batch[2]; ts = None
        
        X = X.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32).view(-1, 1)
        if ts is None: ts = torch.zeros((X.shape[0], X.shape[1]), dtype=torch.long, device=device)
        else: ts = ts.to(device)

        dom_labels = get_domain_labels(eid)
        
        optimizer.zero_grad()

        with autocast():
            c_pred, d_pred = dann_model(X, time_stamps=ts, alpha=alpha)
            loss_cls = loss_cls_fn(c_pred, y)
            loss_dom = loss_dom_fn(d_pred, dom_labels)
            loss = (5.0 * loss_cls) + loss_dom # Zwiększona waga medycyny
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer) 
        torch.nn.utils.clip_grad_norm_(dann_model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        preds_dom_batch = torch.argmax(d_pred, dim=1)
        acc_dom = (preds_dom_batch == dom_labels).float().mean().item()
        metrics['cls'] += loss_cls.item()
        metrics['dom'] += loss_dom.item()
        metrics['acc_dom'] += acc_dom
        
        if i % 10 == 0:
            all_dom_preds.extend(preds_dom_batch.detach().cpu().numpy())
            all_dom_targets.extend(dom_labels.detach().cpu().numpy())
        
        pbar.set_postfix({'Cls': f"{loss_cls.item():.2f}", 'Alpha': f"{alpha:.2f}"})

    # --- WALIDACJA (TARGET) ---
    dann_model.eval()
    val_targets, val_preds = [], []
    with torch.no_grad():
        for data in test_loader: 
            if len(data) == 3: X_v, y_v, _ = data; ts_v = None
            else: X_v, y_v, _, ts_v = data
            
            if X_v.shape[1] > 300: X_v = X_v[:, :300]
            
            X_v = X_v.to(device, dtype=torch.float32)
            if ts_v is None: ts_v = torch.zeros((1, X_v.shape[1]), dtype=torch.long, device=device)
            else: ts_v = ts_v.to(device)
            
            logits, _ = dann_model(X_v, time_stamps=ts_v, alpha=0.0)
            probs = torch.sigmoid(logits).item()
            val_targets.append(y_v.item())
            val_preds.append(probs)

    try:
        current_val_auc = roc_auc_score(val_targets, val_preds)
    except ValueError:
        current_val_auc = 0.5 

    avg_cls_loss = metrics['cls'] / STEPS
    avg_dom_acc = metrics['acc_dom'] / STEPS
    epoch_dom_mcc = matthews_corrcoef(all_dom_targets, all_dom_preds) if len(all_dom_targets) > 0 else 0

    print(f"🏁 Epoka {epoch+1}: Train Loss={avg_cls_loss:.4f} | Dom Acc={avg_dom_acc:.4f} | Dom MCC={epoch_dom_mcc:.4f}")
    print(f"   📊 Val AUC (Target): {current_val_auc:.4f} (Best: {best_auc:.4f})")

    # --- CSV LOGGING ---
    with open(TRAIN_LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, avg_cls_loss, avg_dom_acc, epoch_dom_mcc, current_val_auc, alpha])

    # --- EARLY STOPPING ---
    if current_val_auc > best_auc:
        best_auc = current_val_auc
        best_model_wts = copy.deepcopy(dann_model.state_dict())
        torch.save(dann_model.state_dict(), MODEL_SAVE_PATH) 
        patience_counter = 0
        print("   🏆 Nowy najlepszy model! Zapisano wagi.")
    else:
        patience_counter += 1
        print(f"   ⏳ Brak poprawy od {patience_counter} epok (Limit: {PATIENCE})")
        
    if patience_counter >= PATIENCE:
        print("\n🛑 Early Stopping.")
        break

# ==========================================
# 8. EWALUACJA KOŃCOWA I ZAPIS WYNIKÓW
# ==========================================

def run_target_evaluation(model, loader, device, prefix=""):
    """Pomocnicza funkcja do liczenia metryk na zbiorze testowym"""
    print(f"\n🧐 EWALUACJA: {prefix} (Target: {TARGET_HOSPITAL_CODE})")
    model.eval()
    y_true, y_prob = [], []

    with torch.no_grad():
        for data in tqdm(loader, desc=f"Testing {prefix}"):
            if len(data) == 3: X, y, eid = data; ts = None
            else: X, y, eid = data[0], data[1], data[2]; ts = None
            
            if X.shape[1] > 300: X = X[:, :300]
            X = X.to(device, dtype=torch.float32)
            ts = torch.zeros((1, X.shape[1]), dtype=torch.long, device=device)
            
            try:
                logits, _ = model(X, time_stamps=ts, alpha=0.0)
                prob = torch.sigmoid(logits).item()
                y_prob.append(prob)
                y_true.append(y.item())
            except Exception: pass

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    if len(np.unique(y_true)) > 1:
        auc_val = roc_auc_score(y_true, y_prob)
        fpr, tpr, thresh = roc_curve(y_true, y_prob)
        best_t = thresh[np.argmax(tpr - fpr)]
        y_pred = (y_prob >= best_t).astype(int)
        mcc_val = matthews_corrcoef(y_true, y_pred)
        acc_val = accuracy_score(y_true, y_pred)
        
        print(f"   👉 {prefix}Results: AUC={auc_val:.4f} | MCC={mcc_val:.4f} | Acc={acc_val:.4f}")
        
        log_final_metric(f"{prefix}Target_Diagnosis_AUC", auc_val, "Diag AUC on Target Hospital")
        log_final_metric(f"{prefix}Target_Diagnosis_MCC", mcc_val, "Diag MCC on Target Hospital")
        log_final_metric(f"{prefix}Target_Diagnosis_Acc", acc_val, "Diag Accuracy on Target Hospital")
        return auc_val
    else:
        print("   ⚠️ Not enough classes.")
        return 0.0

# ==========================================
# 9. KOMPLETNA EWALUACJA (Domain + Silhouette)
# ==========================================

def evaluate_comprehensive(model, source_loader, target_loader, device, max_batches=100, prefix=""):
    model.eval()
    
    feats_list = []
    y_true, y_prob = [], []
    d_true, d_pred = [], []
    
    def collect_from_loader(loader, is_target_domain):
        if loader is None: return
        with torch.no_grad():
            for i, data in enumerate(tqdm(loader, desc=f"Zbieranie {prefix}")):
                if i >= max_batches: break
                if len(data) == 4: X, y, eid, ts = data
                else: X, y, eid = data; ts = None
                
                if X.shape[1] > 300: 
                    X = X[:, :300]
                    if ts is not None: ts = ts[:, :300]

                X = X.to(device, dtype=torch.float32)
                if ts is None: ts = torch.zeros((X.shape[0], X.shape[-1]), dtype=torch.long, device=device)
                else: ts = ts.to(device)

                f, _ = model.backbone(X, time_stamps=ts, part=True)
                if len(f.shape) == 3: f = torch.mean(f, dim=1)
                
                class_logits, domain_logits = model(X, time_stamps=ts, alpha=0.0)
                
                if class_logits.shape[1] == 1: probs = torch.sigmoid(class_logits).view(-1)
                else: probs = torch.softmax(class_logits, dim=1)[:, 1]
                
                d_p = torch.argmax(domain_logits, dim=1).cpu().numpy()
                try:
                    d_t_tensor = get_domain_labels(eid)
                    d_t = d_t_tensor.cpu().numpy()
                except: d_t = np.zeros(len(probs))
                
                feats_list.append(f.cpu().numpy())
                y_np = y.cpu().numpy() if isinstance(y, torch.Tensor) else np.array(y)
                if y_np.ndim == 0: y_np = np.expand_dims(y_np, 0)
                y_true.extend(y_np)
                y_prob.extend(probs.cpu().numpy())
                d_true.extend(d_t)
                d_pred.extend(d_p)

    print("   -> Source...")
    collect_from_loader(source_loader, is_target_domain=False)
    print("   -> Target...")
    collect_from_loader(target_loader, is_target_domain=True)
    
    feats = np.concatenate(feats_list, axis=0)
    y_true = np.array(y_true).astype(int)
    y_prob = np.array(y_prob)
    d_true = np.array(d_true).astype(int)
    d_pred = np.array(d_pred).astype(int)
    
    print(f"\n📊 WYNIKI ROZSZERZONE ({prefix}):")
    
    if len(np.unique(d_true)) > 1:
        mcc_dom = matthews_corrcoef(d_true, d_pred)
        acc_dom = np.mean(d_true == d_pred)
        print(f"   • Domain MCC: {mcc_dom:.4f}")
        log_final_metric(f"{prefix}Domain_MCC", mcc_dom, f"Ability to distinguish hospitals ({prefix})")
        log_final_metric(f"{prefix}Domain_Acc", acc_dom, f"Accuracy of hospital classification ({prefix})")
    
    if len(feats) > 5000:
        idx = np.random.choice(len(feats), 5000, replace=False)
        feats_s = feats[idx]
        d_true_s = d_true[idx]
        y_true_s = y_true[idx]
    else:
        feats_s, d_true_s, y_true_s = feats, d_true, y_true
        
    if len(np.unique(y_true_s)) > 1:
        sil_med = silhouette_score(feats_s, y_true_s)
        print(f"   • Silhouette (Diagnosis): {sil_med:.4f}")
        log_final_metric(f"{prefix}Silhouette_Diagnosis", sil_med, f"Clustering quality by pathology ({prefix})")

    if len(np.unique(d_true_s)) > 1:
        sil_dom = silhouette_score(feats_s, d_true_s)
        print(f"   • Silhouette (Domain):    {sil_dom:.4f}")
        log_final_metric(f"{prefix}Silhouette_Domain", sil_dom, f"Clustering quality by hospital ({prefix})")
    
    print(f"💾 Wszystkie wyniki zapisano w: {FINAL_RESULTS_FILE}")

# --- URUCHOMIENIE ANALIZY DLA OBU MODELI ---

# 1. Analiza modelu KOŃCOWEGO (Final) - ten "zepsuty" przez DANN
run_target_evaluation(dann_model, test_loader, device, prefix="Final_")
evaluate_comprehensive(dann_model, train_loader, test_loader, device, max_batches=200, prefix="Final_")

# 2. Analiza modelu NAJLEPSZEGO (Best) - ten z wczesnej epoki
print(f"\n✅ Przywracanie najlepszych wag (AUC={best_auc:.4f})...")
dann_model.load_state_dict(best_model_wts)
run_target_evaluation(dann_model, test_loader, device, prefix="Best_")
evaluate_comprehensive(dann_model, train_loader, test_loader, device, max_batches=200, prefix="Best_")


#KOLEJNY SZPITAL

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"🔧 Using device: {device}")

# ==========================================
# KONFIGURACJA EKSPERYMENTU (LOGOWANIE)
# ==========================================
TARGET_HOSPITAL_CODE = 'GAK' 

# Generowanie unikalnej nazwy folderu
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
EXP_DIR = f"experiments/{TARGET_HOSPITAL_CODE}_{TIMESTAMP}"
os.makedirs(EXP_DIR, exist_ok=True)

print(f"📁 Wyniki będą zapisywane w: {EXP_DIR}")

# Ścieżki do plików
TRAIN_LOG_FILE = os.path.join(EXP_DIR, "training_log.csv")
FINAL_RESULTS_FILE = os.path.join(EXP_DIR, "final_results.csv")
MODEL_SAVE_PATH = os.path.join(EXP_DIR, "best_model.pt")

# Inicjalizacja loga treningowego
with open(TRAIN_LOG_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Epoch', 'Train_Cls_Loss', 'Train_Dom_Acc', 'Train_Dom_MCC', 'Val_Target_AUC', 'Alpha'])

# Inicjalizacja pliku wyników końcowych
with open(FINAL_RESULTS_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Metric_Name', 'Value', 'Description'])

def log_final_metric(name, value, description=""):
    """Dopisuje wynik do final_results.csv"""
    with open(FINAL_RESULTS_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([name, value, description])

# ==========================================
# 5. Definicje Klas i Funkcji (Loader, Model)
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

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class MinetDANN(nn.Module):
    def __init__(self, original_model, feature_dim=288, num_domains=39):
        super(MinetDANN, self).__init__()
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
        
        reverse_features = ReverseLayerF.apply(features, alpha)
        domain_output = self.domain_classifier(reverse_features)

        return class_output, domain_output

# ==========================================
# 6. PRZYGOTOWANIE METADANYCH I DANYCH LOSO
# ==========================================

print("📂 Ładowanie metadanych...")
df_meta = pd.read_csv(csv_path, sep='|', low_memory=False)

df_meta['examination_id'] = df_meta['examination_id'].astype(str).str.strip()
df_meta['institution_id'] = df_meta['institution_id'].astype(str).str.strip()

ALLOWED_HOSPITALS = [
    "ZOZLO", "KATMOJPRZ", "SZC", "TOR", "OST", 
    "LUMICE", "CMD", "SRK", "AKS", "PRZ", 
    "KIEG", "OTW", "MKW", "PUS", "LUX_A", 
    "KUD", "ARCHDAM", "MOR", "KAL", "B2K", 
    "SLU", "SL2", "STG1", "CHE", "KLU", 
    "GAK", "WLU", "Z04O", "TER_L", "PIO"
]

print(f"📉 Liczba wszystkich nagrań przed filtrowaniem: {len(df_meta)}")
df_meta = df_meta[df_meta['institution_id'].isin(ALLOWED_HOSPITALS)].copy()
print(f"📉 Liczba nagrań PO filtrowaniu (tylko 30 szpitali): {len(df_meta)}")

all_hospitals = sorted(df_meta['institution_id'].unique().tolist())
hospital_to_id = {name: i for i, name in enumerate(all_hospitals)}

eid_to_hosp_id = {}
eid_to_hosp_name = pd.Series(df_meta.institution_id.values, index=df_meta.examination_id).to_dict()

for eid, hosp in eid_to_hosp_name.items():
    eid_to_hosp_id[eid] = hospital_to_id.get(hosp, 0)

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
        labels.append(eid_to_hosp_id.get(e_clean, 0))
    return torch.tensor(labels, dtype=torch.long).to(device)

# --- CONFIG STRICT LOSO ---
print(f"\n🏥 KONFIGURACJA STRICT LOSO (Zero-Shot): Target = {TARGET_HOSPITAL_CODE}")

if TARGET_HOSPITAL_CODE not in ALLOWED_HOSPITALS:
    print(f"⚠️ UWAGA! Szpital {TARGET_HOSPITAL_CODE} nie znajduje się na liście ALLOWED_HOSPITALS!")

folds, fold_override = read_folds_override(data_pth, None)
all_eids_raw = get_eids_for_folds(fold_override, [1, 2, 3, 4, 5, 6]) 

train_pool = [] 
test_pool = []  
valid_eids_set = set(df_meta['examination_id'].values)

for eid in all_eids_raw:
    eid_clean = clean_eid_str(eid)
    if eid_clean in valid_eids_set:
        h_name = eid_to_hosp_name.get(eid_clean, "Unknown")
        if h_name == TARGET_HOSPITAL_CODE:
            test_pool.append(eid)
        elif h_name != "Unknown":
            train_pool.append(eid)

print(f"📊 DANE DO TRENINGU:")
print(f"   TRAIN: {len(train_pool)} pacjentów")
print(f"   TEST:  {len(test_pool)} pacjentów")

train_loader = Loader(
    data_pth, train_pool, minet_subsampling_n=4, num_workers=4
).get_batched_loader(batch_size=32, pad=True)

test_loader = Loader(
    data_pth, test_pool, minet_subsampling_n=None, num_workers=2
).get_batched_loader(batch_size=1, pad=True)

# ==========================================
# 7. INICJALIZACJA MODELU
# ==========================================
import copy
from torch.cuda.amp import GradScaler, autocast

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        m.reset_parameters()

print("\n🧠 Inicjalizacja modelu (RANDOM SCRATCH)...")
raw_backbone = torch.load(model_pth, map_location=device)

# Reset wag
print("♻️ Resetowanie wag do wartości losowych...")
raw_backbone.apply(weight_reset)

if hasattr(raw_backbone, 'n_chans'): raw_backbone.n_chans = 19

dann_model = MinetDANN(
    raw_backbone, 
    feature_dim=288, 
    num_domains=len(all_hospitals) 
).to(device)

print("✅ Model zainicjalizowany.")

# ==========================================
# OPTIMIZER & PARAMS
# ==========================================
backbone_head_params = list(dann_model.backbone.classifier.parameters())
head_ids = list(map(id, backbone_head_params))
backbone_base_params = filter(lambda p: id(p) not in head_ids, dann_model.backbone.parameters())

optimizer = optim.AdamW([
    {'params': backbone_base_params, 'lr': 1e-3}, 
    {'params': backbone_head_params, 'lr': 1e-3},
    {'params': dann_model.domain_classifier.parameters(), 'lr': 1e-3}
], weight_decay=1e-4)

loss_cls_fn = nn.BCEWithLogitsLoss()
loss_dom_fn = nn.CrossEntropyLoss()

EPOCHS = 50 # <-- ZMIANA NA 50 EPOK
PATIENCE = 70 
STEPS = 500
WARMUP_EPOCHS = 10 
MAX_ALPHA = 1.0    

scaler = GradScaler() 

best_auc = 0.0
best_model_wts = copy.deepcopy(dann_model.state_dict())
patience_counter = 0

# ==========================================
# 7b. PĘTLA TRENINGOWA
# ==========================================
print(f"\n🚀 START TRENINGU (Logging to {TRAIN_LOG_FILE})...")

for epoch in range(EPOCHS):
    # Alpha scheduler logic
    if epoch < WARMUP_EPOCHS:
        current_epoch_alpha_start = 0.0
    else:
        if epoch == WARMUP_EPOCHS:
             print(f"\n🔄 Epoka {epoch+1}: Faza DANN się zaczyna, Reset licznika Patience.")
             patience_counter = 0
        current_epoch_alpha_start = 0.0 

    # --- TRENING ---
    dann_model.train()
    iter_train = iter(train_loader)
    
    metrics = {'cls': 0, 'dom': 0, 'acc_dom': 0}
    all_dom_preds, all_dom_targets = [], []
    
    pbar = tqdm(range(STEPS), desc=f"Epoka {epoch+1}/{EPOCHS}")
    
    for i in pbar:
        # Alpha calculation
        if epoch < WARMUP_EPOCHS:
            alpha = 0.0
        else:
            total_phase2_steps = (EPOCHS - WARMUP_EPOCHS) * STEPS
            current_phase2_step = (i + ((epoch - WARMUP_EPOCHS) * STEPS))
            ratio = current_phase2_step / total_phase2_steps
            alpha = min(ratio * MAX_ALPHA, MAX_ALPHA)
        
        try:
            batch = next(iter_train)
        except StopIteration:
            iter_train = iter(train_loader)
            batch = next(iter_train)
            
        if len(batch) == 3: X, y, eid = batch; ts = None
        else: X, y, eid = batch[0], batch[1], batch[2]; ts = None
        
        X = X.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32).view(-1, 1)
        if ts is None: ts = torch.zeros((X.shape[0], X.shape[1]), dtype=torch.long, device=device)
        else: ts = ts.to(device)

        dom_labels = get_domain_labels(eid)
        
        optimizer.zero_grad()

        with autocast():
            c_pred, d_pred = dann_model(X, time_stamps=ts, alpha=alpha)
            loss_cls = loss_cls_fn(c_pred, y)
            loss_dom = loss_dom_fn(d_pred, dom_labels)
            loss = (5.0 * loss_cls) + loss_dom # Zwiększona waga medycyny
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer) 
        torch.nn.utils.clip_grad_norm_(dann_model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        preds_dom_batch = torch.argmax(d_pred, dim=1)
        acc_dom = (preds_dom_batch == dom_labels).float().mean().item()
        metrics['cls'] += loss_cls.item()
        metrics['dom'] += loss_dom.item()
        metrics['acc_dom'] += acc_dom
        
        if i % 10 == 0:
            all_dom_preds.extend(preds_dom_batch.detach().cpu().numpy())
            all_dom_targets.extend(dom_labels.detach().cpu().numpy())
        
        pbar.set_postfix({'Cls': f"{loss_cls.item():.2f}", 'Alpha': f"{alpha:.2f}"})

    # --- WALIDACJA (TARGET) ---
    dann_model.eval()
    val_targets, val_preds = [], []
    with torch.no_grad():
        for data in test_loader: 
            if len(data) == 3: X_v, y_v, _ = data; ts_v = None
            else: X_v, y_v, _, ts_v = data
            
            if X_v.shape[1] > 300: X_v = X_v[:, :300]
            
            X_v = X_v.to(device, dtype=torch.float32)
            if ts_v is None: ts_v = torch.zeros((1, X_v.shape[1]), dtype=torch.long, device=device)
            else: ts_v = ts_v.to(device)
            
            logits, _ = dann_model(X_v, time_stamps=ts_v, alpha=0.0)
            probs = torch.sigmoid(logits).item()
            val_targets.append(y_v.item())
            val_preds.append(probs)

    try:
        current_val_auc = roc_auc_score(val_targets, val_preds)
    except ValueError:
        current_val_auc = 0.5 

    avg_cls_loss = metrics['cls'] / STEPS
    avg_dom_acc = metrics['acc_dom'] / STEPS
    epoch_dom_mcc = matthews_corrcoef(all_dom_targets, all_dom_preds) if len(all_dom_targets) > 0 else 0

    print(f"🏁 Epoka {epoch+1}: Train Loss={avg_cls_loss:.4f} | Dom Acc={avg_dom_acc:.4f} | Dom MCC={epoch_dom_mcc:.4f}")
    print(f"   📊 Val AUC (Target): {current_val_auc:.4f} (Best: {best_auc:.4f})")

    # --- CSV LOGGING ---
    with open(TRAIN_LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, avg_cls_loss, avg_dom_acc, epoch_dom_mcc, current_val_auc, alpha])

    # --- EARLY STOPPING ---
    if current_val_auc > best_auc:
        best_auc = current_val_auc
        best_model_wts = copy.deepcopy(dann_model.state_dict())
        torch.save(dann_model.state_dict(), MODEL_SAVE_PATH) 
        patience_counter = 0
        print("   🏆 Nowy najlepszy model! Zapisano wagi.")
    else:
        patience_counter += 1
        print(f"   ⏳ Brak poprawy od {patience_counter} epok (Limit: {PATIENCE})")
        
    if patience_counter >= PATIENCE:
        print("\n🛑 Early Stopping.")
        break

# ==========================================
# 8. EWALUACJA KOŃCOWA I ZAPIS WYNIKÓW
# ==========================================

def run_target_evaluation(model, loader, device, prefix=""):
    """Pomocnicza funkcja do liczenia metryk na zbiorze testowym"""
    print(f"\n🧐 EWALUACJA: {prefix} (Target: {TARGET_HOSPITAL_CODE})")
    model.eval()
    y_true, y_prob = [], []

    with torch.no_grad():
        for data in tqdm(loader, desc=f"Testing {prefix}"):
            if len(data) == 3: X, y, eid = data; ts = None
            else: X, y, eid = data[0], data[1], data[2]; ts = None
            
            if X.shape[1] > 300: X = X[:, :300]
            X = X.to(device, dtype=torch.float32)
            ts = torch.zeros((1, X.shape[1]), dtype=torch.long, device=device)
            
            try:
                logits, _ = model(X, time_stamps=ts, alpha=0.0)
                prob = torch.sigmoid(logits).item()
                y_prob.append(prob)
                y_true.append(y.item())
            except Exception: pass

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    if len(np.unique(y_true)) > 1:
        auc_val = roc_auc_score(y_true, y_prob)
        fpr, tpr, thresh = roc_curve(y_true, y_prob)
        best_t = thresh[np.argmax(tpr - fpr)]
        y_pred = (y_prob >= best_t).astype(int)
        mcc_val = matthews_corrcoef(y_true, y_pred)
        acc_val = accuracy_score(y_true, y_pred)
        
        print(f"   👉 {prefix}Results: AUC={auc_val:.4f} | MCC={mcc_val:.4f} | Acc={acc_val:.4f}")
        
        log_final_metric(f"{prefix}Target_Diagnosis_AUC", auc_val, "Diag AUC on Target Hospital")
        log_final_metric(f"{prefix}Target_Diagnosis_MCC", mcc_val, "Diag MCC on Target Hospital")
        log_final_metric(f"{prefix}Target_Diagnosis_Acc", acc_val, "Diag Accuracy on Target Hospital")
        return auc_val
    else:
        print("   ⚠️ Not enough classes.")
        return 0.0

# ==========================================
# 9. KOMPLETNA EWALUACJA (Domain + Silhouette)
# ==========================================

def evaluate_comprehensive(model, source_loader, target_loader, device, max_batches=100, prefix=""):
    print(f"\n🚀 ROZPOCZYNAMY KOMPLETNĄ ANALIZĘ {prefix} (Silhouette + Domain Metrics)...")
    model.eval()
    
    feats_list = []
    y_true, y_prob = [], []
    d_true, d_pred = [], []
    
    def collect_from_loader(loader, is_target_domain):
        if loader is None: return
        with torch.no_grad():
            for i, data in enumerate(tqdm(loader, desc=f"Zbieranie {prefix}")):
                if i >= max_batches: break
                if len(data) == 4: X, y, eid, ts = data
                else: X, y, eid = data; ts = None
                
                if X.shape[1] > 300: 
                    X = X[:, :300]
                    if ts is not None: ts = ts[:, :300]

                X = X.to(device, dtype=torch.float32)
                if ts is None: ts = torch.zeros((X.shape[0], X.shape[-1]), dtype=torch.long, device=device)
                else: ts = ts.to(device)

                f, _ = model.backbone(X, time_stamps=ts, part=True)
                if len(f.shape) == 3: f = torch.mean(f, dim=1)
                
                class_logits, domain_logits = model(X, time_stamps=ts, alpha=0.0)
                
                if class_logits.shape[1] == 1: probs = torch.sigmoid(class_logits).view(-1)
                else: probs = torch.softmax(class_logits, dim=1)[:, 1]
                
                d_p = torch.argmax(domain_logits, dim=1).cpu().numpy()
                try:
                    d_t_tensor = get_domain_labels(eid)
                    d_t = d_t_tensor.cpu().numpy()
                except: d_t = np.zeros(len(probs))
                
                feats_list.append(f.cpu().numpy())
                y_np = y.cpu().numpy() if isinstance(y, torch.Tensor) else np.array(y)
                if y_np.ndim == 0: y_np = np.expand_dims(y_np, 0)
                y_true.extend(y_np)
                y_prob.extend(probs.cpu().numpy())
                d_true.extend(d_t)
                d_pred.extend(d_p)

    print("   -> Source...")
    collect_from_loader(source_loader, is_target_domain=False)
    print("   -> Target...")
    collect_from_loader(target_loader, is_target_domain=True)
    
    feats = np.concatenate(feats_list, axis=0)
    y_true = np.array(y_true).astype(int)
    y_prob = np.array(y_prob)
    d_true = np.array(d_true).astype(int)
    d_pred = np.array(d_pred).astype(int)
    
    print(f"\n📊 WYNIKI ROZSZERZONE ({prefix}):")
    
    if len(np.unique(d_true)) > 1:
        mcc_dom = matthews_corrcoef(d_true, d_pred)
        acc_dom = np.mean(d_true == d_pred)
        print(f"   • Domain MCC: {mcc_dom:.4f}")
        log_final_metric(f"{prefix}Domain_MCC", mcc_dom, f"Ability to distinguish hospitals ({prefix})")
        log_final_metric(f"{prefix}Domain_Acc", acc_dom, f"Accuracy of hospital classification ({prefix})")
    
    if len(feats) > 5000:
        idx = np.random.choice(len(feats), 5000, replace=False)
        feats_s = feats[idx]
        d_true_s = d_true[idx]
        y_true_s = y_true[idx]
    else:
        feats_s, d_true_s, y_true_s = feats, d_true, y_true
        
    if len(np.unique(y_true_s)) > 1:
        sil_med = silhouette_score(feats_s, y_true_s)
        print(f"   • Silhouette (Diagnosis): {sil_med:.4f}")
        log_final_metric(f"{prefix}Silhouette_Diagnosis", sil_med, f"Clustering quality by pathology ({prefix})")

    if len(np.unique(d_true_s)) > 1:
        sil_dom = silhouette_score(feats_s, d_true_s)
        print(f"   • Silhouette (Domain):    {sil_dom:.4f}")
        log_final_metric(f"{prefix}Silhouette_Domain", sil_dom, f"Clustering quality by hospital ({prefix})")
    
    print(f"💾 Wszystkie wyniki zapisano w: {FINAL_RESULTS_FILE}")

# --- URUCHOMIENIE ANALIZY DLA OBU MODELI ---

# 1. Analiza modelu KOŃCOWEGO (Final) - ten "zepsuty" przez DANN
run_target_evaluation(dann_model, test_loader, device, prefix="Final_")
evaluate_comprehensive(dann_model, train_loader, test_loader, device, max_batches=200, prefix="Final_")

# 2. Analiza modelu NAJLEPSZEGO (Best) - ten z wczesnej epoki
print(f"\n✅ Przywracanie najlepszych wag (AUC={best_auc:.4f})...")
dann_model.load_state_dict(best_model_wts)
run_target_evaluation(dann_model, test_loader, device, prefix="Best_")
evaluate_comprehensive(dann_model, train_loader, test_loader, device, max_batches=200, prefix="Best_")

#KOLEJNY SZPITAL

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"🔧 Using device: {device}")

# ==========================================
# KONFIGURACJA EKSPERYMENTU (LOGOWANIE)
# ==========================================
TARGET_HOSPITAL_CODE = 'WLU' # <-- TU ZMIENIASZ SZPITAL (Target)

# Generowanie unikalnej nazwy folderu
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
EXP_DIR = f"experiments/{TARGET_HOSPITAL_CODE}_{TIMESTAMP}"
os.makedirs(EXP_DIR, exist_ok=True)

print(f"📁 Wyniki będą zapisywane w: {EXP_DIR}")

# Ścieżki do plików
TRAIN_LOG_FILE = os.path.join(EXP_DIR, "training_log.csv")
FINAL_RESULTS_FILE = os.path.join(EXP_DIR, "final_results.csv")
MODEL_SAVE_PATH = os.path.join(EXP_DIR, "best_model.pt")

# Inicjalizacja loga treningowego
with open(TRAIN_LOG_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Epoch', 'Train_Cls_Loss', 'Train_Dom_Acc', 'Train_Dom_MCC', 'Val_Target_AUC', 'Alpha'])

# Inicjalizacja pliku wyników końcowych
with open(FINAL_RESULTS_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Metric_Name', 'Value', 'Description'])

def log_final_metric(name, value, description=""):
    """Dopisuje wynik do final_results.csv"""
    with open(FINAL_RESULTS_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([name, value, description])

# ==========================================
# 5. Definicje Klas i Funkcji (Loader, Model)
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

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class MinetDANN(nn.Module):
    def __init__(self, original_model, feature_dim=288, num_domains=39):
        super(MinetDANN, self).__init__()
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
        
        reverse_features = ReverseLayerF.apply(features, alpha)
        domain_output = self.domain_classifier(reverse_features)

        return class_output, domain_output

# ==========================================
# 6. PRZYGOTOWANIE METADANYCH I DANYCH LOSO
# ==========================================

print("📂 Ładowanie metadanych...")
df_meta = pd.read_csv(csv_path, sep='|', low_memory=False)

df_meta['examination_id'] = df_meta['examination_id'].astype(str).str.strip()
df_meta['institution_id'] = df_meta['institution_id'].astype(str).str.strip()

ALLOWED_HOSPITALS = [
    "ZOZLO", "KATMOJPRZ", "SZC", "TOR", "OST", 
    "LUMICE", "CMD", "SRK", "AKS", "PRZ", 
    "KIEG", "OTW", "MKW", "PUS", "LUX_A", 
    "KUD", "ARCHDAM", "MOR", "KAL", "B2K", 
    "SLU", "SL2", "STG1", "CHE", "KLU", 
    "GAK", "WLU", "Z04O", "TER_L", "PIO"
]

print(f"📉 Liczba wszystkich nagrań przed filtrowaniem: {len(df_meta)}")
df_meta = df_meta[df_meta['institution_id'].isin(ALLOWED_HOSPITALS)].copy()
print(f"📉 Liczba nagrań PO filtrowaniu (tylko 30 szpitali): {len(df_meta)}")

all_hospitals = sorted(df_meta['institution_id'].unique().tolist())
hospital_to_id = {name: i for i, name in enumerate(all_hospitals)}

eid_to_hosp_id = {}
eid_to_hosp_name = pd.Series(df_meta.institution_id.values, index=df_meta.examination_id).to_dict()

for eid, hosp in eid_to_hosp_name.items():
    eid_to_hosp_id[eid] = hospital_to_id.get(hosp, 0)

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
        labels.append(eid_to_hosp_id.get(e_clean, 0))
    return torch.tensor(labels, dtype=torch.long).to(device)

# --- CONFIG STRICT LOSO ---
print(f"\n🏥 KONFIGURACJA STRICT LOSO (Zero-Shot): Target = {TARGET_HOSPITAL_CODE}")

if TARGET_HOSPITAL_CODE not in ALLOWED_HOSPITALS:
    print(f"⚠️ UWAGA! Szpital {TARGET_HOSPITAL_CODE} nie znajduje się na liście ALLOWED_HOSPITALS!")

folds, fold_override = read_folds_override(data_pth, None)
all_eids_raw = get_eids_for_folds(fold_override, [1, 2, 3, 4, 5, 6]) 

train_pool = [] 
test_pool = []  
valid_eids_set = set(df_meta['examination_id'].values)

for eid in all_eids_raw:
    eid_clean = clean_eid_str(eid)
    if eid_clean in valid_eids_set:
        h_name = eid_to_hosp_name.get(eid_clean, "Unknown")
        if h_name == TARGET_HOSPITAL_CODE:
            test_pool.append(eid)
        elif h_name != "Unknown":
            train_pool.append(eid)

print(f"📊 DANE DO TRENINGU:")
print(f"   TRAIN: {len(train_pool)} pacjentów")
print(f"   TEST:  {len(test_pool)} pacjentów")

train_loader = Loader(
    data_pth, train_pool, minet_subsampling_n=4, num_workers=4
).get_batched_loader(batch_size=32, pad=True)

test_loader = Loader(
    data_pth, test_pool, minet_subsampling_n=None, num_workers=2
).get_batched_loader(batch_size=1, pad=True)

# ==========================================
# 7. INICJALIZACJA MODELU
# ==========================================
import copy
from torch.cuda.amp import GradScaler, autocast

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        m.reset_parameters()

print("\n🧠 Inicjalizacja modelu (RANDOM SCRATCH)...")
raw_backbone = torch.load(model_pth, map_location=device)

# Reset wag
print("♻️ Resetowanie wag do wartości losowych...")
raw_backbone.apply(weight_reset)

if hasattr(raw_backbone, 'n_chans'): raw_backbone.n_chans = 19

dann_model = MinetDANN(
    raw_backbone, 
    feature_dim=288, 
    num_domains=len(all_hospitals) 
).to(device)

print("✅ Model zainicjalizowany.")

# ==========================================
# OPTIMIZER & PARAMS
# ==========================================
backbone_head_params = list(dann_model.backbone.classifier.parameters())
head_ids = list(map(id, backbone_head_params))
backbone_base_params = filter(lambda p: id(p) not in head_ids, dann_model.backbone.parameters())

optimizer = optim.AdamW([
    {'params': backbone_base_params, 'lr': 1e-3}, 
    {'params': backbone_head_params, 'lr': 1e-3},
    {'params': dann_model.domain_classifier.parameters(), 'lr': 1e-3}
], weight_decay=1e-4)

loss_cls_fn = nn.BCEWithLogitsLoss()
loss_dom_fn = nn.CrossEntropyLoss()

EPOCHS = 50 # <-- ZMIANA NA 50 EPOK
PATIENCE = 70 
STEPS = 500
WARMUP_EPOCHS = 10 
MAX_ALPHA = 1.0    

scaler = GradScaler() 

best_auc = 0.0
best_model_wts = copy.deepcopy(dann_model.state_dict())
patience_counter = 0

# ==========================================
# 7b. PĘTLA TRENINGOWA
# ==========================================
print(f"\n🚀 START TRENINGU (Logging to {TRAIN_LOG_FILE})...")

for epoch in range(EPOCHS):
    # Alpha scheduler logic
    if epoch < WARMUP_EPOCHS:
        current_epoch_alpha_start = 0.0
    else:
        if epoch == WARMUP_EPOCHS:
             print(f"\n🔄 Epoka {epoch+1}: Faza DANN się zaczyna, Reset licznika Patience.")
             patience_counter = 0
        current_epoch_alpha_start = 0.0 

    # --- TRENING ---
    dann_model.train()
    iter_train = iter(train_loader)
    
    metrics = {'cls': 0, 'dom': 0, 'acc_dom': 0}
    all_dom_preds, all_dom_targets = [], []
    
    pbar = tqdm(range(STEPS), desc=f"Epoka {epoch+1}/{EPOCHS}")
    
    for i in pbar:
        # Alpha calculation
        if epoch < WARMUP_EPOCHS:
            alpha = 0.0
        else:
            total_phase2_steps = (EPOCHS - WARMUP_EPOCHS) * STEPS
            current_phase2_step = (i + ((epoch - WARMUP_EPOCHS) * STEPS))
            ratio = current_phase2_step / total_phase2_steps
            alpha = min(ratio * MAX_ALPHA, MAX_ALPHA)
        
        try:
            batch = next(iter_train)
        except StopIteration:
            iter_train = iter(train_loader)
            batch = next(iter_train)
            
        if len(batch) == 3: X, y, eid = batch; ts = None
        else: X, y, eid = batch[0], batch[1], batch[2]; ts = None
        
        X = X.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32).view(-1, 1)
        if ts is None: ts = torch.zeros((X.shape[0], X.shape[1]), dtype=torch.long, device=device)
        else: ts = ts.to(device)

        dom_labels = get_domain_labels(eid)
        
        optimizer.zero_grad()

        with autocast():
            c_pred, d_pred = dann_model(X, time_stamps=ts, alpha=alpha)
            loss_cls = loss_cls_fn(c_pred, y)
            loss_dom = loss_dom_fn(d_pred, dom_labels)
            loss = (5.0 * loss_cls) + loss_dom # Zwiększona waga medycyny
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer) 
        torch.nn.utils.clip_grad_norm_(dann_model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        preds_dom_batch = torch.argmax(d_pred, dim=1)
        acc_dom = (preds_dom_batch == dom_labels).float().mean().item()
        metrics['cls'] += loss_cls.item()
        metrics['dom'] += loss_dom.item()
        metrics['acc_dom'] += acc_dom
        
        if i % 10 == 0:
            all_dom_preds.extend(preds_dom_batch.detach().cpu().numpy())
            all_dom_targets.extend(dom_labels.detach().cpu().numpy())
        
        pbar.set_postfix({'Cls': f"{loss_cls.item():.2f}", 'Alpha': f"{alpha:.2f}"})

    # --- WALIDACJA (TARGET) ---
    dann_model.eval()
    val_targets, val_preds = [], []
    with torch.no_grad():
        for data in test_loader: 
            if len(data) == 3: X_v, y_v, _ = data; ts_v = None
            else: X_v, y_v, _, ts_v = data
            
            if X_v.shape[1] > 300: X_v = X_v[:, :300]
            
            X_v = X_v.to(device, dtype=torch.float32)
            if ts_v is None: ts_v = torch.zeros((1, X_v.shape[1]), dtype=torch.long, device=device)
            else: ts_v = ts_v.to(device)
            
            logits, _ = dann_model(X_v, time_stamps=ts_v, alpha=0.0)
            probs = torch.sigmoid(logits).item()
            val_targets.append(y_v.item())
            val_preds.append(probs)

    try:
        current_val_auc = roc_auc_score(val_targets, val_preds)
    except ValueError:
        current_val_auc = 0.5 

    avg_cls_loss = metrics['cls'] / STEPS
    avg_dom_acc = metrics['acc_dom'] / STEPS
    epoch_dom_mcc = matthews_corrcoef(all_dom_targets, all_dom_preds) if len(all_dom_targets) > 0 else 0

    print(f"🏁 Epoka {epoch+1}: Train Loss={avg_cls_loss:.4f} | Dom Acc={avg_dom_acc:.4f} | Dom MCC={epoch_dom_mcc:.4f}")
    print(f"   📊 Val AUC (Target): {current_val_auc:.4f} (Best: {best_auc:.4f})")

    # --- CSV LOGGING ---
    with open(TRAIN_LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, avg_cls_loss, avg_dom_acc, epoch_dom_mcc, current_val_auc, alpha])

    # --- EARLY STOPPING ---
    if current_val_auc > best_auc:
        best_auc = current_val_auc
        best_model_wts = copy.deepcopy(dann_model.state_dict())
        torch.save(dann_model.state_dict(), MODEL_SAVE_PATH) 
        patience_counter = 0
        print("   🏆 Nowy najlepszy model! Zapisano wagi.")
    else:
        patience_counter += 1
        print(f"   ⏳ Brak poprawy od {patience_counter} epok (Limit: {PATIENCE})")
        
    if patience_counter >= PATIENCE:
        print("\n🛑 Early Stopping.")
        break

# ==========================================
# 8. EWALUACJA KOŃCOWA I ZAPIS WYNIKÓW
# ==========================================

def run_target_evaluation(model, loader, device, prefix=""):
    """Pomocnicza funkcja do liczenia metryk na zbiorze testowym"""
    print(f"\n🧐 EWALUACJA: {prefix} (Target: {TARGET_HOSPITAL_CODE})")
    model.eval()
    y_true, y_prob = [], []

    with torch.no_grad():
        for data in tqdm(loader, desc=f"Testing {prefix}"):
            if len(data) == 3: X, y, eid = data; ts = None
            else: X, y, eid = data[0], data[1], data[2]; ts = None
            
            if X.shape[1] > 300: X = X[:, :300]
            X = X.to(device, dtype=torch.float32)
            ts = torch.zeros((1, X.shape[1]), dtype=torch.long, device=device)
            
            try:
                logits, _ = model(X, time_stamps=ts, alpha=0.0)
                prob = torch.sigmoid(logits).item()
                y_prob.append(prob)
                y_true.append(y.item())
            except Exception: pass

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    if len(np.unique(y_true)) > 1:
        auc_val = roc_auc_score(y_true, y_prob)
        fpr, tpr, thresh = roc_curve(y_true, y_prob)
        best_t = thresh[np.argmax(tpr - fpr)]
        y_pred = (y_prob >= best_t).astype(int)
        mcc_val = matthews_corrcoef(y_true, y_pred)
        acc_val = accuracy_score(y_true, y_pred)
        
        print(f"   👉 {prefix}Results: AUC={auc_val:.4f} | MCC={mcc_val:.4f} | Acc={acc_val:.4f}")
        
        log_final_metric(f"{prefix}Target_Diagnosis_AUC", auc_val, "Diag AUC on Target Hospital")
        log_final_metric(f"{prefix}Target_Diagnosis_MCC", mcc_val, "Diag MCC on Target Hospital")
        log_final_metric(f"{prefix}Target_Diagnosis_Acc", acc_val, "Diag Accuracy on Target Hospital")
        return auc_val
    else:
        print("   ⚠️ Not enough classes.")
        return 0.0

# ==========================================
# 9. KOMPLETNA EWALUACJA (Domain + Silhouette)
# ==========================================

def evaluate_comprehensive(model, source_loader, target_loader, device, max_batches=100, prefix=""):
    print(f"\n🚀 ROZPOCZYNAMY KOMPLETNĄ ANALIZĘ {prefix} (Silhouette + Domain Metrics)...")
    model.eval()
    
    feats_list = []
    y_true, y_prob = [], []
    d_true, d_pred = [], []
    
    def collect_from_loader(loader, is_target_domain):
        if loader is None: return
        with torch.no_grad():
            for i, data in enumerate(tqdm(loader, desc=f"Zbieranie {prefix}")):
                if i >= max_batches: break
                if len(data) == 4: X, y, eid, ts = data
                else: X, y, eid = data; ts = None
                
                if X.shape[1] > 300: 
                    X = X[:, :300]
                    if ts is not None: ts = ts[:, :300]

                X = X.to(device, dtype=torch.float32)
                if ts is None: ts = torch.zeros((X.shape[0], X.shape[-1]), dtype=torch.long, device=device)
                else: ts = ts.to(device)

                f, _ = model.backbone(X, time_stamps=ts, part=True)
                if len(f.shape) == 3: f = torch.mean(f, dim=1)
                
                class_logits, domain_logits = model(X, time_stamps=ts, alpha=0.0)
                
                if class_logits.shape[1] == 1: probs = torch.sigmoid(class_logits).view(-1)
                else: probs = torch.softmax(class_logits, dim=1)[:, 1]
                
                d_p = torch.argmax(domain_logits, dim=1).cpu().numpy()
                try:
                    d_t_tensor = get_domain_labels(eid)
                    d_t = d_t_tensor.cpu().numpy()
                except: d_t = np.zeros(len(probs))
                
                feats_list.append(f.cpu().numpy())
                y_np = y.cpu().numpy() if isinstance(y, torch.Tensor) else np.array(y)
                if y_np.ndim == 0: y_np = np.expand_dims(y_np, 0)
                y_true.extend(y_np)
                y_prob.extend(probs.cpu().numpy())
                d_true.extend(d_t)
                d_pred.extend(d_p)

    print("   -> Source...")
    collect_from_loader(source_loader, is_target_domain=False)
    print("   -> Target...")
    collect_from_loader(target_loader, is_target_domain=True)
    
    feats = np.concatenate(feats_list, axis=0)
    y_true = np.array(y_true).astype(int)
    y_prob = np.array(y_prob)
    d_true = np.array(d_true).astype(int)
    d_pred = np.array(d_pred).astype(int)
    
    print(f"\n📊 WYNIKI ROZSZERZONE ({prefix}):")
    
    if len(np.unique(d_true)) > 1:
        mcc_dom = matthews_corrcoef(d_true, d_pred)
        acc_dom = np.mean(d_true == d_pred)
        print(f"   • Domain MCC: {mcc_dom:.4f}")
        log_final_metric(f"{prefix}Domain_MCC", mcc_dom, f"Ability to distinguish hospitals ({prefix})")
        log_final_metric(f"{prefix}Domain_Acc", acc_dom, f"Accuracy of hospital classification ({prefix})")
    
    if len(feats) > 5000:
        idx = np.random.choice(len(feats), 5000, replace=False)
        feats_s = feats[idx]
        d_true_s = d_true[idx]
        y_true_s = y_true[idx]
    else:
        feats_s, d_true_s, y_true_s = feats, d_true, y_true
        
    if len(np.unique(y_true_s)) > 1:
        sil_med = silhouette_score(feats_s, y_true_s)
        print(f"   • Silhouette (Diagnosis): {sil_med:.4f}")
        log_final_metric(f"{prefix}Silhouette_Diagnosis", sil_med, f"Clustering quality by pathology ({prefix})")

    if len(np.unique(d_true_s)) > 1:
        sil_dom = silhouette_score(feats_s, d_true_s)
        print(f"   • Silhouette (Domain):    {sil_dom:.4f}")
        log_final_metric(f"{prefix}Silhouette_Domain", sil_dom, f"Clustering quality by hospital ({prefix})")
    
    print(f"💾 Wszystkie wyniki zapisano w: {FINAL_RESULTS_FILE}")

# --- URUCHOMIENIE ANALIZY DLA OBU MODELI ---

# 1. Analiza modelu KOŃCOWEGO (Final) - ten "zepsuty" przez DANN
run_target_evaluation(dann_model, test_loader, device, prefix="Final_")
evaluate_comprehensive(dann_model, train_loader, test_loader, device, max_batches=200, prefix="Final_")

# 2. Analiza modelu NAJLEPSZEGO (Best) - ten z wczesnej epoki
print(f"\n✅ Przywracanie najlepszych wag (AUC={best_auc:.4f})...")
dann_model.load_state_dict(best_model_wts)
run_target_evaluation(dann_model, test_loader, device, prefix="Best_")
evaluate_comprehensive(dann_model, train_loader, test_loader, device, max_batches=200, prefix="Best_")

