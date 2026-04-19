# ==========================================
# 1. Biblioteki Standardowe i Konfiguracja
# ==========================================
import os
import csv
import json
import datetime
import multiprocessing
import random
import copy 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5pickle
from tqdm import tqdm

from sklearn.metrics import (
    roc_curve, roc_auc_score, 
    accuracy_score, matthews_corrcoef,
    silhouette_score
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Function
from torch.cuda.amp import GradScaler, autocast
import torchdata.datapipes as dp
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService

# Obejście dla File Descriptor Limit w multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from loader_utils import (
    stratified_sample, 
    read_folds_override, 
    get_eids_for_folds
)

# ==========================================
# ŚCIEŻKI
# ==========================================
data_pth = "/dmj/fizmed/mpoziomska/ELMIKO/neuroscreening-fuw/data/elmiko/processed_all_MIL_800"
model_pth = "/dmj/fizmed/jpelczar/od_martyny/minet/models/minet_raw_fold_6"

new_csv_path = '/dmj/fizmed/jpelczar/od_martyny/minet/base_92_094_pred_NEW2.csv'
old_csv_path = 'used_label_database.csv'

# Używamy nowej nazwy folderu, aby odróżnić eksperymenty z nowymi labelami
EXP_ROOT = "experiments_multitask_naive_new_labels"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(EXP_ROOT, exist_ok=True)

# ==========================================
# 2. GLOBALNA FUNKCJA CZYSZCZĄCA EID
# ==========================================
def clean_eid_str(eid_raw):
    if isinstance(eid_raw, (list, tuple, np.ndarray)): 
        if len(eid_raw) > 0: eid_raw = eid_raw[0]
    if isinstance(eid_raw, bytes): 
        eid_raw = eid_raw.decode('utf-8')
    
    s = str(eid_raw).strip()
    if "{" in s and "}" in s:
        start = s.find("{")
        end = s.find("}")
        return s[start:end+1].lower()
    return s.lower()

# ==========================================
# 3. BEZPOŚREDNIE MAPOWANIE: STARY -> NOWY
# ==========================================
ALLOWED_HOSPITALS = [
    "ZOZLO", "KATMOJPRZ", "SZC", "TOR", "OST", 
    "LUMICE", "CMD", "SRK", "AKS", "PRZ", 
    "KIEG", "OTW", "MKW", "PUS", "LUX_A", 
    "KUD", "ARCHDAM", "MOR", "KAL", "B2K", 
    "SLU", "SL2", "STG1", "CHE", "KLU", 
    "GAK", "WLU", "Z04O", "TER_L", "PIO"
]

uuid_to_label = {}
eid_to_hosp_name = {}

# Pobieranie pacjentów fizycznie obecnych w plikach HDF5
folds, fold_override = read_folds_override(data_pth, None)
all_eids_raw = get_eids_for_folds(fold_override, [1, 2, 3, 4, 5, 6]) 
cleaned_all_eids_raw = {clean_eid_str(e) for e in all_eids_raw}

def map_to_binary(val):
    val = str(val).strip().lower()
    if val in ['0', '0.0', 'normal', 'norma', 'norm', 'false']:
        return 0
    elif val in ['unclassified', 'unknown', 'nan', 'none', '']:
        return None
    elif val in ['1', '1.0', 'pat', 'patho', 'epilepsy', 'other_patho', 'padaczka', 'inne', 'true']:
        return 1
    else:
        return 1 # Fallback 

print("\n KROK 1: Ładowanie fundamentu ze STAREGO pliku (classification_latest)...")
df_old = pd.read_csv(old_csv_path, sep='|', low_memory=False)

old_labels_applied = 0
for idx, row in df_old.iterrows():
    eid_clean = clean_eid_str(row['examination_id'])
    hosp = str(row['institution_id']).strip()
    
    if hosp in ALLOWED_HOSPITALS and eid_clean in cleaned_all_eids_raw:
        label_val = map_to_binary(row.get('classification_latest', ''))
        if label_val is not None:
            eid_to_hosp_name[eid_clean] = hosp
            uuid_to_label[eid_clean] = label_val
            old_labels_applied += 1

print(f"   -> Wczytano bazę {old_labels_applied} pacjentów ze starymi etykietami.")

print("\n KROK 2: Nadpisywanie z NOWEGO pliku (omijanie zepsutych kolumn)...")
if os.path.exists(new_csv_path):
    new_labels_applied = 0
    
    with open(new_csv_path, 'r', encoding='iso-8859-2', errors='ignore') as f:
        header = f.readline().strip().split('|')
        if 'CLAS_SVC_1B' in header:
            dist_from_right = len(header) - header.index('CLAS_SVC_1B')
            for line in f:
                parts = line.strip().split('|')
                if len(parts) < 10: continue
                eid_clean = clean_eid_str(parts[0])
                if eid_clean in uuid_to_label:
                    try: new_clas_val = parts[-dist_from_right]
                    except IndexError: continue
                        
                    binary_val = map_to_binary(new_clas_val)
                    if binary_val is not None:
                        uuid_to_label[eid_clean] = binary_val
                        new_labels_applied += 1
            print(f"   -> UDAŁO SIĘ! Aplikowano i nadpisano {new_labels_applied} etykiet z nowej tabeli.")
else:
    print(f"   ⚠️ Nie znaleziono pliku {new_csv_path}. Lecimy tylko na starych.")

all_hospitals = sorted(list(set(eid_to_hosp_name.values())))
hospital_to_id = {name: i for i, name in enumerate(all_hospitals)}

eid_to_hosp_id = {}
for eid, hosp in eid_to_hosp_name.items():
    eid_to_hosp_id[eid] = hospital_to_id.get(hosp, 0)

def get_domain_labels(eids):
    labels = []
    flat_eids = eids if isinstance(eids, (list, tuple)) else [eids]
    for e in flat_eids:
        e_clean = clean_eid_str(e)
        labels.append(eid_to_hosp_id.get(e_clean, 0))
    return torch.tensor(labels, dtype=torch.long).to(device)

print(f"\n   📊 FINALNA BAZA DO TRENINGU: {len(uuid_to_label)} pacjentów z {len(all_hospitals)} szpitali.")

# ==========================================
# 4. KLASY LOADERA I MODELU (MULTI-TASK NAIVE)
# ==========================================
def collate_pad(batch):
    X, y, eid = zip(*batch)
    y = torch.tensor(y)
    X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True)
    return [X, y, eid]

class Loader:
    def __init__(self, data_path, eids, uuid_to_label_dict, override_non_mil=False, minet_subsampling_n=None, num_workers=None):
        self._num_workers = num_workers
        self.minet_subsampling_n = minet_subsampling_n
        self._data_file = h5pickle.File(os.path.join(data_path, 'features', 'data.hdf5'), 'r')
        self.uuid_to_label = uuid_to_label_dict  
        
        if override_non_mil: self._loader_type = 'none'
        else: self._loader_type = 'MIL'
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
            eid_clean = clean_eid_str(eid)
            if eid_clean not in self.uuid_to_label:
                return (torch.zeros(10, 19, 600), torch.tensor(0), eid)

            cls = self.uuid_to_label[eid_clean]
            cls_torch = torch.tensor(int(cls), dtype=torch.int64)
            
            eid_key = str(eid) 
            if eid_key in self._data_file['metadata'].attrs:
                meta_str = self._data_file['metadata'].attrs[eid_key]
                additional_metadata = json.loads(meta_str)
            else: additional_metadata = {}

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
            
            if self._loader_type == "none": return [(data[i], cls_torch, eid) for i in range(len(data))]
            else: return (data, cls_torch, eid)
        except Exception:
             return (torch.zeros(10, 19, 600), torch.tensor(0), eid)
    
    def get_batched_loader(self, batch_size, pad=True):
        pipe = self.construct_data_pipe(batch_size, pad)
        num_workers = self._num_workers if self._num_workers is not None else min([6, multiprocessing.cpu_count() - 1])
        mp_rs = MultiProcessingReadingService(num_workers=num_workers)
        return DataLoader2(pipe, reading_service=mp_rs)

class NormalLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # NORMALNA PROPAGACJA - brak odwracania gradientu (Multi-task)
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
# FUNKCJA POMOCNICZA DO SZYBKIEJ EWALUACJI LOSS/AUC
# ==========================================
def evaluate_loss_and_auc(model, dataloader, device, loss_cls_fn):
    model.eval()
    total_loss = 0.0
    steps = 0
    y_true, y_prob = [], []
    
    with torch.no_grad():
        for data in dataloader:
            if len(data) == 3: X, y, _ = data; ts = None
            else: X, y, _, ts = data
            if X.shape[1] > 300: X = X[:, :300]
            
            X = X.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32).view(-1, 1)
            ts = torch.zeros((X.shape[0], X.shape[1]), dtype=torch.long, device=device) if ts is None else ts.to(device)

            c_pred, _ = model(X, time_stamps=ts, alpha=0.0)
            loss = loss_cls_fn(c_pred, y)
            total_loss += loss.item()
            steps += 1

            probs = torch.sigmoid(c_pred)
            y_true.extend(y.cpu().numpy().flatten())
            y_prob.extend(probs.cpu().numpy().flatten())

    avg_loss = total_loss / steps if steps > 0 else 0.0
    try: auc = roc_auc_score(y_true, y_prob)
    except: auc = 0.5
    return avg_loss, auc

# ==========================================
# PĘTLA SZPITALI (TARGETS)
# ==========================================
MY_TARGET_HOSPITALS = [
    "ZOZLO", "KATMOJPRZ", "SZC", "TOR", "OST", 
    "LUMICE", "CMD", "SRK", "AKS", "PRZ", 
    "KIEG", "OTW", "MKW", "PUS", "LUX_A"
]

completed_hospitals = set()
for folder_name in os.listdir(EXP_ROOT):
    if os.path.exists(os.path.join(EXP_ROOT, folder_name, "final_results.csv")):
        hosp = folder_name.split('_')[0]
        if folder_name.startswith("LUX_A"): hosp = "LUX_A"
        elif folder_name.startswith("TER_L"): hosp = "TER_L"
        completed_hospitals.add(hosp)

MY_TARGET_HOSPITALS = [h for h in MY_TARGET_HOSPITALS if h not in completed_hospitals]
print(f"\n Do policzenia pozostało: {MY_TARGET_HOSPITALS}")

for TARGET_HOSPITAL_CODE in MY_TARGET_HOSPITALS:
    print("\n" + "="*70)
    print(f" ROZPOCZYNAM PRZETWARZANIE DLA SZPITALA (NAIVE MULTITASK): {TARGET_HOSPITAL_CODE}")
    print("="*70)

    # 1. PRZYGOTOWANIE KATALOGU EKSPERYMENTU
    EXP_DIR = f"{EXP_ROOT}/{TARGET_HOSPITAL_CODE}_{TIMESTAMP}"
    os.makedirs(EXP_DIR, exist_ok=True)

    TRAIN_LOG_FILE = os.path.join(EXP_DIR, "training_log.csv")
    FINAL_RESULTS_FILE = os.path.join(EXP_DIR, "final_results.csv")
    MODEL_BEST_PATH = os.path.join(EXP_DIR, "best_model.pt")
    MODEL_FINAL_PATH = os.path.join(EXP_DIR, "final_model.pt")

    with open(TRAIN_LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_Total_Loss', 'Train_Cls_Loss', 'Train_Dom_Loss', 'Train_Dom_Acc', 'Val_Cls_Loss', 'Val_AUC', 'Test_Cls_Loss', 'Test_AUC', 'Alpha'])

    def log_final_metric(name, value, description=""):
        with open(FINAL_RESULTS_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, value, description])

    with open(FINAL_RESULTS_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric_Name', 'Value', 'Description'])

    # 2. PRZYGOTOWANIE PODZIAŁU: TRAIN, VAL, TEST
    test_pool = []
    source_hospitals_eids = {h: [] for h in ALLOWED_HOSPITALS if h != TARGET_HOSPITAL_CODE}

    for eid in all_eids_raw:
        eid_clean = clean_eid_str(eid)
        if eid_clean in uuid_to_label:  
            h_name = eid_to_hosp_name.get(eid_clean, "Unknown")
            if h_name == TARGET_HOSPITAL_CODE:
                test_pool.append(eid)
            elif h_name in source_hospitals_eids:
                source_hospitals_eids[h_name].append(eid)

    train_pool = []
    val_pool = []

    for h_name, eids_list in source_hospitals_eids.items():
        random.shuffle(eids_list) 
        n_val = min(100, len(eids_list))
        val_pool.extend(eids_list[:n_val])
        train_pool.extend(eids_list[n_val:])

    print(f" PODZIAŁ DANYCH:")
    print(f"   TRAIN:      {len(train_pool)} pacjentów")
    print(f"   VALIDATION: {len(val_pool)} pacjentów (źródłowe)")
    print(f"   TEST:       {len(test_pool)} pacjentów (docelowy)")

    if len(test_pool) == 0:
        print(f" Brak pacjentów testowych. Pomijanie...")
        continue

    train_loader = Loader(data_pth, train_pool, uuid_to_label, minet_subsampling_n=4, num_workers=4).get_batched_loader(32, pad=True)
    val_loader = Loader(data_pth, val_pool, uuid_to_label, minet_subsampling_n=4, num_workers=2).get_batched_loader(32, pad=True)
    test_loader_epoch = Loader(data_pth, test_pool, uuid_to_label, minet_subsampling_n=4, num_workers=2).get_batched_loader(32, pad=True)
    test_loader_final = Loader(data_pth, test_pool, uuid_to_label, minet_subsampling_n=None, num_workers=0).get_batched_loader(1, pad=True)

    # 3. INICJALIZACJA MODELU (OD ZERA)
    def weight_reset(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear): m.reset_parameters()
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d): m.reset_parameters()

    raw_backbone = torch.load(model_pth, map_location=device)
    raw_backbone.apply(weight_reset)
    if hasattr(raw_backbone, 'n_chans'): raw_backbone.n_chans = 19

    multitask_model = MinetMultiTask(raw_backbone, feature_dim=288, num_domains=len(all_hospitals)).to(device)

    backbone_head_params = list(multitask_model.backbone.classifier.parameters())
    head_ids = list(map(id, backbone_head_params))
    backbone_base_params = filter(lambda p: id(p) not in head_ids, multitask_model.backbone.parameters())

    optimizer = optim.AdamW([
        {'params': backbone_base_params, 'lr': 1e-3}, 
        {'params': backbone_head_params, 'lr': 1e-3},
        {'params': multitask_model.domain_classifier.parameters(), 'lr': 1e-3}
    ], weight_decay=1e-4)

    loss_cls_fn = nn.BCEWithLogitsLoss()
    loss_dom_fn = nn.CrossEntropyLoss()
    scaler = GradScaler() 

    EPOCHS = 50 
    STEPS = 500
    WARMUP_EPOCHS = 10 
    MAX_ALPHA = 1.0    

    best_val_auc = 0.0
    best_model_wts = copy.deepcopy(multitask_model.state_dict())

    # 4. PĘTLA TRENINGOWA
    print(f"\n START TRENINGU (Sztywno {EPOCHS} epok, model selection on Source Validation)...")
    for epoch in range(EPOCHS):
        multitask_model.train()
        iter_train = iter(train_loader)
        metrics = {'total': 0, 'cls': 0, 'dom': 0, 'acc_dom': 0}
        
        pbar = tqdm(range(STEPS), desc=f"Epoka {epoch+1}/{EPOCHS}", leave=False)
        
        for i in pbar:
            if epoch < WARMUP_EPOCHS: alpha = 0.0
            else:
                total_phase2_steps = (EPOCHS - WARMUP_EPOCHS) * STEPS
                current_phase2_step = (i + ((epoch - WARMUP_EPOCHS) * STEPS))
                ratio = current_phase2_step / total_phase2_steps
                alpha = min(ratio * MAX_ALPHA, MAX_ALPHA)
            
            try: batch = next(iter_train)
            except StopIteration:
                iter_train = iter(train_loader)
                batch = next(iter_train)
                
            if len(batch) == 3: X, y, eid = batch; ts = None
            else: X, y, eid = batch[0], batch[1], batch[2]; ts = None
            
            X = X.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32).view(-1, 1)
            ts = torch.zeros((X.shape[0], X.shape[1]), dtype=torch.long, device=device) if ts is None else ts.to(device)

            dom_labels = get_domain_labels(eid)
            
            optimizer.zero_grad()

            with autocast():
                c_pred, d_pred = multitask_model(X, time_stamps=ts, alpha=alpha)
                loss_cls = loss_cls_fn(c_pred, y)
                loss_dom = loss_dom_fn(d_pred, dom_labels)
                loss = (5.0 * loss_cls) + loss_dom 
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer) 
            torch.nn.utils.clip_grad_norm_(multitask_model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            preds_dom_batch = torch.argmax(d_pred, dim=1)
            acc_dom = (preds_dom_batch == dom_labels).float().mean().item()
            metrics['total'] += loss.item()
            metrics['cls'] += loss_cls.item()
            metrics['dom'] += loss_dom.item()
            metrics['acc_dom'] += acc_dom
            
            pbar.set_postfix({'Total': f"{loss.item():.2f}", 'Cls': f"{loss_cls.item():.2f}", 'Dom': f"{loss_dom.item():.2f}"})

        avg_train_total = metrics['total'] / STEPS
        avg_train_cls = metrics['cls'] / STEPS
        avg_train_dom = metrics['dom'] / STEPS
        avg_train_dom_acc = metrics['acc_dom'] / STEPS

        val_loss, val_auc = evaluate_loss_and_auc(multitask_model, val_loader, device, loss_cls_fn)
        test_loss, test_auc = evaluate_loss_and_auc(multitask_model, test_loader_epoch, device, loss_cls_fn)

        print(f" Epoka {epoch+1:02d} | Train Total: {avg_train_total:.4f} | Cls: {avg_train_cls:.4f} | Dom: {avg_train_dom:.4f} | Val AUC: {val_auc:.4f} | Test AUC: {test_auc:.4f}")

        with open(TRAIN_LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, avg_train_total, avg_train_cls, avg_train_dom, avg_train_dom_acc, val_loss, val_auc, test_loss, test_auc, alpha])

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_wts = copy.deepcopy(multitask_model.state_dict())
            torch.save(multitask_model.state_dict(), MODEL_BEST_PATH) 

    print("\n Trening zakończony. Zapis modelu Final...")
    torch.save(multitask_model.state_dict(), MODEL_FINAL_PATH)

    # 5. EWALUACJA KOŃCOWA I ZAPIS WYNIKÓW
    def run_target_evaluation(model, loader, device, prefix=""):
        print(f"\n EWALUACJA: {prefix} (Target: {TARGET_HOSPITAL_CODE})")
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

        y_true, y_prob = np.array(y_true), np.array(y_prob)

        if len(np.unique(y_true)) > 1:
            auc_val = roc_auc_score(y_true, y_prob)
            fpr, tpr, thresh = roc_curve(y_true, y_prob)
            best_t = thresh[np.argmax(tpr - fpr)]
            y_pred = (y_prob >= best_t).astype(int)
            mcc_val = matthews_corrcoef(y_true, y_pred)
            acc_val = accuracy_score(y_true, y_pred)
            
            print(f"    {prefix}Results: AUC={auc_val:.4f} | MCC={mcc_val:.4f} | Acc={acc_val:.4f}")
            log_final_metric(f"{prefix}Target_Diagnosis_AUC", auc_val, "Diag AUC on Target Hospital")
            log_final_metric(f"{prefix}Target_Diagnosis_MCC", mcc_val, "Diag MCC on Target Hospital")
            log_final_metric(f"{prefix}Target_Diagnosis_Acc", acc_val, "Diag Accuracy on Target Hospital")
        else:
            print("    Not enough classes.")

    def evaluate_comprehensive(model, source_loader, target_loader, device, max_batches=100, prefix=""):
        print(f"\n ROZPOCZYNAMY KOMPLETNĄ ANALIZĘ {prefix} (Silhouette + Domain Metrics)...")
        model.eval()
        feats_list = []
        y_true, y_prob, d_true, d_pred = [], [], [], []
        
        def collect_from_loader(loader):
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
                    ts = torch.zeros((X.shape[0], X.shape[-1]), dtype=torch.long, device=device) if ts is None else ts.to(device)

                    f, _ = model.backbone(X, time_stamps=ts, part=True)
                    if len(f.shape) == 3: f = torch.mean(f, dim=1)
                    
                    class_logits, domain_logits = model(X, time_stamps=ts, alpha=0.0)
                    probs = torch.sigmoid(class_logits).view(-1) if class_logits.shape[1] == 1 else torch.softmax(class_logits, dim=1)[:, 1]
                    d_p = torch.argmax(domain_logits, dim=1).cpu().numpy()
                    
                    try: d_t = get_domain_labels(eid).cpu().numpy()
                    except: d_t = np.zeros(len(probs))
                    
                    feats_list.append(f.cpu().numpy())
                    y_np = y.cpu().numpy() if isinstance(y, torch.Tensor) else np.array(y)
                    if y_np.ndim == 0: y_np = np.expand_dims(y_np, 0)
                    y_true.extend(y_np)
                    y_prob.extend(probs.cpu().numpy())
                    d_true.extend(d_t)
                    d_pred.extend(d_p)

        collect_from_loader(train_loader)
        collect_from_loader(test_loader_final)
        
        feats = np.concatenate(feats_list, axis=0)
        y_true, d_true, d_pred = np.array(y_true).astype(int), np.array(d_true).astype(int), np.array(d_pred).astype(int)
        
        if len(np.unique(d_true)) > 1:
            mcc_dom = matthews_corrcoef(d_true, d_pred)
            log_final_metric(f"{prefix}Domain_MCC", mcc_dom, f"Ability to distinguish hospitals ({prefix})")
            print(f"   • Domain MCC: {mcc_dom:.4f}")
        
        if len(feats) > 5000:
            idx = np.random.choice(len(feats), 5000, replace=False)
            feats, d_true, y_true = feats[idx], d_true[idx], y_true[idx]
            
        if len(np.unique(y_true)) > 1:
            sil_med = silhouette_score(feats, y_true)
            log_final_metric(f"{prefix}Silhouette_Diagnosis", sil_med, f"Clustering quality by pathology ({prefix})")

        if len(np.unique(d_true)) > 1:
            sil_dom = silhouette_score(feats, d_true)
            log_final_metric(f"{prefix}Silhouette_Domain", sil_dom, f"Clustering quality by hospital ({prefix})")

    run_target_evaluation(multitask_model, test_loader_final, device, prefix="Final_")
    evaluate_comprehensive(multitask_model, train_loader, test_loader_final, device, max_batches=200, prefix="Final_")

    print(f"\n Przywracanie wag BEST (Val_AUC={best_val_auc:.4f})...")
    multitask_model.load_state_dict(best_model_wts)
    run_target_evaluation(multitask_model, test_loader_final, device, prefix="Best_")
    evaluate_comprehensive(multitask_model, train_loader, test_loader_final, device, max_batches=200, prefix="Best_")

print("\n ZAKOŃCZONO PĘTLĘ NAIVE MULTI-TASK")