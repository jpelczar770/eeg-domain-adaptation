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
import glob
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

# Ścieżka do gotowych modeli DANN (skąd pobierzemy wagi do zamrożenia)
DANN_MODELS_DIR = "experiments_dann_new_labels"

# Nowy folder dla eksperymentów Probing
EXP_ROOT = "experiments_probing_new_labels"

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

folds, fold_override = read_folds_override(data_pth, None)
all_eids_raw = get_eids_for_folds(fold_override, [1, 2, 3, 4, 5, 6]) 
cleaned_all_eids_raw = {clean_eid_str(e) for e in all_eids_raw}

def map_to_binary(val):
    val = str(val).strip().lower()
    if val in ['0', '0.0', 'normal', 'norma', 'norm', 'false']: return 0
    elif val in ['unclassified', 'unknown', 'nan', 'none', '']: return None
    elif val in ['1', '1.0', 'pat', 'patho', 'epilepsy', 'other_patho', 'padaczka', 'inne', 'true']: return 1
    else: return 1 

print("\n KROK 1: Ładowanie fundamentu ze STAREGO pliku...")
df_old = pd.read_csv(old_csv_path, sep='|', low_memory=False)

for idx, row in df_old.iterrows():
    eid_clean = clean_eid_str(row['examination_id'])
    hosp = str(row['institution_id']).strip()
    
    if hosp in ALLOWED_HOSPITALS and eid_clean in cleaned_all_eids_raw:
        label_val = map_to_binary(row.get('classification_latest', ''))
        if label_val is not None:
            eid_to_hosp_name[eid_clean] = hosp
            uuid_to_label[eid_clean] = label_val

print("\n KROK 2: Nadpisywanie z NOWEGO pliku...")
if os.path.exists(new_csv_path):
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

all_hospitals = sorted(list(set(eid_to_hosp_name.values())))
hospital_to_id = {name: i for i, name in enumerate(all_hospitals)}
eid_to_hosp_id = {eid: hospital_to_id.get(hosp, 0) for eid, hosp in eid_to_hosp_name.items()}

def get_domain_labels(eids):
    labels = []
    flat_eids = eids if isinstance(eids, (list, tuple)) else [eids]
    for e in flat_eids: labels.append(eid_to_hosp_id.get(clean_eid_str(e), 0))
    return torch.tensor(labels, dtype=torch.long).to(device)

print(f"\n   📊 FINALNA BAZA DO TRENINGU: {len(uuid_to_label)} pacjentów z {len(all_hospitals)} szpitali.")

# ==========================================
# 4. KLASY LOADERA I MODELU (DANN do wczytania)
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
        self._loader_type = 'none' if override_non_mil else 'MIL'
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
            if eid_clean not in self.uuid_to_label: return (torch.zeros(10, 19, 600), torch.tensor(0), eid)
            cls = self.uuid_to_label[eid_clean]
            cls_torch = torch.tensor(int(cls), dtype=torch.int64)
            eid_key = str(eid) 
            
            if eid_key in self._data_file['metadata'].attrs:
                additional_metadata = json.loads(self._data_file['metadata'].attrs[eid_key])
            else: additional_metadata = {}

            if eid_key in self._data_file['features']:
                data_h5 = np.array(self._data_file['features'][eid_key])
            else: return (torch.zeros(10, 19, 600), cls_torch, eid)

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
        except Exception: return (torch.zeros(10, 19, 600), torch.tensor(0), eid)
    
    def get_batched_loader(self, batch_size, pad=True):
        pipe = self.construct_data_pipe(batch_size, pad)
        num_workers = self._num_workers if self._num_workers is not None else min([6, multiprocessing.cpu_count() - 1])
        return DataLoader2(pipe, reading_service=MultiProcessingReadingService(num_workers=num_workers))

# Musimy zainicjować pełnego DANN-a, żeby poprawnie wczytać wagi ze słownika 'best_model.pt'
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
            nn.Linear(feature_dim, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(64, num_domains)
        )
    def forward(self, x, time_stamps=None, alpha=1.0, part=False):
        out = self.backbone(x, time_stamps=time_stamps, part=True)
        if isinstance(out, tuple): features = out[0]
        else: features = out
        if features.dim() == 3: features = torch.mean(features, dim=1)
        if part: return features, None
        class_output = self.backbone.classifier(features)
        reverse_features = ReverseLayerF.apply(features, alpha)
        domain_output = self.domain_classifier(reverse_features)
        return class_output, domain_output

def evaluate_loss_and_auc(model, dataloader, device, loss_cls_fn):
    model.eval()
    total_loss, steps = 0.0, 0
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
    print(f" ROZPOCZYNAM PRZETWARZANIE PROBING (FROZEN DANN): {TARGET_HOSPITAL_CODE}")
    print("="*70)

    # 1. SZUKANIE WAG DANN DLA DANEGO SZPITALA
    dann_model_path = None
    if os.path.exists(DANN_MODELS_DIR):
        # Szukamy folderu z wynikami dla tego szpitala
        for d_folder in os.listdir(DANN_MODELS_DIR):
            if d_folder.startswith(TARGET_HOSPITAL_CODE + "_"):
                potential_path = os.path.join(DANN_MODELS_DIR, d_folder, "best_model.pt")
                if os.path.exists(potential_path):
                    dann_model_path = potential_path
                    break

    if dann_model_path is None:
        print(f" ❌ UWAGA: Nie znaleziono best_model.pt dla {TARGET_HOSPITAL_CODE} w folderze {DANN_MODELS_DIR}!")
        print(" Pomijam ten szpital.")
        continue

    # 2. PRZYGOTOWANIE KATALOGU EKSPERYMENTU
    EXP_DIR = f"{EXP_ROOT}/{TARGET_HOSPITAL_CODE}_{TIMESTAMP}"
    os.makedirs(EXP_DIR, exist_ok=True)

    TRAIN_LOG_FILE = os.path.join(EXP_DIR, "training_log.csv")
    FINAL_RESULTS_FILE = os.path.join(EXP_DIR, "final_results.csv")
    MODEL_BEST_PATH = os.path.join(EXP_DIR, "best_model.pt")
    MODEL_FINAL_PATH = os.path.join(EXP_DIR, "final_model.pt")

    with open(TRAIN_LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss', 'Val_AUC', 'Test_Loss', 'Test_AUC'])

    def log_final_metric(name, value, description=""):
        with open(FINAL_RESULTS_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, value, description])

    with open(FINAL_RESULTS_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric_Name', 'Value', 'Description'])

    # 3. PRZYGOTOWANIE PODZIAŁU: TRAIN, VAL, TEST
    test_pool = []
    source_hospitals_eids = {h: [] for h in ALLOWED_HOSPITALS if h != TARGET_HOSPITAL_CODE}

    for eid in all_eids_raw:
        eid_clean = clean_eid_str(eid)
        if eid_clean in uuid_to_label:  
            h_name = eid_to_hosp_name.get(eid_clean, "Unknown")
            if h_name == TARGET_HOSPITAL_CODE: test_pool.append(eid)
            elif h_name in source_hospitals_eids: source_hospitals_eids[h_name].append(eid)

    train_pool, val_pool = [], []
    for h_name, eids_list in source_hospitals_eids.items():
        random.shuffle(eids_list) 
        n_val = min(100, len(eids_list))
        val_pool.extend(eids_list[:n_val])
        train_pool.extend(eids_list[n_val:])

    print(f" PODZIAŁ DANYCH:")
    print(f"   TRAIN:      {len(train_pool)} pacjentów")
    print(f"   VALIDATION: {len(val_pool)} pacjentów (źródłowe)")
    print(f"   TEST:       {len(test_pool)} pacjentów (docelowy)")

    train_loader = Loader(data_pth, train_pool, uuid_to_label, minet_subsampling_n=4, num_workers=4).get_batched_loader(32, pad=True)
    val_loader = Loader(data_pth, val_pool, uuid_to_label, minet_subsampling_n=4, num_workers=2).get_batched_loader(32, pad=True)
    test_loader_epoch = Loader(data_pth, test_pool, uuid_to_label, minet_subsampling_n=4, num_workers=2).get_batched_loader(32, pad=True)
    test_loader_final = Loader(data_pth, test_pool, uuid_to_label, minet_subsampling_n=None, num_workers=0).get_batched_loader(1, pad=True)

    # 4. INICJALIZACJA MODELU I ZAMRAŻANIE (FROZEN BACKBONE)
    print(f" Ładowanie i zamrażanie wag z: {dann_model_path}...")
    raw_backbone = torch.load(model_pth, map_location=device)
    if hasattr(raw_backbone, 'n_chans'): raw_backbone.n_chans = 19

    probing_model = MinetDANN(raw_backbone, feature_dim=288, num_domains=len(all_hospitals)).to(device)
    
    # Wczytujemy zeszłą wiedzę DANN
    probing_model.load_state_dict(torch.load(dann_model_path, map_location=device))

    # ZAMRAŻAMY CAŁY MODEL
    for param in probing_model.parameters():
        param.requires_grad = False
        
    # ODMRAŻAMY TYLKO KLASYFIKATOR KOŃCOWY
    for param in probing_model.backbone.classifier.parameters():
        param.requires_grad = True

    # Optymalizator operuje tylko na odblokowanych wagach, z mniejszym LR
    optimizer = optim.AdamW(probing_model.backbone.classifier.parameters(), lr=1e-4, weight_decay=1e-4)

    loss_cls_fn = nn.BCEWithLogitsLoss()
    scaler = GradScaler() 

    EPOCHS = 50 
    STEPS = 500   

    best_val_auc = 0.0
    best_model_wts = copy.deepcopy(probing_model.state_dict())

    # 5. PĘTLA TRENINGOWA
    print(f"\n START TRENINGU PROBING (Sztywno {EPOCHS} epok, LR=1e-4)...")
    for epoch in range(EPOCHS):
        probing_model.train()
        # Ważne: Wymuszamy, by zamrożone warstwy BatchNorm działały w trybie ewaluacji, 
        # żeby nie niszczyć statystyk, ale klasyfikator ma się uczyć.
        probing_model.eval() 
        probing_model.backbone.classifier.train()

        iter_train = iter(train_loader)
        running_cls_loss = 0.0
        pbar = tqdm(range(STEPS), desc=f"Epoka {epoch+1}/{EPOCHS}", leave=False)
        
        for i in pbar:
            try: batch = next(iter_train)
            except StopIteration:
                iter_train = iter(train_loader)
                batch = next(iter_train)
                
            if len(batch) == 3: X, y, eid = batch; ts = None
            else: X, y, eid = batch[0], batch[1], batch[2]; ts = None
            
            X = X.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32).view(-1, 1)
            ts = torch.zeros((X.shape[0], X.shape[1]), dtype=torch.long, device=device) if ts is None else ts.to(device)
            
            optimizer.zero_grad()

            with autocast():
                # Używamy DANN z alpha=0.0 (ignorujemy loss domeny, chcemy tylko c_pred)
                c_pred, _ = probing_model(X, time_stamps=ts, alpha=0.0)
                loss = loss_cls_fn(c_pred, y)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer) 
            torch.nn.utils.clip_grad_norm_(probing_model.backbone.classifier.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            running_cls_loss += loss.item()
            pbar.set_postfix({'Loss': f"{loss.item():.2f}"})

        avg_train_loss = running_cls_loss / STEPS

        val_loss, val_auc = evaluate_loss_and_auc(probing_model, val_loader, device, loss_cls_fn)
        test_loss, test_auc = evaluate_loss_and_auc(probing_model, test_loader_epoch, device, loss_cls_fn)

        print(f" Epoka {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Val AUC: {val_auc:.4f} | Test AUC: {test_auc:.4f}")

        with open(TRAIN_LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, avg_train_loss, val_loss, val_auc, test_loss, test_auc])

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_wts = copy.deepcopy(probing_model.state_dict())
            torch.save(probing_model.state_dict(), MODEL_BEST_PATH) 

    print("\n Trening zakończony. Zapis modelu Final...")
    torch.save(probing_model.state_dict(), MODEL_FINAL_PATH)

    # 6. EWALUACJA KOŃCOWA I ZAPIS WYNIKÓW
    def run_target_evaluation(model, loader, device, prefix=""):
        print(f"\n EWALUACJA DIAGNOSTYCZNA: {prefix} (Target: {TARGET_HOSPITAL_CODE})")
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

    # 1. Final Model
    run_target_evaluation(probing_model, test_loader_final, device, prefix="Final_")
    
    # 2. Best Model
    print(f"\n Przywracanie wag BEST (Val_AUC={best_val_auc:.4f})...")
    probing_model.load_state_dict(best_model_wts)
    run_target_evaluation(probing_model, test_loader_final, device, prefix="Best_")

print("\n ZAKOŃCZONO PĘTLĘ LINEAR PROBING")