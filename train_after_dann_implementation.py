# ==========================================
# 1. Biblioteki Standardowe i Konfiguracja
# ==========================================
import os
import csv
import json
import glob
import copy 
import numpy as np
import pandas as pd
import multiprocessing
import matplotlib.pyplot as plt
import h5pickle
from tqdm import tqdm

# ==========================================
# 2. Scikit-Learn i PyTorch
# ==========================================
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, matthews_corrcoef
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Function
import torchdata.datapipes as dp
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService

# ==========================================
# 3. Lokalne Narzędzia
# ==========================================
from loader_utils import (
    label_override, load_label_override, load_metadata_database, 
    stratified_sample, read_folds_override, get_eids_for_folds
)

# Ścieżki bazowe
data_pth = "/dmj/fizmed/mpoziomska/ELMIKO/neuroscreening-fuw/data/elmiko/processed_all_MIL_800"
model_pth = "/dmj/fizmed/jpelczar/od_martyny/minet/models/minet_raw_fold_6"
csv_path = 'used_label_database.csv'
base_experiments_dir = "experiments" # Folder z oryginalnymi eksperymentami DANN

# WYMUSZENIE UŻYCIA CPU
device = torch.device("cpu")
print(f"🔧 Using device: {device} (Forced CPU execution)")

# ==========================================
# 4. Definicje Klas i Funkcji (Loader, Model)
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
        self._loader_type = 'none' if override_non_mil else 'MIL'
        self._data_path = data_path
        self._eids = eids

    def construct_data_pipe(self, batch_size, pad):
        pipe = dp.map.SequenceWrapper(self._eids)
        pipe = pipe.shuffle().sharding_filter().map(self.loader_mapping_func)
        if self._loader_type == "none": pipe = pipe.unbatch()
        pipe = pipe.batch(batch_size=batch_size, drop_last=True)
        if self._loader_type == 'MIL' and pad: pipe = pipe.collate(collate_pad)
        else: pipe = pipe.collate()
        return pipe

    def loader_mapping_func(self, eid):
        try:
            cls = 1 - label_override([eid], self.labels_database)[0]
            eid_key = str(eid)
            additional_metadata = json.loads(self._data_file['metadata'].attrs[eid_key]) if eid_key in self._data_file['metadata'].attrs else {}
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
            if self._loader_type == "none": return [(data[i], cls_torch, eid) for i in range(len(data))]
            else: return (data, cls_torch, eid)
        except Exception:
             return (torch.zeros(10, 19, 600), torch.tensor(0), eid)
    
    def get_batched_loader(self, batch_size, pad=True):
        pipe = self.construct_data_pipe(batch_size, pad)
        num_workers = self._num_workers if self._num_workers is not None else min([6, multiprocessing.cpu_count() - 1])
        mp_rs = MultiProcessingReadingService(num_workers=num_workers)
        return DataLoader2(pipe, reading_service=mp_rs)

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class MinetDANN(nn.Module):
    def __init__(self, original_model, feature_dim=288, num_domains=39):
        super(MinetDANN, self).__init__()
        self.backbone = original_model
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, num_domains)
        )

    def forward(self, x, time_stamps=None, alpha=1.0, part=False):
        out = self.backbone(x, time_stamps=time_stamps, part=True)
        features = out[0] if isinstance(out, tuple) else out
        if features.dim() == 3: features = torch.mean(features, dim=1)
        if part: return features, None
        class_output = self.backbone.classifier(features)
        reverse_features = ReverseLayerF.apply(features, alpha)
        domain_output = self.domain_classifier(reverse_features)
        return class_output, domain_output

# ==========================================
# 5. PRZYGOTOWANIE METADANYCH
# ==========================================
print("📂 Ładowanie metadanych...")
df_meta = pd.read_csv(csv_path, sep='|', low_memory=False)
df_meta['examination_id'] = df_meta['examination_id'].astype(str).str.strip()
df_meta['institution_id'] = df_meta['institution_id'].astype(str).str.strip()

ALLOWED_HOSPITALS = [
    "ZOZLO", "KATMOJPRZ", "SZC", "TOR", "OST", "LUMICE", "CMD", "SRK", "AKS", "PRZ", 
    "KIEG", "OTW", "MKW", "PUS", "LUX_A", "KUD", "ARCHDAM", "MOR", "KAL", "B2K", 
    "SLU", "SL2", "STG1", "CHE", "KLU", "GAK", "WLU", "Z04O", "TER_L", "PIO"
]
df_meta = df_meta[df_meta['institution_id'].isin(ALLOWED_HOSPITALS)].copy()
all_hospitals = sorted(df_meta['institution_id'].unique().tolist())
eid_to_hosp_name = pd.Series(df_meta.institution_id.values, index=df_meta.examination_id).to_dict()

def clean_eid_str(eid_raw):
    if isinstance(eid_raw, (list, tuple, np.ndarray)) and len(eid_raw) > 0: eid_raw = eid_raw[0]
    if isinstance(eid_raw, bytes): return eid_raw.decode('utf-8')
    return str(eid_raw).strip()

# ==========================================
# 6. PĘTLA PO SZPITALACH I FOLDERACH
# ==========================================
# Użyj swojej listy celów (1/2 lub 2/2)
MY_TARGET_HOSPITALS = [
    "ZOZLO", "KATMOJPRZ", "SZC", "TOR", "OST", 
    "LUMICE", "CMD", "SRK", "AKS", "PRZ", 
    "KIEG", "OTW", "MKW", "PUS", "LUX_A"
]

print("\n🔍 Sprawdzanie zakończonych eksperymentów Probing w DANN...")

for TARGET_HOSPITAL_CODE in MY_TARGET_HOSPITALS:
    print("\n" + "="*60)
    print(f"🏥 SZUKANIE WYNIKÓW DLA SZPITALA: {TARGET_HOSPITAL_CODE}")
    print("="*60)

    # Znajdź najnowszy folder dla tego szpitala
    folder_pattern = os.path.join(base_experiments_dir, f"{TARGET_HOSPITAL_CODE}_*")
    matching_folders = sorted(glob.glob(folder_pattern))
    
    if not matching_folders:
        print(f"⚠️ Nie znaleziono żadnego folderu eksperymentu dla {TARGET_HOSPITAL_CODE}. Pomijam.")
        continue
        
    target_dir = matching_folders[-1] # Wybieramy najnowszy folder
    model_weights_path = os.path.join(target_dir, "best_model.pt")
    log_file_path = os.path.join(target_dir, "training_log.csv")
    
    # --- NOWE PLIKI DLA PROBINGU (W TYM SAMYM FOLDERZE) ---
    final_results_file = os.path.join(target_dir, "final_results_frozen_alpha.csv")
    probing_log_file = os.path.join(target_dir, "train_log_alpha_frozen.csv")
    
    # CHECK WHAT'S DONE
    if os.path.exists(final_results_file):
        print(f"⏩ Probing już wykonany dla {TARGET_HOSPITAL_CODE} (znaleziono plik wyników). Pomijam.")
        continue
        
    if not os.path.exists(model_weights_path) or not os.path.exists(log_file_path):
        print(f"⚠️ W folderze {target_dir} brakuje pliku best_model.pt lub starych logów. Pomijam.")
        continue

    print(f"📁 Znaleziono folder roboczy: {target_dir}")
    
    # Inicjalizacja nowych plików (Nadpisujemy jeśli są jakieś resztki z przerwanego treningu)
    with open(final_results_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric_Name', 'Value', 'Description'])
        
    with open(probing_log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_BCE_Loss', 'Val_Target_AUC'])

    def log_final_metric(name, value, description=""):
        with open(final_results_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, value, description])
            
    def log_probing_epoch(epoch, loss, auc):
        with open(probing_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, loss, auc])
    
    # Odczytywanie logów DANN, żeby dowiedzieć się jakie było Alpha (tylko informacyjnie)
    log_df = pd.read_csv(log_file_path)
    if not log_df.empty and 'Val_Target_AUC' in log_df.columns:
        best_epoch_idx = log_df['Val_Target_AUC'].idxmax()
        best_prev_auc = log_df.loc[best_epoch_idx, 'Val_Target_AUC']
        best_alpha = log_df.loc[best_epoch_idx, 'Alpha']
        best_epoch_num = log_df.loc[best_epoch_idx, 'Epoch']
        print(f"📈 Z logów DANN: Najlepszy model był w epoce {best_epoch_num} (AUC={best_prev_auc:.4f}, Alpha={best_alpha:.2f})")
    else:
        best_prev_auc = 0.5
        best_alpha = 0.0

    # ---------------------------------------------------------
    # Przygotowanie danych (Loader) - num_workers=0 przeciw deadlockom
    # ---------------------------------------------------------
    folds, fold_override = read_folds_override(data_pth, None)
    all_eids_raw = get_eids_for_folds(fold_override, [1, 2, 3, 4, 5, 6]) 
    train_pool, test_pool = [], []
    valid_eids_set = set(df_meta['examination_id'].values)

    for eid in all_eids_raw:
        eid_clean = clean_eid_str(eid)
        if eid_clean in valid_eids_set:
            h_name = eid_to_hosp_name.get(eid_clean, "Unknown")
            if h_name == TARGET_HOSPITAL_CODE: test_pool.append(eid)
            elif h_name != "Unknown": train_pool.append(eid)

    if len(test_pool) == 0:
        print(f"⚠️ Brak pacjentów testowych. Pomijam.")
        continue

    train_loader = Loader(data_pth, train_pool, minet_subsampling_n=4, num_workers=0).get_batched_loader(batch_size=32, pad=True)
    test_loader = Loader(data_pth, test_pool, minet_subsampling_n=None, num_workers=0).get_batched_loader(batch_size=1, pad=True)

    # ---------------------------------------------------------
    # Inicjalizacja Modelu i wczytanie starych wag z best_model.pt
    # ---------------------------------------------------------
    print(f"🧠 Wczytywanie modelu bazowego MINET i załadowanie wag DANN...")
    raw_backbone = torch.load(model_pth, map_location=device)
    if hasattr(raw_backbone, 'n_chans'): raw_backbone.n_chans = 19
    dann_model = MinetDANN(raw_backbone, feature_dim=288, num_domains=len(all_hospitals)).to(device)
    
    dann_model.load_state_dict(torch.load(model_weights_path, map_location=device))

    # ZAMRAŻANIE (Linear Probing)
    print("❄️ Zamrażanie ekstraktora cech (Linear Probing)...")
    for param in dann_model.parameters():
        param.requires_grad = False
    for param in dann_model.backbone.classifier.parameters():
        param.requires_grad = True # Odmrażamy TYLKO klasyfikator medyczny (diagnozę)

    probing_optimizer = optim.AdamW(dann_model.backbone.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_cls_fn = nn.BCEWithLogitsLoss()
    
    # UWAGA: Usunięto instancję GradScaler() pod kątem wymuszenia CPU

    # ---------------------------------------------------------
    # PĘTLA TRENINGOWA (tylko head, 30 Epok)
    # ---------------------------------------------------------
    PROBING_EPOCHS = 30 
    STEPS = 500
    
    best_probing_auc = 0.0
    best_probing_wts = copy.deepcopy(dann_model.state_dict())
    
    print(f"🚀 Rozpoczynam dotrenowywanie głowy medycznej przez {PROBING_EPOCHS} epok...")
    print(f"   (Logi treningowe zapisywane do: {probing_log_file})")
    
    for epoch in range(PROBING_EPOCHS):
        dann_model.train()
        dann_model.backbone.eval() # Zabezpiecza BatchNorm
        dann_model.backbone.classifier.train() 
        
        iter_train = iter(train_loader)
        pbar = tqdm(range(STEPS), desc=f"Probing {epoch+1}/{PROBING_EPOCHS}")
        
        epoch_loss = 0.0
        
        for i in pbar:
            try: batch = next(iter_train)
            except StopIteration:
                iter_train = iter(train_loader)
                batch = next(iter_train)
                
            X = batch[0].to(device, dtype=torch.float32)
            y = batch[1].to(device, dtype=torch.float32).view(-1, 1)
            ts = torch.zeros((X.shape[0], X.shape[1]), dtype=torch.long, device=device)

            probing_optimizer.zero_grad()
            
            # Usunięto z tego bloku with autocast(): 
            c_pred, _ = dann_model(X, time_stamps=ts, alpha=0.0)
            loss_cls = loss_cls_fn(c_pred, y)
            
            # Klasyczny backpropagation dla CPU (bez Scalera)
            loss_cls.backward()
            torch.nn.utils.clip_grad_norm_(dann_model.backbone.classifier.parameters(), 1.0)
            probing_optimizer.step()
            
            epoch_loss += loss_cls.item()
            pbar.set_postfix({'BCE_Loss': f"{loss_cls.item():.3f}"})

        avg_loss = epoch_loss / STEPS

        # Walidacja co epokę
        dann_model.eval() # Zabezpieczenie przed błędem wariancji dla batch_size=1
        val_targets, val_preds = [], []
        with torch.no_grad():
            for data in test_loader: 
                X_v, y_v = data[0].to(device, dtype=torch.float32), data[1].to(device, dtype=torch.float32)
                if X_v.shape[1] > 300: X_v = X_v[:, :300]
                ts_v = torch.zeros((1, X_v.shape[1]), dtype=torch.long, device=device)
                
                logits, _ = dann_model(X_v, time_stamps=ts_v, alpha=0.0)
                val_targets.append(y_v.item())
                val_preds.append(torch.sigmoid(logits).item())
                
        try: current_probing_auc = roc_auc_score(val_targets, val_preds)
        except ValueError: current_probing_auc = 0.5 
        
        print(f"   🔬 Epoka {epoch+1}: Train BCE = {avg_loss:.4f} | Val AUC = {current_probing_auc:.4f}")
        
        # Zapis logów
        log_probing_epoch(epoch+1, avg_loss, current_probing_auc)
        
        if current_probing_auc > best_probing_auc:
            best_probing_auc = current_probing_auc
            best_probing_wts = copy.deepcopy(dann_model.state_dict())
            torch.save(dann_model.state_dict(), os.path.join(target_dir, "probing_best_model.pt"))

    # ---------------------------------------------------------
    # EWALUACJA I ZAPIS
    # ---------------------------------------------------------
    print(f"\n✅ Zakończono Probing dla {TARGET_HOSPITAL_CODE}. Przywracam najlepsze wagi (AUC={best_probing_auc:.4f})...")
    dann_model.load_state_dict(best_probing_wts)
    dann_model.eval()
    
    y_true, y_prob = [], []
    with torch.no_grad():
        for data in tqdm(test_loader, desc=f"Testing Probing_Best"):
            X, y = data[0].to(device, dtype=torch.float32), data[1]
            if X.shape[1] > 300: X = X[:, :300]
            ts = torch.zeros((1, X.shape[1]), dtype=torch.long, device=device)
            try:
                logits, _ = dann_model(X, time_stamps=ts, alpha=0.0)
                y_prob.append(torch.sigmoid(logits).item())
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
        
        print(f"   👉 Ostateczne wyniki Probing: AUC={auc_val:.4f} | MCC={mcc_val:.4f} | Acc={acc_val:.4f}")
        
        log_final_metric(f"Probing_Best_Target_Diagnosis_AUC", auc_val, f"Diag AUC after Linear Probing (Best DANN alpha={best_alpha:.2f})")
        log_final_metric(f"Probing_Best_Target_Diagnosis_MCC", mcc_val, "Diag MCC after Linear Probing")
        log_final_metric(f"Probing_Best_Target_Diagnosis_Acc", acc_val, "Diag Accuracy after Linear Probing")
        
        print(f"💾 Wyniki zapisano do {final_results_file}")

print("\n🎉 ZAKOŃCZONO LINEAR PROBING DLA WSZYSTKICH SZPITALI!")