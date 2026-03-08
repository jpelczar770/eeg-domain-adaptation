# EEG Domain Adaptation: Mitigating Site Effects using DANN

This repository contains the code for my Master's Thesis, focused on mitigating "site effects" (covariate shift) in multi-center clinical Electroencephalography (EEG) data. It leverages Domain-Adversarial Neural Networks (DANN) and the MINET architecture to learn domain-invariant, yet highly discriminative features for neurological diagnosis.

## 🧠 Background & Motivation

In multi-center medical studies, machine learning models often suffer from a performance drop when tested on data from an unseen hospital. This "site effect" is caused by differences in hardware, recording protocols, reference electrodes, and patient demographics.

To solve this, this project implements a Strict Leave-One-Site-Out (LOSO) evaluation strategy across 30 different hospitals, comparing several Domain Adaptation techniques to harmonize the learned EEG representations.

## 🏗️ Implemented Architectures

The core feature extractor is based on the MINET (1D Convolutional Neural Network with Attention). On top of this backbone, the following approaches are implemented:

1. **Baseline (Unharmonized)**: Standard MINET trained purely on the medical diagnosis task.

2. **Naive Multi-Task Learning**: The model simultaneously predicts the diagnosis and the origin hospital (Domain) without reversing the gradient. It retains domain-specific knowledge.

3. **DANN (Raw)**: Domain-Adversarial Neural Network trained from scratch. It utilizes a Gradient Reversal Layer (GRL) to penalize the model for learning hospital-specific features, forcing the feature space to be domain-invariant.

4. **DANN (Fine-tuning)**: The backbone is initialized with pre-trained weights, and the network is fine-tuned using the adversarial domain loss.

## 🔒 Data Privacy Note

The clinical EEG datasets (HDF5 files and label metadata CSVs) used in this research belong to multiple clinical centers and are highly confidential. Therefore, the dataset is not included in this repository. The code is provided for academic reference, methodology sharing, and transparency.

## 🎓 Citation & Acknowledgements

## Acknowledgments

This project is built upon foundational research in both EEG-based machine learning and domain adaptation. Specifically, it leverages the methodology for EEG pathology detection by **Poziomska et al. (2025)**, integrating it with the Domain-Adversarial Neural Network (DANN) framework introduced by **Ganin et al. (2016)**.

**References:**
> Poziomska, M., Dovgialo, M., Olbratowski, P., Niedbalski, P., Ogniewski, P., Zych, J., ... & Żygierewicz, J. (2025). Quantity versus diversity: Influence of data on detecting EEG pathology with advanced ML models. *Neural Networks*, 108073.

> Ganin, Y., Ustinova, E., Ajakan, H., Germain, P., Larochelle, H., Laviolette, F., ... & Lempitsky, V. (2016). Domain-adversarial training of neural networks. *The Journal of Machine Learning Research*, 17(1), 2096-2030.
