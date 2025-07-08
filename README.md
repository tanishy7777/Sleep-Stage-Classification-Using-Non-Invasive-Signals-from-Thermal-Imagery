# Sleep Stage Classification Using Non-Invasive Signals from Thermal Imagery

## Poster Presentation
(click on the image for higher resolution)
![TanishYelgoe_PosterPresentation_final-1](https://github.com/user-attachments/assets/43aa10aa-d196-44e3-8cd3-f50f96fb8295)

## 📌 Motivation
Accurate identification of sleep stages is crucial for health assessment and early disease detection.
However, manual scoring is labor-intensive and invasive. While current state-of-the-art ML/DL
models achieve high accuracy, they rely on EEG, EOG, and EMG signals which are invasive and
cannot be derived from thermal imagery. This highlights the need for non-invasive alternatives
for sleep stage classification

 This project aims to:
 
    1. Evaluate whether machine learning models using non-invasive signals can approach the performance of EEG-based models.
    
    2. Analyze key EEG features (e.g., K-complexes, sleep spindles, α-rhythms, sawtooth waves) to understand their critical role as defined by AASM standards.

## Exploratory Data Analysis (EDA)
## Distribution of sleep stages in the dataset
![image](https://github.com/user-attachments/assets/f2e90cac-b611-4aed-af8f-f4262125406b)
![image](https://github.com/user-attachments/assets/ecc4393b-b453-48dd-a409-621c17217cf0)

### Sleep Stages for Subject 5 over time (a) 6 sleep stages (b) 2 sleep stages
![image](https://github.com/user-attachments/assets/de74224c-95ab-4e01-aeac-e5f7a208ef76)
![image](https://github.com/user-attachments/assets/ae5b0a54-5c33-4bae-9024-c767acaefe4c)

### Flow and Ribcage Signals(Non-invasive) for subjects
We observe the need to normalize the signals and apply smoothening
![image](https://github.com/user-attachments/assets/144aa650-0dc5-45fd-96ad-d48c943e585b)
![image](https://github.com/user-attachments/assets/8a9b3627-00c5-4b9f-893d-217f444552bd)


### Preprocessing using Savitsky-Goalsy filter with various hyperparameters for window length and polynomial order
![image](https://github.com/user-attachments/assets/8a35faa8-9880-4ffb-a987-e4a365d5b7c2)
![image](https://github.com/user-attachments/assets/c7d00997-a91c-4ae4-830f-0dd3f5076056)

###  Normalizing and smootening data
![image](https://github.com/user-attachments/assets/6149a92a-3894-485b-8f47-73b2682d1868)


## ⚙️ Methodology
### Preprocessing: Normalization, smoothing (Savitzky–Golay filter).

### Feature Extraction (3 ways are experimented here):
1. Using TSFEL
2. CNNs
3. hand‑crafted statistical & physiological features.

### Modeling: Decision Tree based(XGboost, Random Forests), sequence‑based (BiLSTM), and attention‑based (Transformer) architectures.
![image](https://github.com/user-attachments/assets/21372077-0b04-4913-8b7a-699ddef128ff)

### Evaluation: Accuracy and specificity on 2‑, 3‑, and 4‑class sleep staging tasks.
![image](https://github.com/user-attachments/assets/d5898645-b667-40ba-af81-708487367161)


## 📊 Why Specificity Matters
High specificity means fewer false positives in medical diagnostics:

Better resource allocation: Focus interventions (e.g., CPAP therapy) on truly affected patients.

Greater clinical confidence: Avoid unnecessary treatments caused by false alarms.

## ✅ Conclusions
  Non-invasive signals can be used for simple Sleep/Wake classification in smartwatches, etc as accuracy is \textbf{0.83}.
    
  Feature engineering improves specificity of the classes significantly. Specificity is really important especially in medical domain, as it tells us how confident we are that the negatives predicted for a class are actually negative. For example:

  Medical Diagnosis: High specificity means that when a model says that a patient is not having a particular sleep disorder (e.g., obstructive sleep apnea), clinicians can confidently exclude that diagnosis. This prevents unnecessary treatments, medications, or interventions that carry their own side effects and costs.
    
Resource Allocation: In sleep medicine, treatments like CPAP therapy are resource-intensive. High specificity reduces false positives, ensuring these resources are directed to patients who truly need them.
EEG signals are critical for multi-stage classification (REM vs NREM, Light vs Deep, etc). but XGboost with appropriate feature extraction can be promising.


## 📚 References
AASM Scoring Manual (2017)

Mohammadi Foumani et al., Data Mining and Knowledge Discovery, 2023

Lee et al., IEEE Transactions on Neural Systems and Rehabilitation Engineering, 2024










