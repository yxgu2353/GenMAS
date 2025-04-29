# GenMAS:An Artificial Intelligence-Driven Pharmacokinetics Modelling and Assessment Strategy Incorporating ADMET and PBPK


**Our Group**: Yaxin Gu†, Peng Qi†, Lingling Ma, Guodong Zhang*, Biao Lu, Fanglong Yang, Haizhou Zhang @ Changchun Genescience Pharma
**Last Update**: 2025年4月

---

## 📝 Abstract
GenMAS, the AI-driven pharmacokinetic modeling and assessment strategy adaptable to various application scenarios, offers a promising non-animal alternative.
---

## ✨ Key Modules
- **AI-ADMET module**：Compound ADMET property prediction.
- **AI-assist PBPK module**：AI parameter prediction incorporated into GastroPlus® PBPK model
- **Self-built ADMET-PBPK HTS module**：compound property prediction and a custom-built PBPK model were packaged into a Python-based pipeline

---

## 🚀 Quick Start
### Install
```bash
conda create -n [env_name] python=3.8
conda activate [env_name]
pip install rdkit, scikit-learn, torch, dgl, dgllife
