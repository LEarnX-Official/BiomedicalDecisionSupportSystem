# 🔍 Process Mining & Automata Learning – Controllability Analysis

## 📘 Overview

This project presents a **complete pipeline** for analyzing process behavior using **Process Mining** and **Automata Learning** techniques. Through three progressive assignments, we extract, evaluate, and visualize processes using Petri nets and Markov models with a focus on **controllability analysis**.

## 🚀 Highlights

- **🧠 Entropy & Information Gain Analysis**
- **📈 Full Petri Net Visualizations**
- **📊 Trace Alignment & Model Quality Evaluation**
- **⚙️ Controllability Assessment via Relabeling & Entropy Metrics**
- **🎨 High-quality Visuals** for Petri nets, entropy plots, and comparisons

## 📂 Assignments Breakdown

### ✅ Assignment 1: Markov Process Construction
- Extract Petri Nets using **Inductive**, **Alpha**, and **Heuristics** algorithms
- Build Markov models from event logs
- Perform trace alignment and visualize conformance quality

### ✅ Assignment 2: Entropy and Information Gain
- Calculate **state** and **process-level** entropy
- Derive **information gain** to assess decision point clarity
- Visualize uncertainty across process models

### ✅ Assignment 3: Trace Test & Controllability
- Relabel traces based on control scenarios (Sets A & B)
- Compare entropy pre/post relabeling
- Quantify and visualize process controllability

## 🛠️ Tech Stack

- **Python 3.8+**, **Jupyter Notebook**
- **PM4Py**, **Graphviz**, **NetworkX**
- **Matplotlib**, **Seaborn**, **Plotly**
- **NumPy**, **Pandas**, **SciPy**

## 📁 Directory Structure

```
process_mining/
├── README(a2).md
├── requirements.txt
├── process_mining_complete.ipynb
├── results/
│   ├── data/ → logs, stats, entropy values
│   ├── models/ → pickled Petri nets, Markov models
│   ├── plots/ → assignment-specific and comparative visualizations
│   └── reports/ → controllability_analysis_report.md
```

## ⚡ Quickstart

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Launch notebook:
```bash
jupyter notebook process_mining_complete.ipynb
```

3. Run all cells to perform full analysis.

## 📊 Key Formulas

- **State Entropy**: H(Q) = -∑ P(Q→Qi) × log₂ P(Q→Qi)  
- **Process Entropy**: H(Process) = ∑ P(Q) × H(Q)  
- **Information Gain**: IG = H(Process) - H(TraceTest)

## 🎨 Visual Features

- Petri net graphs with places/transitions/flows
- State transition diagrams with probabilities
- Entropy distributions and heatmaps
- Controllability comparison charts

## 📈 Outcomes

- Accurate Petri net models from logs
- Markov process construction with transition weights
- Entropy-based process predictability analysis
- Relabeling-based controllability evaluation

## 👨‍🎓 Author

**Emil Alizada**  
Biomedical Decision Support Systems (BDSS), 4th Semester, 2025

---

For educational purposes only. All results reproducible via the included Jupyter notebook.
