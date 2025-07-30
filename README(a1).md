# 🔍 Quantitative Association Rule Mining — Advanced Apriori Framework

## 📘 Project Summary
This project presents a comprehensive framework for **quantitative association rule mining**, featuring **three distinct Apriori-based algorithms**. The solution includes rigorous statistical validation, advanced rule analysis, and **Shapley value-based interpretation** for measuring attribute contributions using **J-Measure coalition payoff optimization**.

> ✅ Designed for advanced data mining coursework and academic submission, with high-performance implementations, insightful visualizations, and statistically sound outputs.

---

## ⚙️ Core Algorithms Implemented

| Algorithm                | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| 🔹 **Optimized Apriori** | Classical level-wise strategy with efficient interval handling              |
| 🔸 **Randomic Apriori**  | Depth-first exploration with probabilistic sampling and rule filtering      |
| 🔁 **Distributed Apriori** | Multi-threaded version for parallel itemset mining and global result merging |

---

## 🧪 Exercise Breakdown

### 📌 Exercise 1 – Algorithm Design & Execution  
- Implemented all three Apriori variants  
- Verified via benchmark datasets  
- Optimized for execution time and correctness  

### 📌 Exercise 2 – Rule Extraction & Statistical Validation  
- Rules mined with **confidence ≥ 0.8**  
- Statistical checks: **Chi-square p-values**, **Lift**, **Support**  
- Exported clean rule sets with significant patterns

### 📌 Exercise 3 – Shapley Value Analysis  
- Coalition modeling with **J-Measure payoff function**  
- Attribute importance ranked using **Monte Carlo approximation**  
- Implemented custom **conflict resolution** (Cl function)

---

## 🆕 What's New

- 🎨 **Interactive Visualizations**: Modern HTML charts for exploration and presentation  
- 📈 **Statistical Validity**: All rules tested with p-values and confidence intervals  
- 🧠 **Shapley Integration**: Full contribution analysis for rule interpretability  
- 🚀 **Performance Benchmarks**: Execution profiling and comparative analysis  
- 💾 **Exportable Reports**: Clean CSVs, plots, and project summaries for review or publication

---





### ▶️ Running the Notebook
```bash
jupyter notebook a1.ipynb
```
Or open in VS Code and execute all cells to run the complete pipeline.

---

## 📊 Key Results

- **Itemsets Mined**: 893 unique quantitative patterns  
- **Rules Extracted**: 1,180 association rules  
- **Perfect Patterns**: 1.0 confidence discovered in weather data  
- **Execution Time**: Under 0.8 seconds (all algorithms)  
- **Quality Metrics**: Confidence, Lift, J-Measure, and p-value computed

---

## 🧬 Implementation Highlights

### 🔄 Data Generator
- Simulates realistic environmental data  
- Features: `Temperature`, `Humidity`, `Pressure`, and `Count`  
- Built-in correlations for richer rule mining

### 🧠 Apriori Framework
- Custom interval-based itemsets  
- Smart support pruning and interval shrinking  
- Clean lattice traversal and frontier detection

### 📈 Rule Mining Engine
- Exhaustive rule construction from frontier itemsets  
- Filters by confidence, lift > 1.5, and **p < 0.05**  
- Automatically exports detailed rule breakdown

### 🧮 Shapley Value Engine
- Monte Carlo sampling for approximation  
- Rule attributes mapped to coalitions  
- **J-Measure used as coalition payoff**  
- Conflict handling ensures reliable attribution

---

## 📐 Mathematical Backbone

- **Support (ε)**: Relative frequency of satisfied transactions  
- **Confidence**: Likelihood of consequent given antecedent  
- **Lift**: Strength of rule relative to expected independence  
- **J-Measure**: Rule informativeness based on entropy  
- **Shapley Value**: Attribute contribution across coalitions

---

## 📦 Submission Package

- ✅ Fully implemented exercises (1–3)  
- ✅ Results with statistical rigor  
- ✅ Visualizations for presentation  
- ✅ Documentation + code + exports  
- ✅ Ready for academic or industry submission  

---

## 🎓 Academic Notes

- 💡 Algorithms derived from: Agrawal & Srikant (1994/1996), Shapley (1953)  
- 📘 Built for: **Biomedical Decision Support Systems (BDSS)**, 4th Semester, 2025  
- 🎤 Oral Exam Ready: Includes algorithm design, performance metrics, and mathematical justifications

---

## 👨‍💻 Author

**Ritwick Haldar**  
*4th Semester – Biomedical Decision Support Systems*  


---

## 🧾 References
1. Agrawal, R., & Srikant, R. (1994). *Fast Algorithms for Mining Association Rules*  
2. Srikant, R., & Agrawal, R. (1996). *Mining Quantitative Association Rules*  
3. Shapley, L. S. (1953). *A Value for N-Person Games*
