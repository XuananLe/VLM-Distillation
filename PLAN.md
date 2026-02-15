# VLM Teacher Selection & Distillation Pipeline

## Problem Statement

**Setup**: A pool of $N$ teacher VLMs with diverse architectures, 1 fixed student VLM (smaller architecture)

**Goal**: Select $K$ optimal teachers from the pool and train the student to maximize knowledge transfer

---

## 🔥 Key Novel Contributions

| # | Contribution | Step | Why It's Novel |
|---|--------------|------|----------------|
| **1** | **Spectral Knowledge Complementarity (SKC)** | Selection | First to use eigenspectrum overlap to measure knowledge redundancy vs. complementarity |
| **2** | **Cross-Modal Alignment Fidelity (CMAF)** | Selection | Novel metric for vision-language alignment preservation |
| **4** | **Sample-Adaptive Teacher Routing (SATR)** | Training | Per-sample dynamic teacher weighting based on uncertainty |
| **5** | **Curriculum Teacher Scheduling (CTS)** | Training | Progressive teacher introduction during training |
| **3** | **Knowledge Transfer Potential (KTP)** | Selection | Information-theoretic bound on distillable knowledge 
| **6** | **Teacher-Guided MoE (TG-MoE)** | Training | SKC-informed teacher-to-expert assignment |

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        STEP 1: TEACHER SELECTION                            │
│                         (Static, Before Training)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  1.1 Data Subset Selection (G-Vendi)                                        │
│       ↓                                                                     │
│  1.2 Teacher Competence Evaluation                                          │
│       ↓                                                                     │
│  1.3 Teacher Diversity Analysis (SKC)                                       │
│       ↓                                                                     │
│  1.4 Student Compatibility Assessment (KTP + CMAF)                          │
│       ↓                                                                     │
│  1.5 Optimal Selection (Submodular Optimization)                            │
│       ↓                                                                     │
│  Output: K selected teachers + SKC matrix + KTP scores                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                        STEP 2: STUDENT TRAINING                             │
│                         (Dynamic, During Training)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  2.1 MoE Expert Assignment (from SKC clustering)                            │
│       ↓                                                                     │
│  2.2 Curriculum Teacher Scheduling (CTS)                                    │
│       ↓                                                                     │
│  2.3 Sample-Adaptive Teacher Routing (SATR)                                 │
│       ↓                                                                     │
│  2.4 Teacher-Guided MoE Distillation (TG-MoE)                               │
│       ↓                                                                     │
│  Output: Trained student MoE model                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

# STEP 1: TEACHER SELECTION (Static, Before Training)

This step runs **once** before training begins. It selects K optimal teachers from the pool of N candidates.

---

## 1.1 Data Subset Selection (G-Vendi)

### Motivation
Evaluating all teachers on the full dataset is expensive. We need a small, diverse, representative subset that serves as an "entrance exam" for teachers.

### Method: G-Vendi Score (Gradient-based Diversity)

Standard Vendi Score uses feature similarity, but ignores **how the model will use** the data. G-Vendi uses gradient similarity to capture **functional diversity**.

**Gradient Similarity Kernel**:
$$K_{ij}^{\text{grad}} = \cos(\nabla_\theta \mathcal{L}(x_i, \theta), \nabla_\theta \mathcal{L}(x_j, \theta))$$

**G-Vendi Score**:
$$\text{G-Vendi}(S) = \exp\left(-\sum_{i=1}^{n} \lambda_i \log \lambda_i\right)$$

Where $\lambda_i$ are eigenvalues of the gradient similarity matrix $K^{\text{grad}}$.

### Why G-Vendi?

| Aspect | Standard Vendi | G-Vendi |
|--------|----------------|---------|
| **Kernel** | Feature similarity (RBF) | Gradient similarity |
| **Measures** | Semantic diversity | Functional diversity |
| **Key insight** | Diverse in feature space | Diverse in gradient space → diverse learning signal |

### Algorithm
1. Compute gradients for candidate samples using a reference model
2. Build gradient similarity matrix $K^{\text{grad}}$
3. Greedily select samples that maximize G-Vendi score
4. Return subset $D_{\text{subset}}$

---

## 1.2 Teacher Competence Evaluation

### Motivation
Filter out teachers that perform poorly. No point in distilling from a bad teacher.

### Method

**Performance Evaluation**:
$$c_i = \text{eval}(\text{teacher}_i, D_{\text{subset}})$$

**Filtering**: Remove teachers where $c_i < \tau$ (threshold = student baseline)

**Normalization**:
$$\tilde{c}_i = \frac{c_i - c_{\min}}{c_{\max} - c_{\min}}$$

---

## 1.3 Teacher Diversity Analysis

### Motivation
Multiple teachers with similar knowledge provide redundant signal. We want teachers that provide **complementary** knowledge.

### Novel Metric: Spectral Knowledge Complementarity (SKC)

CKA measures *alignment* but not *complementarity*. Two teachers may have low CKA but encode **the same information** in different bases. SKC measures true non-redundancy.

**Key Insight**: The **eigenspectrum** of a representation reveals its information structure.

Given teacher representation $H \in \mathbb{R}^{n \times d}$, compute SVD: $H = U\Sigma V^\top$

**Spectral Signature** (normalized eigenvalues):
$$\sigma_T = \left(\frac{\lambda_1}{\sum_j \lambda_j}, \frac{\lambda_2}{\sum_j \lambda_j}, \ldots\right)$$

**Spectral Difference** (Jensen-Shannon divergence):
$$\text{SpectralDiff}(T_1, T_2) = 1 - \text{JS}\left(\sigma_{T_1} \| \sigma_{T_2}\right)$$

**Subspace Overlap Score** (top-k principal components):
$$\text{SubspaceOverlap}(T_1, T_2, k) = \frac{1}{k}\sum_{i=1}^{k} \max_j |\langle v_i^{(1)}, v_j^{(2)} \rangle|^2$$

**Final SKC**:
$$\text{SKC}(T_1, T_2) = \underbrace{\text{SpectralDiff}}_{\text{Different info structure}} \times \underbrace{(1 - \text{SubspaceOverlap})}_{\text{Non-redundant subspaces}}$$

### Why SKC Is Novel
- **Prior work (Vendi Score)**: Measures diversity of *data points*
- **Our SKC**: Measures *knowledge complementarity* between *teacher representations*
- **Key difference**: Analyzes both spectral *shape* AND subspace *overlap*

---

## 1.4 Student Compatibility Assessment

### Motivation
Not all teachers transfer knowledge equally well to a given student. The capacity gap matters.

### Novel Metrics

#### Knowledge Transfer Potential (KTP)

Not all knowledge is **transferable**. KTP provides an information-theoretic upper bound.

$$\text{KTP}(T, S) = \text{CKA}(H_T, H_S) \times \frac{\text{EffectiveDim}(H_S)}{\text{EffectiveDim}(H_T)}$$

**Effective Dimension**:
$$\text{EffectiveDim}(H) = \exp\left(-\sum_i \hat{\lambda}_i \log \hat{\lambda}_i\right)$$

**Key insight**: A powerful teacher is useless if the student can't absorb its knowledge.

#### Cross-Modal Alignment Fidelity (CMAF)

VLMs must maintain **vision-language alignment**. CMAF measures if the student can learn the teacher's alignment structure.

**Cross-Modal Alignment Matrix**:
$$A = \text{softmax}\left(\frac{VW_pL^\top}{\tau}\right)$$

**CMAF**:
$$\text{CMAF}(T, S) = \text{CKA}(A_T, A_S) \times \exp\left(-|H(A_T) - H(A_S)|\right)$$

**Key insight**: Good distillation requires matching how V and L relate, not just V and L separately.

### Background: Centered Kernel Alignment (CKA)

CKA is the foundation for compatibility metrics:

**Linear CKA**:
$$\text{CKA}(X, Y) = \frac{\|Y^\top X\|_F^2}{\|X^\top X\|_F \cdot \|Y^\top Y\|_F}$$

| Property | Why It Matters |
|----------|----------------|
| **Invariant to orthogonal transformation** | Works across different architectures |
| **Invariant to isotropic scaling** | Robust to layer norm differences |
| **Works with different feature dimensions** | Compares ViT vs CNN |

**VLM Component-Level CKA**:
$$\text{compat} = 0.2 \times \text{CKA}_{vision} + 0.3 \times \text{CKA}_{LM} + 0.2 \times \text{CKA}_{projector} + 0.3 \times \text{CKA}_{output}$$

---

## 1.5 Optimal Teacher Selection

### Novel Score Function

$$\text{Score}(T_i | S, \mathcal{T}_{selected}) = \underbrace{\alpha \cdot \tilde{c}_i}_{\text{Competence}} + \underbrace{\beta \cdot \text{KTP}(T_i, S)}_{\text{Transfer Potential}} + \underbrace{\gamma \cdot \text{SKC}(T_i, \mathcal{T}_{selected})}_{\text{Complementarity}} + \underbrace{\delta \cdot \text{CMAF}(T_i, S)}_{\text{Cross-Modal Fidelity}}$$

**Recommended Weights**: $\alpha = 0.30$, $\beta = 0.25$, $\gamma = 0.25$, $\delta = 0.20$

### Greedy Teacher Selection Algorithm

1. Initialize $\mathcal{T}_{selected} = \emptyset$
2. For $k = 1$ to $K$:
   - For each candidate $T_i \notin \mathcal{T}_{selected}$:
     - Compute $\text{Score}(T_i | S, \mathcal{T}_{selected})$
   - Select $T^* = \arg\max_{T_i} \text{Score}(T_i)$
   - Add $T^*$ to $\mathcal{T}_{selected}$
3. Return $\mathcal{T}_{selected}$

**Note**: SKC term is recomputed at each step to measure complementarity with already-selected teachers.

### Output of Step 1
- **K selected teachers** $\{T_1, ..., T_K\}$
- **SKC matrix** $\mathbf{S} \in \mathbb{R}^{K \times K}$ (for MoE assignment)
- **KTP scores** for each teacher (for curriculum scheduling)
- **Competence scores** (for load balancing)

---

# STEP 2: STUDENT TRAINING (Dynamic, During Training)

This step uses the selected teachers to train the student. Components adapt **dynamically** during training.

---

## 2.1 MoE Expert Assignment (from SKC)

### Motivation
Standard multi-teacher distillation averages signals, losing specialization. MoE enables each expert to learn from specific teacher(s).

### SKC-Informed Assignment

Teachers with **low SKC** (redundant) → share an expert  
Teachers with **high SKC** (complementary) → separate experts

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| **1-to-1 Mapping** | Each expert ← one teacher | K teachers = K experts |
| **SKC Clustering** | Cluster teachers by SKC → each cluster maps to one expert | K teachers > desired experts |
| **Soft Assignment** | Each expert learns from weighted teacher combination | Smooth knowledge mixing |

### Algorithm
1. Compute pairwise SKC matrix $\mathbf{S}$ (from Step 1)
2. Convert to distance: $D_{ij} = 1 - S_{ij}$
3. Cluster teachers into $M$ groups via spectral clustering
4. Assign $E_{ij} = 1$ if teacher $i$ in cluster $j$

**Expert Assignment Matrix**: $\mathbf{E} \in \mathbb{R}^{K \times M}$

---

## 2.2 Curriculum Teacher Scheduling (CTS)

### Motivation
Curriculum learning is well-established for data, but not for **teachers**. The optimal teacher set changes during training.

### Three-Phase Curriculum

| Phase | Training Progress | Teacher Focus | Why |
|-------|-------------------|---------------|-----|
| **Early** | 0-30% | High Compatibility (KTP) | Build foundation with easily learnable knowledge |
| **Middle** | 30-70% | Balanced | Introduce diverse perspectives |
| **Late** | 70-100% | High Complementarity (SKC) | Fill gaps with complementary knowledge |

### Mathematical Formulation

**Compatibility-Diversity Tradeoff**:
$$\alpha(t) = \alpha_0 \cdot \exp(-\gamma \cdot t / T)$$

**Teacher Scheduling Policy**:
$$\mathcal{T}(t) = \text{Top-K}\left(\alpha(t) \cdot \text{KTP} + (1 - \alpha(t)) \cdot \text{SKC}\right)$$

### Why CTS Is Novel
- **Prior work**: Curriculum for data difficulty
- **Our CTS**: First curriculum for **teacher selection** in distillation

---

## 2.3 Sample-Adaptive Teacher Routing (SATR)

### Motivation
Different teachers excel at different samples. We should **dynamically weight teachers per sample**.

### Uncertainty-Guided Routing

When the student is uncertain, weight teachers that historically helped on similar uncertain samples.

**Student Uncertainty**:
$$u(x) = H\left(\text{softmax}(f_S(x))\right)$$

**Teacher Routing Weights**:
$$w_i(x) = \text{softmax}\left(\frac{\text{sim}(x, \mu_i)}{\tau} + \lambda \cdot \text{conf}_i(x)\right)$$

Where:
- $\mu_i$: Prototype of samples where teacher $i$ helped most
- $\text{conf}_i(x)$: Teacher $i$'s confidence on sample $x$

### Why SATR Is Novel
- **Prior work**: Fixed weights or instance-agnostic routing
- **Our SATR**: Uncertainty-guided, sample-adaptive routing with learnable prototypes

---

## 2.4 Teacher-Guided MoE Distillation (TG-MoE)

### Unified Router

The MoE router is trained to be consistent with SATR's teacher routing:

$$r(x) = \text{softmax}\left(\frac{W_r \cdot h(x)}{\tau}\right)$$

**Teacher-Guided Router Loss**:
$$\mathcal{L}_{router} = \text{KL}\left(r(x) \| \sum_{i=1}^{K} E_{:,i} \cdot w_i^{SATR}(x)\right)$$

### MoE Distillation Loss

**Per-Expert Teacher-Specific Loss**:
$$\mathcal{L}_{MoE}(x) = \sum_{j=1}^{M} r_j(x) \cdot \left[\sum_{i=1}^{K} E_{ij} \cdot \mathcal{L}_{KD}(e_j(x), f_{T_i}(x))\right]$$

### Competence-Aware Load Balancing

$$\mathcal{L}_{balance} = \sum_{j=1}^{M} \left(\bar{r}_j - \frac{\sum_{i} E_{ij} \cdot c_i}{\sum_i c_i}\right)^2$$

Encourages routing proportional to teacher competence, not uniform.

### Total Training Loss

$$\mathcal{L}_{total} = \mathcal{L}_{MoE} + \lambda_1 \mathcal{L}_{router} + \lambda_2 \mathcal{L}_{balance}$$

### Why TG-MoE Is Novel

| Aspect | Prior MoE Distillation | Our TG-MoE |
|--------|------------------------|------------|
| **Expert assignment** | Random or learned | SKC-informed clustering |
| **Router training** | Standard load balancing | Teacher-guided via SATR |
| **Multi-teacher** | Average all teachers | Per-expert teacher specialization |
| **Selection + Training** | Separate problems | Unified pipeline |

---

# Summary: Static vs Dynamic Components

| Component | When | What Changes |
|-----------|------|--------------|
| **G-Vendi Data Selection** | Step 1 (once) | Fixed data subset |
| **Teacher Competence** | Step 1 (once) | Fixed scores |
| **SKC Matrix** | Step 1 (once) | Fixed pairwise complementarity |
| **KTP & CMAF** | Step 1 (once) | Fixed compatibility scores |
| **Teacher Selection** | Step 1 (once) | K teachers fixed |
| **MoE Expert Assignment** | Step 2 (once) | Fixed assignment matrix |
| **CTS Scheduling** | Step 2 (per epoch) | Active teachers change over training |
| **SATR Routing** | Step 2 (per sample) | Teacher weights change per input |
| **MoE Router** | Step 2 (per sample) | Expert weights change per input |

---

# Experimental Design

### Ablation Studies

1. **SKC vs CKA** for diversity measurement
2. **CMAF contribution** to distillation quality
3. **CTS scheduling** vs static teacher weights
4. **SATR routing** vs uniform weights
5. **MoE vs dense** student architecture

### Baselines

| Baseline | Description |
|----------|-------------|
| Random-K | Randomly select K teachers |
| Top-K Competence | Select K most competent teachers |
| Top-K Diversity | Select K most diverse teachers (min CKA) |
| Top-K Compatible | Select K most KTP-compatible with student |
| Single Best | Use only the best teacher |
| All Teachers | Ensemble all teachers (upper bound) |

---

# Novelty Comparison with Prior Work

| Aspect | Prior Work | Our Approach | Why Novel |
|--------|------------|--------------|-----------|
| **Teacher Diversity** | CKA similarity | **SKC**: Spectral + subspace | Captures *complementarity*, not just *difference* |
| **Compatibility** | CKA alignment | **KTP**: Capacity-adjusted | Accounts for student's limited capacity |
| **VLM-Specific** | Generic distillation | **CMAF**: V-L alignment | First to consider cross-modal structure |
| **Training Dynamics** | Static weights | **SATR + CTS** | Adaptive to sample & training phase |
| **MoE Integration** | Separate | **TG-MoE** | Selection informs MoE design |

---

# Key References

1. **CKA** - Kornblith et al. (2019) - arXiv:1905.00414
2. **Vendi Score** - Friedman & Dieng (2022) - arXiv:2210.02410
3. **G-Vendi Score** - arXiv:2505.20161
4. **Teacher Assistant KD** - Mirzadeh et al. (2019) - arXiv:1902.03393
5. **MT-BERT** - Wu et al. (2021) - arXiv:2106.01023
6. **Beyond Neural Scaling Laws** - Sorscher et al. (2022) - arXiv:2206.14486

---

### Contributions
1. **Spectral Knowledge Complementarity (SKC)**: Novel metric for knowledge non-redundancy
2. **Cross-Modal Alignment Fidelity (CMAF)**: VLM-specific V-L alignment metric
3. **Knowledge Transfer Potential (KTP)**: Information-theoretic transferability bound
4. **Sample-Adaptive Teacher Routing (SATR)**: Uncertainty-guided per-sample weighting
5. **Curriculum Teacher Scheduling (CTS)**: Progressive teacher introduction
6. **Teacher-Guided MoE (TG-MoE)**: SKC-informed expert assignment with unified routing
