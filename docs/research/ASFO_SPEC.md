# Adaptive Semantic Federated Optimization (ASFO) - Research Specification

## 1. Introduction
Adaptive Semantic Federated Optimization (ASFO) is a novel server-side federated optimization algorithm for cross-domain intrusion detection. Standard federated algorithms (e.g., FedAvg, FedProx) optimize for Structural Empirical Risk Minimization (SERM) by weighting client updates based on dataset volume or local training steps. This inherently biases the global model toward majority classes (benign traffic) and suppresses the gradients of rare, high-impact cyber threats (e.g., zero-day exploits).

ASFO introduces **Semantic Risk Minimization (SRM)**. By querying a Threat Intelligence Knowledge Graph (KG), ASFO evaluates the semantic novelty, severity, and rarity of the attack distribution on each client. It frames the global aggregation as an **adaptive scalarization of two objectives**, balancing volumetric representation learning with semantic threat adaptation via an adaptive momentum-based solver.

## 2. Optimization Formulation

Let $w \in \mathbb{R}^d$ be the global model parameters. In standard FL, the goal is to minimize the global volumetric objective $F_{vol}(w) = \sum_{k=1}^N p_k F_k(w)$, where $p_k = \frac{|D_k|}{\sum |D_j|}$.

ASFO introduces a parallel semantic objective:
$$ F_{sem}(w) = \sum_{k=1}^N s_k^{(t)} F_k(w) $$
where $s_k^{(t)}$ is the dynamic semantic weight of client $k$ at round $t$.

The overarching ASFO objective is the weighted multi-objective optimization of both risks:
$$ \min_{w} \mathcal{L}(w) = (1 - \lambda^{(t)}) F_{vol}(w) + \lambda^{(t)} F_{sem}(w) $$

Where $\lambda^{(t)} \in [0, 1]$ is the **Adaptive Semantic Trade-off Parameter** computed dynamically at each round to ensure stable convergence while maximizing threat recall.

## 3. Semantic Threat Intelligence Integration

The Knowledge Graph $\mathcal{K} = (\mathcal{V}, \mathcal{E})$ provides deterministic semantic impact scores derived from the knowledge graph for attack classes. The semantic weight $s_k^{(t)}$ is derived purely on the server side (requiring no client-side model modifications).

1. **Client Class Distribution (Privacy):** Each client $k$ transmits its class frequency vector $f_k \in \mathbb{R}^{|C|}$ along with its model update $\Delta w_k^{(t)}$. This is a highly aggregated, low-dimensional metadata vector containing no sample-level information, preserving privacy constraints. For strict environments, it is compatible with standard Differential Privacy (DP) mechanisms by adding Laplacian noise before transmission.
   
   To prevent clients with larger datasets from skewing the semantic impact, the frequency vector is L1-normalized:
   $$ \hat{f}_k = \frac{f_k}{\|f_k\|_1} $$

2. **KG Impact Vector:** The server queries $\mathcal{K}$ to retrieve a deterministic Impact vector $I \in \mathbb{R}^{|C|}$, where $I_c$ represents the severity of attack $c$ (e.g., derived from CVSS base scores or computed via Knowledge Graph centrality metrics).

3. **Global Rarity Vector:** The server maintains an inverse frequency vector $R^{(t)} \in \mathbb{R}^{|C|}$ tracking the rarity of classes. Let $n_c^{(t)}$ be the total global sample count of class $c$ observed up to round $t$. The rarity is defined as:
   $$ R_c^{(t)} = \frac{1}{\epsilon + n_c^{(t)}} $$
   where $\epsilon$ is a small smoothing constant.

The **Semantic Risk Score (SRS)** for client $k$ is the dot product of its normalized distribution against the threat intelligence:
$$ \text{SRS}_k^{(t)} = \hat{f}_k \cdot (I \odot R^{(t)}) $$

The semantic weights are obtained via a Softmax transformation scaled by temperature $\beta > 0$:
$$ s_k^{(t)} = \frac{\exp(\beta \cdot \text{SRS}_k^{(t)})}{\sum_{j=1}^N \exp(\beta \cdot \text{SRS}_j^{(t)})} $$
*Note: A small $\beta$ results in uniform semantic weights, while a large $\beta$ allows the single highest-risk client to dominate the semantic gradient.*

## 4. Aggregation Equations

To solve the dual-objective problem without altering the local client optimizer, ASFO operates strictly on the server side using the pseudo-gradients $\Delta w_k^{(t)} = w_k^{(t)} - w^{(t-1)}$.

The server computes two distinct pseudo-gradients:
1. **Volumetric Pseudo-Gradient:** $v_{vol}^{(t)} = \sum_{k=1}^N p_k \Delta w_k^{(t)}$
2. **Semantic Pseudo-Gradient:** $v_{sem}^{(t)} = \sum_{k=1}^N s_k^{(t)} \Delta w_k^{(t)}$

**Adaptive Trade-off ($\lambda^{(t)}$):** 
To prevent the semantic gradient from destabilizing the base representations, $\lambda^{(t)}$ is adjusted based on the cosine similarity between the two pseudo-gradients:
$$ \lambda^{(t)} = \lambda_{max} \cdot \left( 1 - \max\left(0, \frac{\langle v_{vol}^{(t)}, v_{sem}^{(t)} \rangle}{\|v_{vol}^{(t)}\| \|v_{sem}^{(t)}\|} \right) \right) $$
If the semantic update strongly diverges from the volumetric update (indicating the discovery of novel threats), $\lambda^{(t)}$ increases to prioritize the semantic shift. When gradients agree ($\cos \approx 1$), semantic influence dynamically decreases.

**Final Global Update Rule:**
$$ w^{(t)} = w^{(t-1)} + \eta_{server} \left[ (1 - \lambda^{(t)}) v_{vol}^{(t)} + \lambda^{(t)} v_{sem}^{(t)} \right] $$
where $\eta_{server}$ is the server learning rate. The inclusion of a server learning rate resembles adaptive federated optimizers (like FedOpt/FedAdam) and allows controlled scaling of the pseudo-gradient step.

## 5. Convergence Discussion

ASFO is designed to converge under standard non-convex federated assumptions (L-smoothness and bounded variance). 
- **Stability:** By bounding $\lambda_{max} < 0.5$, the algorithm guarantees that the Volumetric Pseudo-Gradient remains the primary driver of representation learning, preventing catastrophic forgetting of benign traffic patterns.
- **Bounded Gradients:** The Softmax transformation of $s_k^{(t)}$ and the cosine similarity scaling of $\lambda^{(t)}$ strictly bound the magnitude of the semantic divergence term, ensuring the update step size does not violate Lipschitz continuity constraints.
- **Semantic Enhancement:** ASFO increases the contribution of semantically important rare classes through adaptive semantic weighting, counteracting the dominant gradient magnitude of majority classes in highly non-IID federated domains.

## 6. Computational Complexity

- **Client Complexity:** $O(E \cdot |D_k|)$ (Identical to standard FedAvg, requiring no extra computation).
- **Server Aggregation:** $O(N \cdot d)$ where $N$ is the number of clients and $d$ is the number of model parameters. The interpolation step requires two accumulations instead of one.
- **Server KG Complexity:** Assuming precomputed impact scores, semantic scoring is $O(N \cdot |C|)$ where $|C|$ is the number of attack classes. Since $|C| \ll d$, the semantic scoring overhead is $O(1)$ relative to the model size.

## 7. Algorithm Pseudocode

```text
Algorithm 1: Adaptive Semantic Federated Optimization (ASFO)
Input: Initial model w_0, Total rounds T, Server learning rate η, 
       Max semantic trade-off λ_max, Temperature β, Knowledge Graph K

Initialize global rarity counts n_c = 0 for all classes c

For round t = 1 to T do:
    Server broadcasts w_{t-1} to N clients
    
    // Client-side Execution (Standard)
    For each client k in 1..N in parallel do:
        w_k^(t) ← Local_Training(w_{t-1}, D_k)
        Δw_k^(t) ← w_k^(t) - w_{t-1}
        f_k ← Compute_Class_Distribution(D_k)
        Send (Δw_k^(t), f_k) to Server
    End For
    
    // Server-side Aggregation (ASFO)
    I ← Retrieve_Precomputed_Impacts(K)
    
    For each client k:
        // Update global rarity vector
        n_c ← n_c + f_k_c  (for all c)
        
    R_c ← 1 / (ε + n_c)    (for all c)
    
    For k = 1 to N do:
        f_hat_k ← f_k / ||f_k||_1
        SRS_k ← f_hat_k · (I ⊙ R)
        s_k ← exp(β * SRS_k) / sum_j(exp(β * SRS_j))
        p_k ← |D_k| / sum_j(|D_j|)
    End For
    
    v_vol ← sum_{k=1}^N (p_k * Δw_k^(t))
    v_sem ← sum_{k=1}^N (s_k * Δw_k^(t))
    
    cos_sim ← dot(v_vol, v_sem) / (norm(v_vol) * norm(v_sem))
    λ ← λ_max * (1 - max(0, cos_sim))
    
    w_t ← w_{t-1} + η * [ (1 - λ) * v_vol + λ * v_sem ]
End For

Return w_T
```

## 8. Ablation Plan
To isolate the contributions of ASFO for peer review, the experimental pipeline will support the following ablation matrix:

| Experiment | Purpose |
| :--- | :--- |
| **FedAvg** | Baseline pure volumetric aggregation |
| **FedProx** | Local regularization penalty for non-IID data |
| **FedNova** | Normalized volumetric aggregation accounting for local steps |
| **ASFO-NoKG** | ASFO without semantic impact scores ($I_c = 1$) to isolate KG value |
| **ASFO-Static** | ASFO with fixed static trade-off (fixed $\lambda$, adaptive disabled) |
| **ASFO-Adaptive** | ASFO with dynamic $\lambda$ but without KG impact to isolate adaptive scaling |
| **ASFO-Full** | The complete algorithm utilizing KG Impact, Rarity, and Adaptive $\lambda$ |
