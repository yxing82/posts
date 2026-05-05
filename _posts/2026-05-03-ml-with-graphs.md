---
title: "Learning Notes on Machine Learning with Graphs (Updating)"
date: 2026-05-03
categories: [Notes, Machine Learning]
tags: [graphs, ml]
math: true
---

> **Course:** [Stanford CS224W (Winter 2021)](https://www.youtube.com/playlist?list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn)

> **Instructor:** Prof. Jure Leskovec

> **Lectures Covered:** 1.1 – 3.2 

<br>

# 1. Introduction to Graph ML

<br>

## 1.1 Why Graphs?

Many real-world data naturally live on graphs, simply, entities connected by relationships. Rather than treating data points as isolated, graphs let us model the rich structure of interactions between them.

**Key motivation:** Complex systems in biology, society, technology, and science can all be described as networks of interconnected entities. Graphs give us a universal mathematical language to reason about these systems.

**Examples of networks:**
- Social networks (friendships, followers)
- Communication networks (emails, phone calls)
- Biological networks (protein interactions, gene regulation)
- Information networks (the Web, citation graphs)
- Infrastructure networks (roads, power grids)

**Why ML on graphs?** Traditional ML assumes data points are independent (i.i.d.), but graph-structured data violates this — nodes are linked and influence each other. Graph ML lets us exploit relational structure to make better predictions.

**Types of tasks we can perform:**
- **Node-level:** Classify or predict properties of individual nodes (e.g., categorise a user)
- **Link-level:** Predict whether a link exists or will form between two nodes (e.g., friend recommendation)
- **Graph-level:** Classify or predict properties of entire graphs (e.g., predict molecular properties)



## 1.2 Choice of Graph Representation

Before doing ML, we need to decide *how* to represent a graph. 

**Components of a graph G = (V, E):**
- **V** — set of nodes (vertices)
- **E** — set of edges (links)

**Types of graphs:**

| Property | Options |
|---|---|
| Edge direction | Undirected vs. Directed |
| Edge weight | Unweighted vs. Weighted |
| Self-loops | Allowed or not |
| Multigraph | Single edge vs. multiple edges between same pair |
| Bipartite | Nodes split into two disjoint sets, edges only between sets |

**Adjacency matrix $A$:** An N × N matrix where $A_{i,j} = 1$ if there is an edge from node i to node j. 

For undirected graphs, A is symmetric. i.e. $A_{i,j} = A_{j,i}$ and $A_{i,i} = 0$

Real-world graphs are typically *sparse* (most entries are 0), so the adjacency matrix is mostly zeros.

**Other representations:**
- **Edge list:** List of (source, target) pairs, compact for sparse graphs.
- **Adjacency list:** For each node, store a list of its neighbours.

**Key properties:**
- **Node degree $k_{i}$:** number of edges touching node i. For directed graphs, we distinguish in-degree and out-degree.
- Average degree of an undirected graph:
$$\bar{k} = \frac{2|E|}{|V|}$$
- **Connectivity:** a graph is connected if there is a path between every pair of nodes. 
    - a directed graph is **strongly connected** if there is a valid path from every node to every other node, strictly following the direction of the edges.
    - a directed graph is **weakly connected** if it is not strongly connected, but it would be connected if treating all one-way arrows as two-way lines.

<br>
<br>
<br>

# 2. Traditional Feature-based Methods

The traditional ML pipeline on graphs has two steps:
1. Hand-design features that describe nodes, links, graphs;
2. Feed these features into a standard ML model (logistic regression, SVM, random forest, etc.). 

The notes below cover what features to design at each level.

> For simplicity, all discussion below assumes **undirected graphs** unless stated otherwise.

<br>

## 2.1 Node-Level Features

**Goal:** Characterise the structure and position of a single node in the network.

### 1. Importance-based features

Measures how important the node is.

**i) Node Degree (Degree Centrality)** counting the number of edges the node has.

A node is important if the node has more connections, meaning it's critcial to the flow of the network.

$$k_v = |N(v)|$$

, where $N(v)$ is the set of neighbours of the node $v$. 

Limitation: treats all neighbours equally.

**ii) Node Centrality** measuring how *important* a node is within the graph. 

*   **Eigenvector Centrality**
    
    A node is important if its neighbours are important (recursive definition).

    $$c_v = \frac{1}{\lambda} \sum_{u \in N(v)} c_u$$

    This leads to the eigenvector equation **Ac = λc**. We take the eigenvector corresponding to the largest eigenvalue $λ_{max}$.

*   **Betweenness Centrality**
    
    A node is important if it lies on many shortest paths between other nodes.

    $$c_v = \sum_{s \neq v \neq t} \frac{\text{# shortest paths between } s \text{ and } t \text{ that pass through } v}{\text{# shortest paths between } s \text{ and } t}$$

    Captures "bridge" nodes.

*   **Closeness Centrality**

    A node is important if it has short shortest-path distances to all other nodes.

    $$c_v = \frac{1}{\sum_{u \neq v} d(v, u)}$$

    , where d(v, u) is the shortest path length from v to u. 
    
    High closeness = the node can reach everyone quickly.

### 2. Structure-based features

Captures *topological properties* of the graph, and how nodes are embedded with their local neighbours, regardless of what the nodes actually represent.

**i) Node Degree** defines the exact size of a node's immediate neighbours (1-hop ego-graph).

Distinguishes topological roles, e.g. the node is a "central hub", or a "bridge", or an "isolated leaf"


**ii) (Local) Clustering Coefficient** measuring how connected a node's neighbours are to each other. 

The fundamental building block is the **triangle**. If node $v$ is connected to both $u$ and $w$, a triangle is **"closed"** when $u$ and $w$ are also connected. Then the **(local) Clustering Coefficient** measures what fraction of a node's neighbour pairs actually **close the triangle**:

$$
\begin{aligned}
e_{v} &= \frac{\text{# edges among neighbours of } v}{\text{# node pairs among neighbours of } v} \\[1em]
&= \frac{\text{# triangles}}{\binom{k_v}{2}} \in [0, 1]
\end{aligned}
$$

, where $k_{v}$ is the degree of $v$. 

Ranges from 0 (no edges among neighbours) to 1 (neighbours form a complete clique).

*   **Problems with Bipartite Graphs:** 
    *   Triangles can NEVER form: if $u_{1} \in U$ connects to $v_{1} \in V$ and $v_{2} \in V$, the edge $(v_{1}, v_{2})$ cannot exist because both are in the same set $V$.

        The standard Clustering Coefficient is then always **zero** for every node in a bipartite graph

*   **Bipartite Clustering Coefficient (4-Cycles / Squares):** counts 4-cycle (square) as a closed path $u_{1} \rightarrow v_{1} \rightarrow u_{2} \rightarrow v_{2} \rightarrow u_{1}$, where $u_{1}, u_{2} \in U$ and $v_{1}, v_{2} \in V$. 

$$cc_v = \frac{\text{# closed 4-cycles through } v}{\text{# possible 4-cycles through } v}$$

*   
    | Graph type | Smallest cycle | Clustering measures |
    |---|---|---|
    | General (unipartite) | Triangle (3-cycle) | Standard clustering coefficient |
    | Bipartite | Square (4-cycle) | Bipartite clustering coefficient |



**iii) Graphlet Degree Vector (GDV):** A vector that counts, for each graphlet position, how many times node $v$ appears in that position, generalising the notion of counting triangles in the **Clustering Coefficient** by counting *all* small subgraph patterns (graphlets) that a node participates in. 

This provides a rich, detailed signature of a node's local network topology.

*   **Graphlets** are small, *rooted connected*, non-isomorphic subgraphs. 

*   A **single graphlet** has one specific, fixed graph structure.

*   **Non-isomorphic** means for every graphlet in the whole dictionary has a completely *unique* topological shape from all others.



### Comparison of node features:

| Feature | What it captures |
|---|---|
| Degree | How many connections |
| Centrality | Importance / position in global structure |
| Clustering coefficient | Local triangle density |
| GDV | Detailed local topology (generalises clustering coeff.) |


<br>

## 2.2 Link-Level Features

**Goal:** Design features for a *pair* of nodes $(v_{1}, v_{2})$ to predict whether a link exists or will form between them.

### 1. Distance-Based Features

Use the **shortest-path distance** between $v_{1}$ and $v_{2}$, meaning how likely nodes can be connected.

Limitation: this only captures path length, not the *richness* of the connection (degree information, e.g. two nodes at distance 2 might share 1 or 100 common neighbours).

### 2. Local Neighbourhood Overlap

Captures how many neighbours are shared between two nodes. More overlap means more likely to be linked.

**i) Common Neighbours** simply counts how many nodes are neighbours of both $v_{1}$ and $v_{2}$.

$$|N(v_1) \cap N(v_2)|$$


**ii) Jaccard Coefficient** normalises common neighbours by the total size of both neighbourhoods. Gives a proportion rather than a raw count.

$$\frac{|N(v_1) \cap N(v_2)|}{|N(v_1) \cup N(v_2)|}$$


**iii) Adamic-Adar Index** weights each common neighbour by the inverse log of its degree. 

$$\sum_{u \in N(v_1) \cap N(v_2)} \frac{1}{\log(k_u)}$$

The intuition: a shared neighbour who is very popular (high degree) is less meaningful than one with few connections. If two people both know a celebrity, that's less significant than if they both know the same niche researcher.

**Limitation of local features:** If two nodes have no common neighbours, all these features are zero, even if the nodes are structurally very similar or destined to connect.

### 3. Global Neighbourhood Overlap

Considers the entire graph, not just immediate neighbourhoods.

**Katz Index** counts the number of paths of *all lengths* between two nodes, with shorter paths weighted more heavily:

$$S_{v_1, v_2} = \sum_{l=1}^{\infty} \beta^l \cdot \mathbf{A}^l_{v_1, v_2}$$

, where:
- **$A$** is the adjacency matrix
- **$A^l$** gives the number of paths of length l between nodes
- **β ∈ (0, 1)** is a discount factor that exponentially downweights longer paths

In closed form: **S = (I − βA)⁻¹ − I**.

To break down this complicated form, let $P^{(k)}_{uv} = \text{\# paths of length k between nodes u and v}$,

*   Given adjacency matrix $A$, we can directly have $P^{(1)}_{uv} = A_{uv}$.

*   For length 2, we multiply the adjacency matrix $A$ with itself, and derive $P^{(2)}_{uv} = A_{uv} * A_{uv} = A^{2}_{uv}$.

    $$(A^2)_{uv} = \sum_{w=1}^{|V|} A_{uw} A_{wv}$$
    
    *   $A_{uw}$: Is there an edge from node $u$ to an intermediate node $w$? (1 if yes, 0 if no).
    *   $A_{wv}$: Is there an edge from that same intermediate node $w$ to the destination node $v$? (1 if yes, 0 if no).
    *   $\sum$: By summing over every possible intermediate node $w$ in the entire graph (from $1$ to $\vert V \vert$), we count how many two-step paths exist between $u$ and $v$.




The Katz index resolves the limitation of local methods by capturing indirect connections through the entire network.

### Summary of link features:

| Feature type | Examples | Scope |
|---|---|---|
| Distance-based | Shortest path | Global but coarse |
| Local overlap | Common neighbours, Jaccard, Adamic-Adar | Local (immediate neighbours only) |
| Global overlap | Katz index | Global (all paths) |

<br>

## 2.3 Graph-Level Features

**Goal:** Create a feature vector that describes an *entire graph*, enabling graph-level classification.

### Kernel Methods

Instead of explicitly designing a feature vector $\phi(G)$, we can define a **graph kernel** $K(G, G')$ that measures the similarity between two graphs. By the kernel trick, there implicitly exists some feature map $\phi$ such that:

$$K(G, G') = \phi(G)^T \phi(G')$$

This lets us use kernel-based ML methods (SVM, etc.) without ever computing $\phi$ explicitly.

**Key idea:** **Bag-of-\*** represents a graph by counts of certain substructures, ignoring order.

### 1. Graphlet Kernel

Represent a graph by counting the number of each type of **graphlet** it contains.

Steps:
1. Enumerate all graphlets up to $k$ (e.g., $k$ = 3, 4, or 5 nodes).
2. For each graphlet type, count how many times it appears as a subgraph of $G$.
3. This count vector is the graph's feature vector.

The definition of **graphlets at the graph-level** do **not** need to be connected and are **not** rooted at a specific node, which is different from node-level graphlets.

**Normalisation:** If comparing graphs of different sizes, normalise the count vectors (e.g., by the total count) so that large graphs don't dominate.

**Limitation:** Counting graphlets is computationally expensive.

### 2. Weisfeiler-Lehman (WL) Kernel

Use an iterative neighbourhood aggregation (**colour refinement**) scheme to build up a vocabulary of "colours" that encode local structure, then count these colours. This is a much more efficient alternative to the graphlet kernel.

**Colour Refinement Algorithm:**

1. **Initialise:** Assign every node the same colour (or use node labels if available).
2. **Iterate:** For each node, create a new colour by **hashing** together its current colour and the *multiset* of colours of its neighbours.
3. **Repeat** for K iterations. After K steps, each node's colour summarises the structure of its K-hop neighbourhood.

After colour refinement, represent each graph as a count vector of how many nodes have each colour. The WL kernel between two graphs is then the dot product of their colour count vectors.

**Why this works:** The colour refinement process is essentially a generalisation of counting node degrees:
- After 1 iteration, a node's colour encodes its degree (1-hop info).
- After 2 iterations, it encodes the degrees of its neighbours (2-hop info).
- After K iterations, it encodes the full K-hop neighbourhood structure.

**Advantages over graphlet kernel:**
- Computationally efficient: each iteration is linear in the number of edges.
- Only requires K iterations (typically small, e.g., 3–5).
- Captures increasingly rich structural information with each iteration.

**Other graph kernels** (mentioned but not covered in detail):
- Random walk kernel
- Shortest-path graph kernel

<br>
<br>
<br>

# Summary: The Traditional ML Pipeline on Graphs

**What covered:**

| Level | Features | Key Concepts |
|---|---|---|
| **Node** | Degree, centrality, clustering coeff., GDV | Capture importance and local topology |
| **Link** | Shortest path, common neighbours, Jaccard, Adamic-Adar, Katz | Capture pairwise proximity and overlap |
| **Graph** | Graphlet kernel, WL kernel | Capture global substructure patterns |

**Key limitation of traditional methods:** All features are hand-designed. This requires domain knowledge and doesn't automatically learn representations from data. 

The rest of the course (Lectures 3+) addresses this by introducing methods that *learn* features automatically, starting with node embeddings and moving to graph neural networks.

<br>
<br>
<br>

# 3. Node Embeddings: Foundations

<br>

## 3.1 Problem Definition

Given an undirected graph $G = (V, E)$ with adjacency matrix $A$ (binary, no node features), learn a mapping function $f: u \rightarrow \mathbb{R}^{d}$ that assigns each node a low-dimentional vector $\mathbf{z}_{u}$ such that:

$$
\text{similarity}(u, v) \approx \mathbf{z}_{u}^\top \mathbf{z}_{v}
$$

Nodes that are **"similar" in the graph** should be **close in embedding space**.

**Key Questions:**

- How to **define similarity**?

- How to learn the **mapping $f$**?

**Properties:**

- **Unsupervised / Semi-supervised:** 

    No node labels or features are utilised.

    The goal is to directly estimate a set of coordinates (embedding $\mathbf{z}_{u}$) for each node to capture network structure by the decoder (DEC).

    The graph topology itself is the only supervision signal.

- **Task-independent:** 

    Embeddings are trained to preserve general structural properties.

    Not optimised for s specific prediction target.

    The same embeddings can be reused across different downstream tasks without retraining (e.g. node classification, link prediction).

<br>

## 3.2 The Encoder-Decoder Framework

Every node embedding method is an instance of this framework:

| Component | What it does | Typical choice |
|---|---|---|
| **Encoder** | $\text{ENC}(v) = \mathbf{z}_v$ <br> maps node → vector (embedding) | Shallow lookup or GNN |
| **Decoder** | $\text{DEC}(\mathbf{z}_u, \mathbf{z}_v)$ <br> maps vector pair (embeddings) → similarity score | Dot product $\mathbf{z}_u^\top\mathbf{z}_v$ |
| **Similarity function** | Ground-truth notion of "similar" in the graph | Adjacency, co-occurrence, random walk probability |
| **Objective** | Force decoder output to match the similarity function | Maximise likelihood; <br> minimise loss |


### Shallow Encoding

DeepWalk and Node2vec both use a **direct lookup table** as the simplest possible encoder:

$$
\text{ENC}(v) = \mathbf{Z} \cdot \mathbf{v}
$$

*   $\mathbf{Z} \in \mathbb{R}^{d \times \vert V \vert}$: Embedding matrix. Each column is a pspecific node's embedding vector. $d$ is the dimension of embeddings.

*   $\mathbf{v} \in \mathbb{I}^{\vert V \vert}$: One-hot indicator for node $v$

> In some cases, each row in the embedding matrix can be represented as an embedding vector. 

The embedding matrix is what we learn. Every node is assigned a unique, independent embedding vector. Here, no parameter sharing, no generalisation to unseen nodes.

<br>

## 3.3 Defining Similarity via Random Walks

For both DeepWalk and Node2vec, we define similarity as the probability that 2 nodes co-occur on short random walks. Specifically, 

$$
\mathbf{z}_{u}^\top \mathbf{z}_{v} \approx P(\text{u and v co-occur on a random walk})
$$

- **Expressivity:** walks naturally capture multi-hop, higher-order relationships; not just direct edges.

- **Efficiency:** only train pairs that actually co-occur in the walks, not all $O(\vert V \vert^{2})$ pairs.

- **Flexibility:** changing the walk strategy (§3.4) changes what "similar" means.

<br>

## 3.4 Walk Strategy 

DeepWalk and Node2vec are differeciated by walk strategies.

The **walk strategy $R$** determines the **neighbourhood $N_{R}(u)$**, the multiset of nodes visited on walks starting from node $u$.

> "multiset" means we allow the same node appears more than once in the list.

Two hyperparameters control the volume of walk data generated for all walk-based methods:

- **Number of Walks $r$:** dictates number of independent random walks starting from node $u$. It controls the sampling density of the local neighbourhood.
    - if $r = 10$, the algorithm will go back to the starting node $u$ 10 separate times to begin a fresh walk.

- **Length of Walks $l$:** number of hops (edges traversed) the walker takes before the sequence terminates.
    - if $l = 5$, the walker steps to an adjacent node for 5 times.

Together, $r \times l$ determines the size of $N_R(u)$ for each node.

### Uniform Random Walks (DeepWalk)

- **1st-Order Markov walk:** The probability of moving to the next node depends *only* on the current node. It has no memory of how it got there.

- **Limitation:** Blends local and global structure wihout any control over which one dominates


### Biased Random Walks (Node2vec)

- **2nd-Order Markov Walk:** The probability of moving to the next node depends on the current node **and** the immediately preceding node. It has a memory of exactly one previous step.

- Controlled by two paramaters: **Return parameter** $p$ and **In-out parameter** $q$.

**Setup:**

The walk just moved from $v \rightarrow u$ and is choosing the next node $x$ among $u$'s neighbours.

The unnoramlised transition weight depends on the shortest-path distance $d_{vx}$ between the previous node $v$ and the candidate node $x$:

| $d_{vx}$ | Situation | Weight | Effect |
|---|---|---|---|
| 0 | $x = v$ <br> (backtrack to previous) | $1/p$ | **Return parameter $p$** <br> controls backtracking |
| 1 | $x$ is a neighbour of both $u$ and $v$ | $1$ | Baseline <br> stay in local cluster |
| 2 | $x$ is further from $u$ | $1/q$ | **In-out parameter $q$** <br> controls outward exploration |

**Reading $p$ and $q$:** Which nodes end up in the same walk together?

| Setting | Walk behaviour | Exploration type | Captures |
|---|---|---|---|
| Low $p$ | Frequently backtracks | Stays very local | Local structural patterns |
| High $p$ | Avoids revisiting | Pushes outward | Broader exploration |
| Low $q$ <br> (< 1) | Moves away from previous node | **Depth-First Search** <br> ventures far | **Homophily** <br> (outward exploration) |
| High $q$ <br> (> 1) | Stays close to previous node | **Breadth-First Search** <br> stays local | **Structural Equivalence** <br> (local exploration) |

- DFS-like deep walk (low $q$) plunges deep into the graph. 

- BFS-like deep walk (high $q$ or low $p$) circles the local neighbourhood.

 - DeepWalk is the special case, where $p = q = 1$ as uniform walk.

 -  Essentially, $q$ is the "ratio" of BFS vs. DFS 


### BFS vs. DFS: Two Views of the Graph 

Given BFS and DFS, the same graph has two fundamentally different notions of node similarity, selected by the walk strategy:

| | BFS neighbourhood (inward) | DFS neighbourhood (outward) |
|---|---|---|
| **What it samples** | Immediate neighbours of $u$ | Nodes at increasing distance from $u$ |
| **View** | Local Microscopic | Global Macroscopic |
| **Similarity notion** | **Structural equivalence** <br> two hub nodes in **distant** parts of the graph | **Homophily** <br> nodes in the same community or **cluster** |
| **Why it works** | Characterising structural role only requires accurate local topology | Detecting community membership requires seeing the broader network |

In Node2vec, $p$ and $q$ **continuously interpolate** between two extremes within a single algorithm.

<br>

## 3.5 Optimisation

All random-walk embedding methods share the **same optimisation pipeline**. The **differences** between random-walk embedding methods live *entirely* in "How walks are generated" (§3.4).

The walk prediction task is purely a **pretext task** to force the embbedding vectors into a configuration that captures network structure. Once training is done, we throw away the objective and keep **only** the **learned embedding matrix $Z$** for down stream tasks. i.e. Optimising random walk prediction is *NOT* to build an accurate walk predictor. We dont care about the accuracy of the walk predictions themselves.

### Maximum Likelihood Formulation

Treat the random walk neighbourhoods $N_{R}(u)$ as **observed data**, and the embeddings $f$ mapping to $\mathbf{z}_{u}$ as **parameters**.

The goal is to find embeddings that maximise the likelihood of the observed co-occurances:

$$
\max_{\mathbf{f}} \sum_{u \in V} \log P\bigl(N_R(u) \mid \mathbf{z}_u\bigr)
$$

> In the shallow encoding setup, $f$ is just the lookup table, so the actual learnable parameters are the embedding vectors $\mathbf{z}_{u}$ themselves (i.e. the columns of the embedding matrix $Z$). <br> In this case, $f$ is equivalent to specifying $Z$.

We assume that predicting each neighbour $v \in N_{R}(u)$ is conditionally independent given $\mathbf{z}_{u}$. This lets us decompose the joint probability into a product over individual neighbours:

$$
P(N_{R}(u) \mid \mathbf{z}_{u}) = \prod_{v \in N_{R}(u)} P(v \mid \mathbf{z}_{u})
$$

Taking the log converts the product into a sum:

$$
\log(\prod_{v \in N_{R}(u)} P(v \mid \mathbf{z}_{u})) = \sum_{v \in N_{R}(u)} \log(P(v \mid \mathbf{z}_{u}))
$$

Converting from **maximising the positive** log-likelihood to **minimising the negative** one gives us the **Loss**:

$$
\mathcal{L} = \sum_{u \in V} \sum_{v \in N_{R}(u)} -\log(P(v \mid \mathbf{z}_{u}))
$$

### Softmax Parameterisation

To turn dot products into probabilities, for each individual $P(v \mid \mathbf{z}_{u})$, we use a **softmax** over all nodes:

$$
P(v \mid \mathbf{z}_{u}) = \frac{\exp (\mathbf{z}_{u}^\top \mathbf{z}_{u})}{\sum_{n \in V} \exp (\mathbf{z}_{u}^\top \mathbf{z}_{n})}
$$

, where:

- $\mathbf{z}_{u}^\top \mathbf{z}_{u}$ is the raw similarity score between node $u$ and $v$.

- $\exp (\mathbf{z}_{u}^\top \mathbf{z}_{u})$ makes the similarity score *strictly positive* to generate the probability output.

- $\sum_{n \in V} \exp (\mathbf{z}_{u}^\top \mathbf{z}_{n})$ calculate similarity scores between node $u$ and every other node in the graph. This ensures the output is a clean percentage in $(0, 1)$.

- $P(v \mid \mathbf{z}_{u})$ sums to 1 and all entries are positive, making it a valid probability distribution.

- $P(v \mid \mathbf{z}_{u})$ is a monotonic function, where higher $\mathbf{z}_{u}^\top \mathbf{z}_{n}$ implies higher $P(v \mid \mathbf{z}_{u})$.


**Problem:** the denominator requires summing over every node pair in the graph. For each training pair $(u, v)$, this is $O(\vert V \vert)$, and across all pairs it becomes $O(\vert V \vert)^{2}$. Intractable for large graphs.

**Solution:** Negative Sampling.


### Negative Sampling 

Instead of asking "is node $v$ the most similar node to $u$, out of all $\vert V \vert$ nodes?" (Softmax), we ask "can we distinguish the real neighbour $v$ from $K$ random imposters?" (**Binary Cross-Entropy Approximation**).

We approximate the softmax loss with:

$$
-\log (P(v \mid \mathbf{z}_{u})) \approx \underbrace{-\log(\sigma(\mathbf{z}_u^\top\mathbf{z}_v))}_{\text{push positive pair together}} - \sum_{i=1}^{K}\underbrace{\log(\sigma(-\mathbf{z}_u^\top\mathbf{z}_{n_i}))}_{\text{push negative pairs apart}}, \quad n_i \sim P_V
$$

, where $\sigma$ is the sigmoid function, converting similarity scores into probabilities.

In terms of "Binary":
- the 1st term **rewards** the model for scoring real co-occurrences highly (positive sample)
- the 2nd term **penalises** it for scoring random pairs highly (negative sample)
Together, the model learns to discriminate real walk neighbours from random noise 

| Detail | Value / rule |
|---|---|
| $K$ (number of negatives per positive) | 5–20 for small datatset; 2-5 for large dataset. <br> Picked completely random, ignoring edges. |
| Sampling distribution $P_V$ | Proportional to node degree |
| Complexity per training pair | $O(K)$ instead of $O(\vert V \vert)$ |


### Training Pipeline

Here is the concrete procedure for learning the embedding matrix $\mathbf{Z}$:

1. **Initialise** the embedding lookup table $\mathbf{Z} \in \mathbb{R}^{d \times \vert V \vert}$ with small random values. Each column is a node's initial embedding.

2. **Sample walks** form all nodes using the chosen walk strategy $R$, and collect co-occurring pairs $(u, v)$.

3. **For each positive pair** $(u,v)$:
    - Look up $\mathbf{z}_{u}$ and $\mathbf{z}_{v}$ from the embedding matrix;
    - Sample $K$ neagtive nodes $n_1, ..., n_K$ proportional to node degree, so the probability of picking node $n$ as a negative is:
    $$
    P(n) = \frac{\text{deg}(n)}{\sum_{m \in V} \text{deg}(m)}
    $$
    - Compute the negative-sampling loss for this pair

4. **Backpropagate** through the loss to compute gradients $\frac{\partial\mathcal{L}}{\partial\mathbf{z}_u}$, $\frac{\partial\mathcal{L}}{\partial\mathbf{z}_v}$, $\frac{\partial\mathcal{L}}{\partial\mathbf{z}_{n_i}}$.

5. **Update** the corresponding columns of $\mathbf{Z}$ via stochastic gradient descent (SGD), where $\eta$ is the learning rate (i.e. Overwrite random numbers in the embeddings with better numbers to make entries as probabilities we expected):

$$
\mathbf{z}_u \leftarrow \mathbf{z}_u - (\eta \frac{\partial\mathcal{L}}{\partial\mathbf{z}_u})
$$


6. **Iterate** over millions of node pairs. 
    - Sample **true** node pairs $\rightarrow$ Calculate the dot product and squash it with the sigmoid function to get the probability $p$ $\rightarrow$ Get the error between the target and the function result $\rightarrow$ Update in the embedding matrix $\mathbf{Z}$ for both nodes
        - Visual result: true nodes are **pulled together** in the embedding space 
        - Their dot products increase for the next time 

    - Sample **fake** node pairs $\rightarrow$ Calculate the dot product and squash it with the sigmoid function to get the probability $p$ $\rightarrow$ Get the error between the target and the function result $\rightarrow$ Update in the embedding matrix $\mathbf{Z}$ for both nodes
        - Visual result: fake nodes are **pushes apart** in the embedding space 
        - Their dot products decrease for the next time 

7. **Convergence:** continue until the loss stops decreasing meaningfully. At this point, the embedding matrix $\mathbf{Z}$ has stabilised.

8. **Stop the training loop and Keep $\mathbf{Z}$**. Discard the objective, the walks, and the negative sampler. The columns of $\mathbf{Z}$ are the final node embeddings, ready for any downstream tasks.

<br>

## 3.6 Full Algorithm Summary

| Step | What happens | Shared or method-specific? |
|---|---|---|
| 1. **Preprocess** | Compute biased transition probabilities (Node2vec) or skip (DeepWalk) | Method-specific |
| 2. **Walk** | Simulate $r$ walks of length $l$ per node using strategy $R$ | Method-specific (uniform vs. biased) |
| 3. **Collect** | Build $N_R(u)$ for each node from walk co-occurrences | Shared |
| 4. **Optimise** | SGD and Negative Sampling on the log-likelihood objective | Shared |
| 5. **Use** | Plug $\mathbf{z}_u$ vectors into any downstream task | Shared |


<br>

## 3.7 Method Comparison

### DeepWalk vs. Node2Vec

| | DeepWalk | Node2Vec |
|---|---|---|
| Walk type | Uniform, 1st-order | Biased, 2nd-order |
| Parameters | Walk length $l$, walks per node $r$ | $l$, $r$, **return $p$, in-out $q$** |
| Neighbourhood control | None, <br> takes what the walk gives | Full, <br> interpolates BFS ↔ DFS |
| Homophily | Captured implicitly | Explicitly tunable (low $q$) |
| Structural equivalence | Weak | Explicitly tunable (high $q$) |
| Relationship | Special case of node2vec <br> ($p=q=1$) | Generalisation of DeepWalk |


### Homophily vs. Structural Equivalence

| | Homophily | Structural Equivalence |
|---|---|---|
| Question asked | "Are $u$ and $v$ in the same cluster?" | "Do $u$ and $v$ play the same role?" |
| Nodes are similar if | They are close / densely connected | They have similar local topology (even if far apart) |
| Walk strategy | DFS-like (low $q$) <br> explores broadly | BFS-like (high $q$) <br> characterises local structure |
| Example | Two members of the same friend group | Two "bridge" nodes in different parts of the network |


### Full Softmax vs. Negative Sampling

| | Full Softmax | Negative Sampling |
|---|---|---|
| Formulation | **Multi-class** classification over $\vert V \vert$ nodes | **Binary** classification (true pair vs. fake pair) |
| Normalisation scope | All $\vert V \vert$ nodes | $K$ sampled negatives |
| Loss type | Cross-entropy over $\vert V \vert$ classes | Binary cross-entropy approximation |
| Cost per pair | $O(\vert V \vert)$ | $O(K)$ |
| Accuracy | Exact MLE | Approximation (NCE) |
| Scalability | Intractable for large graphs | Practical at scale |

### Shallow Encoding vs. Deep Encoding (GNNs)

| | Shallow (DeepWalk, node2vec) | Deep (GNNs) |
|---|---|---|
| Encoder | Lookup table with one vector per node | Neural network over neighbour features |
| Parameter sharing | None | Shared weights across all nodes |
| Handles unseen nodes? | No (transductive only) | Yes (inductive) |
| Uses node features? | No | Yes |

<br>

## 3.8 Key Equations 

| Concept | Equation |
|---|---|
| Embedding goal | $\text{similarity}(u,v) \approx \mathbf{z}_u^\top\mathbf{z}_v$ |
| Shallow encoder | $\text{ENC}(v) = \mathbf{Z} \cdot \mathbf{v}$ |
| MLE objective | $\max_{\mathbf{f}} \sum_{u} \log P(N_R(u) \mid \mathbf{z}_u)$ |
| Independence assumption | $P(N_R(u) \mid \mathbf{z}_u) = \prod_{v \in N_R(u)} P(v \mid \mathbf{z}_u)$ |
| Loss (negative log-likelihood) | $\mathcal{L} = \sum_{u}\sum_{v \in N_R(u)} -\log (P(v \mid \mathbf{z}_u))$ |
| Softmax (multi-class) | $P(v \mid \mathbf{z}_u) = \frac{\exp(\mathbf{z}_u^\top\mathbf{z}_v)}{\sum_n\exp(\mathbf{z}_u^\top\mathbf{z}_n)}$ |
| Negative sampling (binary) | $\approx -\log(\sigma(\mathbf{z}_u^\top\mathbf{z}_v)) - \sum_{i=1}^k\log(\sigma(-\mathbf{z}_u^\top\mathbf{z}_{n_i}))$ |
| Node2vec bias ($d_{ux}=0,1,2$) | Weights: $\frac{1}{p}$, $1$, $\frac{1}{q}$ |
