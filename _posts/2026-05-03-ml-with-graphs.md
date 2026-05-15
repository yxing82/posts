---
title: "Learning Notes on Machine Learning with Graphs (Updating)"
date: 2026-05-03
categories: [Notes, Machine Learning]
tags: [graphs, ml, random walk, node embedding, graph embedding, pagerank, node classification]
math: true
---

> **Course:** [Stanford CS224W (Winter 2021)](https://www.youtube.com/playlist?list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn)

> **Instructor:** Prof. Jure Leskovec

> **Lectures Covered:** 1.1 – 5.3

<br>

## 1. Introduction to Graph ML

<br>

### 1.1 Why Graphs?

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



### 1.2 Choice of Graph Representation

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

## 2. Traditional Feature-based Methods

The traditional ML pipeline on graphs has two steps:
1. Hand-design features that describe nodes, links, graphs;
2. Feed these features into a standard ML model (logistic regression, SVM, random forest, etc.). 

The notes below cover what features to design at each level.

> For simplicity, all discussion below assumes **undirected graphs** unless stated otherwise.
{: .prompt-tip }

<br>

### 2.1. Node-Level Features

**Goal:** Characterise the structure and position of a single node in the network.

### Importance-based features

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

### Structure-based features

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

> **Problems with Bipartite Graphs:** 
>
> Triangles can NEVER form: if $u_{1} \in U$ connects to $v_{1} \in V$ and $v_{2} \in V$, the edge $(v_{1}, v_{2})$ cannot exist because both are in the same set $V$.
>
> The standard Clustering Coefficient is then always **zero** for every node in a bipartite graph
> 
> **Bipartite Clustering Coefficient (4-Cycles / Squares):** counts 4-cycle (square) as a closed path $u_{1} \rightarrow v_{1} \rightarrow u_{2} \rightarrow v_{2} \rightarrow u_{1}$, where $u_{1}, u_{2} \in U$ and $v_{1}, v_{2} \in V$. 
>
> $$cc_v = \frac{\text{# closed 4-cycles through } v}{\text{# possible 4-cycles through } v}$$
>
> | Graph type | Smallest cycle | Clustering measures |
> |---|---|---|
> | General (unipartite) | Triangle (3-cycle) | Standard clustering coefficient |
> | Bipartite | Square (4-cycle) | Bipartite clustering coefficient |
{: .prompt-info }


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

### 2.2. Link-Level Features

**Goal:** Design features for a *pair* of nodes $(v_{1}, v_{2})$ to predict whether a link exists or will form between them.

### Distance-Based Features

Use the **shortest-path distance** between $v_{1}$ and $v_{2}$, meaning how likely nodes can be connected.

Limitation: this only captures path length, not the *richness* of the connection (degree information, e.g. two nodes at distance 2 might share 1 or 100 common neighbours).

### Local Neighbourhood Overlap

Captures how many neighbours are shared between two nodes. More overlap means more likely to be linked.

**i) Common Neighbours** simply counts how many nodes are neighbours of both $v_{1}$ and $v_{2}$.

$$|N(v_1) \cap N(v_2)|$$


**ii) Jaccard Coefficient** normalises common neighbours by the total size of both neighbourhoods. Gives a proportion rather than a raw count.

$$\frac{|N(v_1) \cap N(v_2)|}{|N(v_1) \cup N(v_2)|}$$


**iii) Adamic-Adar Index** weights each common neighbour by the inverse log of its degree. 

$$\sum_{u \in N(v_1) \cap N(v_2)} \frac{1}{\log(k_u)}$$

The intuition: a shared neighbour who is very popular (high degree) is less meaningful than one with few connections. If two people both know a celebrity, that's less significant than if they both know the same niche researcher.

**Limitation of local features:** If two nodes have no common neighbours, all these features are zero, even if the nodes are structurally very similar or destined to connect.

### Global Neighbourhood Overlap

Considers the entire graph, not just immediate neighbourhoods.

**Katz Index** counts the number of paths of *all lengths* between two nodes, with shorter paths weighted more heavily:

$$S_{v_1, v_2} = \sum_{l=1}^{\infty} \beta^l \cdot \mathbf{A}^l_{v_1, v_2}$$

, where:
- **$A$** is the adjacency matrix
- **$A^l$** gives the number of paths of length l between nodes
- **β ∈ (0, 1)** is a discount factor that exponentially downweights longer paths

In closed form: **S = (I − βA)⁻¹ − I**.

To break down this complicated form, let $P^{(k)}_{uv} = \text{\# paths of length k between nodes u and v}$,

*   Given adjacency matrix $A$, we can directly have $$P^{(1)}_{uv} = A_{uv}$$.

*   For length 2, we multiply the adjacency matrix $A$ with itself, and derive $$P^{(2)}_{uv} = A_{uv} * A_{uv} = A^{2}_{uv}$$.

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

### 2.3. Graph-Level Features

**Goal:** Create a feature vector that describes an *entire graph*, enabling graph-level classification.

### Introduction of Kernel Methods

Instead of explicitly designing a feature vector $\phi(G)$, we can define a **graph kernel** $K(G, G')$ that measures the similarity between two graphs. By the kernel trick, there implicitly exists some feature map $\phi$ such that:

$$K(G, G') = \phi(G)^T \phi(G')$$

This lets us use kernel-based ML methods (SVM, etc.) without ever computing $\phi$ explicitly.

**Key idea:** **Bag-of-\*** represents a graph by counts of certain substructures, ignoring order.

### Graphlet Kernel

Represent a graph by counting the number of each type of **graphlet** it contains.

Steps:
1. Enumerate all graphlets up to $k$ (e.g., $k$ = 3, 4, or 5 nodes).
2. For each graphlet type, count how many times it appears as a subgraph of $G$.
3. This count vector is the graph's feature vector.

The definition of **graphlets at the graph-level** do **not** need to be connected and are **not** rooted at a specific node, which is different from node-level graphlets.

**Normalisation:** If comparing graphs of different sizes, normalise the count vectors (e.g., by the total count) so that large graphs don't dominate.

**Limitation:** Counting graphlets is computationally expensive.

### Weisfeiler-Lehman (WL) Kernel

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

## 3. Summary: Traditional ML Pipeline on Graphs

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

## 4. Node Embeddings

<br>

### 4.1 Problem Definition

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

### 4.2 The Encoder-Decoder Framework

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
{: .prompt-tip }

The embedding matrix is what we learn. Every node is assigned a unique, independent embedding vector. Here, no parameter sharing, no generalisation to unseen nodes.

<br>

### 4.3 Defining Similarity via Random Walks

For both DeepWalk and Node2vec, we define similarity as the probability that 2 nodes co-occur on short random walks. Specifically, 

$$
\mathbf{z}_{u}^\top \mathbf{z}_{v} \approx P(\text{u and v co-occur on a random walk})
$$

- **Expressivity:** walks naturally capture multi-hop, higher-order relationships; not just direct edges.

- **Efficiency:** only train pairs that actually co-occur in the walks, not all $O(\vert V \vert^{2})$ pairs.

- **Flexibility:** changing the walk strategy (§3.4) changes what "similar" means.

<br>

### 4.4 Walk Strategy 

DeepWalk and Node2vec are differeciated by walk strategies.

The **walk strategy $R$** determines the **neighbourhood $N_{R}(u)$**, the multiset of nodes visited on walks starting from node $u$.

> "multiset" means we allow the same node appears more than once in the list.
{: .prompt-tip }

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

### 4.5 Optimisation

All random-walk embedding methods share the **same optimisation pipeline**. The **differences** between random-walk embedding methods live *entirely* in "How walks are generated" (§3.4).

The walk prediction task is purely a **pretext task** to force the embbedding vectors into a configuration that captures network structure. Once training is done, we throw away the objective and keep **only** the **learned embedding matrix $Z$** for down stream tasks. i.e. Optimising random walk prediction is *NOT* to build an accurate walk predictor. We dont care about the accuracy of the walk predictions themselves.

### Maximum Likelihood Formulation

Treat the random walk neighbourhoods $N_{R}(u)$ as **observed data**, and the embeddings $f$ mapping to $\mathbf{z}_{u}$ as **parameters**.

The goal is to find embeddings that maximise the likelihood of the observed co-occurances:

$$
\max_{\mathbf{f}} \sum_{u \in V} \log P\bigl(N_R(u) \mid \mathbf{z}_u\bigr)
$$

> In the shallow encoding setup, $f$ is just the lookup table, so the actual learnable parameters are the embedding vectors $\mathbf{z}_{u}$ themselves (i.e. the columns of the embedding matrix $Z$).
>
> In this case, $f$ is equivalent to specifying $Z$.
{: .prompt-info }

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

- $$\mathbf{z}_{u}^\top \mathbf{z}_{u}$$ is the raw similarity score between node $$u$$ and $$v$$.

- $$\exp (\mathbf{z}_{u}^\top \mathbf{z}_{u})$$ makes the similarity score *strictly positive* to generate the probability output.

- $$\sum_{n \in V} \exp (\mathbf{z}_{u}^\top \mathbf{z}_{n})$$ calculates similarity scores between node $$u$$ and every other node in the graph. This ensures the output is a clean percentage in $$(0, 1)$$.

- $P(v \mid \mathbf{z}_{u})$ sums to 1 and all entries are positive, making it a valid probability distribution.

- $P(v \mid \mathbf{z}_{u})$ is a monotonic function, where a higher dot product score implies a higher probability.


**Problem:** the denominator requires summing over every node pair in the graph. For each training pair $(u, v)$, this is $$O(\vert V \vert)$$, and across all pairs it becomes $$O(\vert V \vert)^{2}$$. Intractable for large graphs.

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
    - Look up $$\mathbf{z}_{u}$$ and $$\mathbf{z}_{v}$$ from the embedding matrix;
    - Sample $K$ neagtive nodes $n_1, ..., n_K$ proportional to node degree, so the probability of picking node $n$ as a negative is:
    $$
    P(n) = \frac{\text{deg}(n)}{\sum_{m \in V} \text{deg}(m)}
    $$
    - Compute the negative-sampling loss for this pair

4. **Backpropagate** through the loss to compute gradients $$\frac{\partial\mathcal{L}}{\partial\mathbf{z}_u}$$, $$\frac{\partial\mathcal{L}}{\partial\mathbf{z}_v}$$, $$\frac{\partial\mathcal{L}}{\partial\mathbf{z}_{n_i}}$$.

5. **Update** the corresponding columns of $\mathbf{Z}$ via stochastic gradient descent (SGD), where $\eta$ is the learning rate (i.e. Overwrite random numbers in the embeddings with better numbers to make entries as probabilities we expected):

$$
\mathbf{z}_u \leftarrow \mathbf{z}_u - (\eta \frac{\partial\mathcal{L}}{\partial\mathbf{z}_u})
$$

6. **Iterate** over millions of node pairs. 
    - Sample **true** node pairs $\rightarrow$ Calculate the dot product and squash it with the sigmoid function to get the probability $p$ $\rightarrow$ Get the error between the target and the function result $\rightarrow$ Update in the embedding matrix $\mathbf{Z}$ for both nodes
        - Visual result: true nodes are **pulled together** in the embedding space 
        - Their dot products increase for the next time 

    - Sample **fake** node pairs $\rightarrow$ Calculate the dot product and squash it with the sigmoid function to get the probability $p$ $\rightarrow$ Get the error between the target and the function result $\rightarrow$ Update in the embedding matrix $\mathbf{Z}$ for both nodes
        - Visual result: fake nodes are **pushed apart** in the embedding space 
        - Their dot products decrease for the next time 

7. **Convergence:** continue until the loss stops decreasing meaningfully. At this point, the embedding matrix $\mathbf{Z}$ has stabilised.

8. **Stop the training loop and Keep $\mathbf{Z}$**. Discard the objective, the walks, and the negative sampler. The columns of $\mathbf{Z}$ are the final node embeddings, ready for any downstream tasks.

<br>

### 4.6 Full Algorithm Summary

| Step | What happens | Shared or method-specific? |
|---|---|---|
| 1. **Preprocess** | Compute biased transition probabilities (Node2vec) or skip (DeepWalk) | Method-specific |
| 2. **Walk** | Simulate $r$ walks of length $l$ per node using strategy $R$ | Method-specific (uniform vs. biased) |
| 3. **Collect** | Build $N_R(u)$ for each node from walk co-occurrences | Shared |
| 4. **Optimise** | SGD and Negative Sampling on the log-likelihood objective | Shared |
| 5. **Use** | Plug $\mathbf{z}_u$ vectors into any downstream task | Shared |


<br>

### 4.7 Method Comparison

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

### 4.8 Key Equations 

| Concept | Equation |
|---|---|
| Embedding goal | $\text{similarity}(u,v) \approx \mathbf{z}_u^\top\mathbf{z}_v$ |
| Shallow encoder | $\text{ENC}(v) = \mathbf{Z} \cdot \mathbf{v}$ |
| MLE objective | $\max_{\mathbf{f}} \sum_{u} \log P(N_R(u) \mid \mathbf{z}_u)$ |
| Independence assumption | $$P(N_R(u) \mid \mathbf{z}_u) = \prod_{v \in N_R(u)} P(v \mid \mathbf{z}_u)$$ |
| Loss (negative log-likelihood) | $\mathcal{L} = \sum_{u}\sum_{v \in N_R(u)} -\log (P(v \mid \mathbf{z}_u))$ |
| Softmax (multi-class) | $P(v \mid \mathbf{z}_u) = \frac{\exp(\mathbf{z}_u^\top\mathbf{z}_v)}{\sum_n\exp(\mathbf{z}_u^\top\mathbf{z}_n)}$ |
| Negative sampling (binary) |  $$\approx -\log(\sigma(\mathbf{z}_u^\top\mathbf{z}_v)) - \sum_{i=1}^k\log(\sigma(-\mathbf{z}_u^\top\mathbf{z}_{n_i}))$$ |
| Node2vec bias ($d_{ux}=0,1,2$) | Weights: $\frac{1}{p}$, $1$, $\frac{1}{q}$ |


<br>
<br>
<br>

## 5. Graph Embeddings

<br>

### 5.1 Problem Definition

To predict the **entire graph** (or **subgraph**), we want to embed it into a single vector:

$$
f: G \rightarrow \mathbf{z}_{G} \in \mathbb{R}^{d}
$$

Typical tasks include (1) classifying toxic vs. non-toxic molecules; (2) identify anomalous graphs; (3) compare graph similarity.

**Core challenges:** How to collapse the variable-size structure of a whole graph into a fixed-length vector, preserving meanful structural information?

<br>

### 5.2 Approaches to Graph Embedding

*How to aggregate node-level information into a graph-level representation?*

### Approach 1 - Aggregate Node Embeddings

- **Strategy:** Run any node embedding method (DeepWalk, Node2Vec, etc.) on $G$, then sum (or average) all node embeddings:

$$
\mathbf{z}_{G} = \sum_{v \in G} \mathbf{z}_{v} \quad \text{or} \quad \mathbf{z}_{G} = \frac{1}{\vert V \vert} \sum_{v \in G} \mathbf{z}_{v}
$$

- **Strengths:** Trivially simple, works with any existing node embedding, no additional training needed.

- **Weakness:** Sum/Avg are very lossy compressions. They destroy information about how nodes relate to each other within the graph. Two sturcturally very different graphs could produce identical sums if their node embeddings happen to average out the same way.


### Approach 2 - Virtue Node

- **Strategy:** Introduce a virtual node, which is connected to all nodes in the (sub)graph to be embeded. Then run a standard node embedding. The embedding of the virtual node is the graph embedding $\mathbf{z}_{G}$.

- **Strengths:** Leverage the existing node embedding withou modification. Since the virtual node is adjacent to every node in $G$, random walks starting from it will traverse the whole graph, naturally aggregating information from all nodes.

- **Weakness:** Adding a node connected to everything changes the graph topology, can distort the original graph structure.

### Approach 3 - Anonymous Walk Embeddings (AWE)

*More details in §4.3.*

<br>

### 5.3 Anonymous Walks

### Definition

An **anonymous walk** strips away node identities (e.g. node $A, B, C$) and replace them with the index recores the order where each node was first encountered (**first-visit indices**).

> **Example:** 
>
> A random walk visits nodes $$A \rightarrow B \rightarrow C \rightarrow B \rightarrow D$$.
> - $A$ is the 1st new node $\rightarrow$ index **1**
> - $B$ is the 2nd new node $\rightarrow$ index **2**
> - $C$ is the 3rd new node $\rightarrow$ index **3**
> - $B$ was already seen $\rightarrow$ index 2
> - $D$ is the 4th new node $\rightarrow$ index **4**
> Anonymous walk will be $$(1, 2, 3, 2, 4)$$.
{: .prompt-example }

With such **agnostic to node identity** walks, two graphs with completely different node labels but the same topology will produce the same anonymous walk distributions.

### Counting Anonymous Walks

For a given walk length $l$, there is a finite set of possible anonymous walks. This limits how long the walks can be before the representation becomes unwieldy.

Anonymous walks follow a strict chronological rule: whenever stepping onto a node that haven't been visited yet, assign it the **lowest unused positive integer**. Therefore, it doesn't allow to skip numbers (e.g. 1, 1, 3). 

> **Example:** 
>
> with $l = 3$ (3 steps), all 5 possible anonymous walks are:
>
> $$
> w_1 = 111, \quad w_2 = 112, \quad w_3 = 121, \quad w_4 = 122, \quad w_5 = 123
> $$
{: .prompt-example }

Essentially, the number of *distinct* anonymous walks grows **exponentially** with length $l$ (Bell Numbers):

| Walk length $l$ | Number of anonymous walks $\eta$ |
|---|---|
| 1 | 1 |
| 2 | 2 |
| 3 | 5 |
| 4 | 15 |
| 5 | 52 |
| 6 | 203 |
| 7 | 877 |

<br>

### 5.4 Two Uses of Anonymous Walks for Graph Embedding

### Feature-based (Simple Counting)

Simulate $m$ anonymous walks of length $l$ starting from every node in $G$. Record the fraction of times each anonymous walk pattern $$w_i$$ occurs. The graph embedding is then this **probability distribution** over walk patterns:

$$
\mathbf{z}_{G}[i] = \text{probability of anonymous walk} w_i \text{occurring in} G
$$

> **Example:** 
>
> For $l = 3$ with 5 possible walks, $$\mathbf{z}_G$$ is a 5-dimensional vector whose entries sum to 1.
{: .prompt-example }

To make the empirical distribution be $\epsilon$-close to the true distribution with probability $\geq 1 - \delta$, the numner of sample is:

$$
m = \left\lceil \frac{2}{\epsilon^2} \left( \log (2^\eta -2) - \log (\delta) \right) \right\rceil
$$

, where $\eta$ is the total number of anonymous walks.

- **Strengths:** 
    - Completely unsupervised;
    - No learning required beyond sampling;
    - Produces a fixed-size representation.

- **Weakness:** 
    - The representation is purely count-based. It doesn't learn which walk pattern are important for a given task. 
    - The dimensionality ($\eta$) grows exponentially.



### Data-driven (Learn Walk Embedding)

Learn a low-dimensionality embedding for each anonymous walk pattern alongside the graph embedding. 

**Goal:** Predict contect walks from co-occurring walks.

**Setup:** 

- $$\mathbf{z}_G \in \mathbb{R}^{d_g}$$: the graph embedding vector

- $$\mathbf{z}_i \in \mathbb{R}^{d_a}$$: the embedding of anonymous walk $$w_i$$

- $\Delta$: window size. Walks that start from the same node within $\Delta$ positions of each other are considered co-occurring.

**Objective:** given the context walks within a $\Delta$-sized window and the graph embedding, predict the target walk:

$$
\max_{\mathbf{Z}, \mathbf{z}_G} \;\; \frac{1}{T} \sum_{t=\Delta}^{T-\Delta} \log P\left(w_t \;\middle|\; w_{t-\Delta}, \ldots, w_{t+\Delta}, \mathbf{z}_G \right)
$$

- $T$: total number of walks sampled in the sequence
- $t$: current index/position in $T$ walks
- $$w_t$$: specific target ealk currently looking at
- $\Delta$: size of context/neighbourhood

> **Example:** 
>
> If $\Delta = 1$, we will look at the exact one walk before the target walk ($$w_{t-1}$$) and the exact one after ($$w_{t+1}$$).
{: .prompt-example }

, where the probability is computed via softmax:

$$
P\left(w_t \;\middle|\; \{w_{t-\Delta}, \ldots, w_{t+\Delta}, \mathbf{z}_G\}\right) = \frac{\exp(y(w_t))}{\sum_{i=1}^{\eta} \exp(y(w_i))}
$$

> This is the probability of seeing target walk ($$w_{t}$$), given the surrounding walk contexts and the overall graph embedding.
{: .prompt-info }

> For the whole graph embedding learning process, 
>   - $\mathbf{z}_G$ is the global clue, meaning a completely separate, special vector of the entire graph. It's treated a shared context for all walks in the graph.
>   - $$w_{t-\Delta}$$ are local clues, representing local structures.
{: .prompt-info }

, and the score function $$y(w_t)$$ is:

$$
y(w_t) = b + U \cdot \text{cat}\left( \frac{1}{2\Delta} \sum_{i=-\Delta, i \neq 0}^{\Delta} \mathbf{z}_{t+i}, \;\; \mathbf{z}_G \right)
$$

That is - Average the embeddings of the context walks, concatenate with the graph embedding, and pass through a linear layer $(U, b)$.


**Training Procedure:**

1. For each node $u$ in $G$, run $T$ random walks of length $l$ to produce $$N_R(u) = \{ w_1^{u}, w_2^{u}, ..., w_T^{u} \}$$.

2. Optimise the objective above over all nodes, learning both the walk embeddings $$\mathbf{z}_i$$ and the graph embedding $$\mathbf{z}_G$$ jointly via SGD.

3. After training, only keep $$\mathbf{z}_G$$ as the final graph-level representation.

<br>

### 5.5 Using Embeddings for Downstream Tasks

- **Cluster / community detection:** Cluster node embeddings to discover communities.

- **Node classification:** Use $$\mathbf{z}_v$$ as input features to a classifier to predict node labels.

- **Link prediction:** Given a pair $(u, v)$, predict whether an edge exists. The pair of embeddings $$(\mathbf{z}_u, \mathbf{z}_v)$$ can be combined in several ways to produce a single feature vector for a binary classifier:

    | Method | Formula | Captures |
    |---|---|---|
    | Concatenation | $$[\mathbf{z}_u ; \mathbf{z}_v]$$ | Full information from both, asymmetric |
    | Hadamard (element-wise) product | $$\mathbf{z}_u \odot \mathbf{z}_v$$ | Per-dimension interaction |
    | Sum / Average | $$\mathbf{z}_u + \mathbf{z}_v$$ | Symmetric, position-agnostic |
    | Distance | $$\|\mathbf{z}_u - \mathbf{z}_v\|$$ | How far apart they are |

- **Graph classification:** Use $$\mathbf{z}_G$$ as input to a classifier to predict graph-level labels (e.g. toxic / non-toxic).

<br>

### 5.6 Comparative Analysis

### Node Embedding vs. Graph Embedding

| | Node Embedding | Graph Embedding |
|---|---|---|
| **Target** | Individual node $v$ | Entire graph $G$ |
| **Output** | $$\mathbf{z}_v \in \mathbb{R}^d$$ per node | $$\mathbf{z}_G \in \mathbb{R}^d$$ per graph |
| **Similarity semantics** | $$\mathbf{z}_u^\top \mathbf{z}_v \approx$$ node-pair similarity | $$\mathbf{z}_{G_1}^\top \mathbf{z}_{G_2} \approx$$ graph-pair similarity |
| **Downstream tasks** | Node classification, link prediction | Graph classification, anomaly detection |
| **Identity dependence** | Depends on specific nodes | Must be invariant to node labelling |


### Anonymous Walks vs. Standard Random Walks (Node Embedding)

| | Standard Random Walk (DeepWalk / node2vec) | Anonymous Walk |
|---|---|---|
| **Records** | Actual node identities visited | First-visit indices (node-identity agnostic) |
| **Goal** | Node-level embedding via co-occurrence | Graph-level embedding via walk-pattern distribution |
| **Similarity captured** | Node similarity (co-occurrence probability) | Structural similarity of entire graph |
| **Identity dependent?** | Yes <br> each node gets its own embedding | No <br> purely structural |
| **Output** | One vector per node | One vector per graph |

<br>

### 5.7 Summary of Key Equations

| Concept | Equation |
|---|---|
| Graph embedding by aggregation | $$\mathbf{z}_G = \sum_{v \in G} \mathbf{z}_v$$ |
| Anonymous walk definition | $$(f(v_1), f(v_2), \ldots, f(v_k))$$ where $$f(v_i)$$ = first-visit index |
| Feature-based AWE | $$\mathbf{z}_G[i] = P(w_i \text{ in } G)$$ |
| Sample size for feature-based AWE | $$m = \lceil \frac{2}{\varepsilon^2}(\log(2^\eta - 2) - \log(\delta)) \rceil$$ |
| Data-driven AWE objective | $$\max \frac{1}{T} \sum_{t=\Delta}^{T-\Delta} \log P(w_t \mid w_{t-\Delta}, \ldots, w_{t+\Delta}, \mathbf{z}_G)$$ |
| Walk prediction score | $$y(w_t) = b + U \cdot \text{cat}\left(\frac{1}{2\Delta}\sum \mathbf{z}_i, \; \mathbf{z}_G\right)$$ |


<br>
<br>
<br>

## 6. PageRank

<br>

### 6.1 Link Analysis Approach(Algorithm)

The Web can be modelled as a **Directed Graph**, where nodes are web pages and edges are hyperlinks.

However, not all web pages are equally important. The key insight behind the PageRank is *"Treating Links as Votes"*. Specifically, in-links are votes for importance, and a vote from an important page counts more.

**Recursive Definition:** 

- Page $j$'s importance depends on the importance of pages that point to it

- Each page distributes its importance equally across its out-links.

Formally, the **PageRank** $r_j$ of page $j$ is defined by the **flow equation**:

$$
r_j = \sum_{i \rightarrow j} \frac{r_i}{d_i}
$$

, where $d_i$ is the out-degree of node $i$ and the sum runs over all nodes $i$ that link to $j$.

<br>

### 6.2 Three Equivalent Formulations

PageRank can be understood from three different angles, which all lead to the same solution.

### Formulation 1 - Matrix

Define the **stochastic adjacency matrix** $M$ as:

- If page $j$ has $d_j$ out-links and $j \rightarrow i$, then $$M_{ij} = \frac{1}{d_j}$$

- $M$ is **column stochastic**, where each column sums to 1

The flow equation can then be written as:

$$
\mathbf{r} = M \cdot \mathbf{r}
$$

, where $\mathbf{r}$ is the **eigenvector** of $M$ corresponding to eigenvalue of 1. We can also refer $\mathbf{r}$ as the **principal eigenvector**.

### Formulation 2 - Random Walk

Imagine a random surfer on the Web:

1. At time $t$, the surfer is on some page $i$

2. At time $t+1$, the surfer follows one of $i$'s out-links uniformly at random

3. This repeats indefinitely

Let $\mathbf{p}(t)$ be the probability distribution over pages at time $t$, then $\mathbf{p}(t+1) = M \cdot \mathbf{p}(t)$.

As $t \rightarrow \infty$, the surfer reaches a stationary distribution $\mathbf{p}$, satisfying $\mathbf{p} = M \cdot \mathbf{p}$.

> $\mathbf{p}$ is exactly the PageRank vector
{: .prompt-tip }

> A page's PageRank = The fraction of time the random surfer spends on it in the long run.
{: .prompt-tip }

### Formulation 3 - Power Iteration

Given $\mathbf{r} = M \cdot \mathbf{r}$, we can compute $\mathbf{r}$ iteratively:

1. **Initialise:** $$\mathbf{r}^{(0)} = \left[\frac{1}{N}, \frac{1}{N}, \ldots, \frac{1}{N}\right]^T$$

2. **Iterate:** $$\mathbf{r}^{(t+1)} = M \cdot \mathbf{r}^{(t)}$$

3. **Stops when:** $$\| \mathbf{r}^{(t+1)} - \mathbf{r}^{(t)} \|_1 < \epsilon$$

Repeatedly multiplying by $M$ amplifies the component along the principal aigenvector, while all other components decay.

> In practice, this process converges within about 50-100 iterations.
{: .prompt-tip }

<br>

### 6.3 Two Problems with Power Iteration

### Spider Traps

A **spider trap** is a set of pages whose out-links all stay within the group. 

The random surfer gets trapped, amd all importance acculates *inside* the trap.

> **Example:**
> 
> If page $b$ has a self-loop as its only out-link, power iteration converges to $$ r_b = 1$$ and $$r = 0$$ for everything else.  
{: .prompt-example }

> This is NOT mathematical failure, since the eigenvector still exists.
> 
> But the resulting scores are meaningless.
{: .prompt-tip }

### Dead Ends

A **dead end** is a page with in-links but **no out-links**.

The matrix $M$ is no longer column stochastic, as the dead-end column sums to 0. So importance "leaks out" of the graph. 

Over iterations, all PageRank drains to zero.

> **Example:**
>
> The surfer reaches page $b$ which has **no out-links**. i.e. *there is nowhere to go.*
>
> The random walk breaks down.
{: .prompt-example }

<br>

### 6.4 Random Teleportation (Solution) 

Both spider-traps and dead-ends can be solved by a mechanism **"Random Teleportation"**.

At each time step, the surfer has two options:

- With probability $\beta$, follow an out-link uniformly at random

- With probability $(1 - \beta)$, **teleport** to any page in the graph uniformly at random

Typically, $$\beta \in [0.8, 0.9]$$.

> **Connection to Marchov Chains:**
>
> For power iteration to converge to a unique stationary distribution, the Markov chain must be irreducible and aperiodic.
>
> Teleporttion guarantees both properties.
>
> - irreducible: any state reachable from any other
> - aperiodic: no fixed-period cycles
{: .prompt-info }

<br>

### The Google Matrix

Incorporating teleportation, the **PageRank equation** becomes:

$$
r_j = \sum_{i \rightarrow j} \beta \cdot \frac{r_i}{d_i} + (1 - \beta) \cdot \frac{1}{N}
$$

In matrix form, define the **Google Matrix** $A$ as:

$$
A = \beta M + (1 - \beta) \cdot \frac{1}{N} \mathbf{e}\mathbf{e}^{T}
$$

, where $$\mathbf{e}$$ is the vector of all ones.

The **PageRank vector** satisfies:

$$
\mathbf{r} = A \cdot \mathbf{r}
$$

, and is computed via Power Iteration as before.

<br>

### 6.5 PageRank Variants

Standard PageRank is a measure of **global importance**. With evenly splitted nodes, the surfer randomly teleports to a any node in the graph. But what if the importance is relative to specific context? 

Below there are three variantes of PageRank, sharing the **same random walk framework** as the Random Teelportation, but differ in **where the surfer teleports to the teleport set $S$** when the walk starts.

| Variant | Teleport set $S$ | What it measures |
|---------|-----------------|------------------|
| **PageRank** | All nodes (uniform) | Global importance |
| **Personalised PageRank (PPR)** | A subset of nodes | Importance relative to a topic |
| **Random Walk with Restarts (RWR)** | A single query node $\{q\}$ | Proximity / similarity to $q$ |

### PPR Mechanism

The resulting stationary distribution is biased toward pages that are close to the seed set. Page that are structually far from $S$ will receive low scores, even they have many in-links.

PPR can be used in the topic-sensitive web search. For example, if $S$ contains pages about "Sports", the PPR scores rank all pages by the relevance of "Sports", combining link structure with topic proximity.

### RWR Mechanism

Given the teleport set $S$ only contains a single node ${q}$, the walker always returns to the node $q$ at each restart.

Specifically,
1. Start at the Query Node $q$
2. At each step, follow the [Random Teleportation Mechanism](#64-random-teleportation-solution)
3. Simulate for many steps, counting how often each node is visited
4. Nodes with the highest visit counts have the **highest proximity** to the Query Node $q$

The visit countes converge to the stationary distribution, giving each node a proximity score relative to $q$.

> The proximity score naturally captures:
> - **Multiple connections** between $q$ and another node
> - **Multiple paths**, not just shortest paths
> - **Direct and Indirect connections**, as multi-hop relationships
> - **Degree of the node**
{: .prompt-tip }

> RWR is particularly effective for **"Recommendation"** in a **Bipartite Graph**.
>
> Consider a user-item Bipartite Graph where users are connected to items they have interacted with. To find an item similar to a query item $q$:
> - Run RWR from $q$
> - Items with the highest visit counts are the best recommendations
{: .prompt-info }

As the teleport set $S$ shrinks from all nodes $\rightarrow$ a subset $\rightarrow$ a single query node, the measure shifts from **global importance** to **local proximity**.

<br>

### 6.6 Matrix Factorisation

Random Walk based embeddings $\equiv$ Matrix Factorisation

Given an [embedding matrix](#4-node-embeddings) $$Z \in \mathbb{R}^{d \times \vert V \vert}$$ where column $$\mathbf{z}_u$$ is the embedding vector of node $u$, the decoder measures **similarity** via the inner product

$$
\text{similarity}(u, v) \approx \mathbf{z}_{u}^\top \mathbf{z}_{v}
$$

> The definition of "similarity" determines which matrix to factorise
{: .prompt-tip }

### Adjacency-based Similarity

Define two nodes are "similar" if they are directly connected. This means

$$
\mathbf{z}_u^{T} \mathbf{z}_v \approx A_{u,v}
$$

, where $A$ is an adjacency matrix.

> The intuition is to have $$Z^{T} Z = A$$.
>
> But this is impossible, since the dimensioanlity of embeddings $d \ll$ the number of nodes $N$, in most cases. 
{: .prompt-tip }

Our optimisation objective becomes

$$
\min_{Z} \| A - Z^{T} Z \|_2
$$

<!-- > The inner product decoder with node similarity which is redefined by edge connectivity $\approx$ Matrix factorisation of adjacency matrix $A$
{: .prompt-info } -->

> Connections among all three perspectives
> - **PageRank** (Random Walks) determines node importance via stationary distribution of a Markov Chain
> - **Node Embeddings** (DeepWalk, Node2Vec) learn node representations by optimising co-occurrance in random walks
> - **Matrix Factorisation** factorise a node similarity matrix into a low-rank embedding vectors 
{: .prompt-info }


<br>

### 6.7 Limitations

1. **No embedding for unseen nodes**
    If the graph is updated with new nodes, embeddings for new nodes would not automatically appear. Need to recompute the whole updated graph.

2. **Cannot capture local structual similarity**
    Two nodes that far apart on the graph will have different embeddings, even they share a similar structure.

3. **Cannot utilise node, edge, or graph features**
    The methods only use the graph structure (adjacency matrix). They completely ignore any potential node features (e.g. user profile information), edge features (e.g. relationship types among nodes), or the graph itself (e.g. metadata).

<br>
<br>
<br>

## 7. Semi-supervised Node Classification

<br>

### 7.1 Problem Definition

Given a graph where only some nodes are labelled, we want to predict labels to remaining unlabelled nodes.

> This is a semi-supervised learning as the model leverages both labelled and unlabelled nodes simultaneously during the training phase.
{: .prompt-tip }

**Network Correlation** is the key insight, where individual behaviours are correlated in the network.

> **Correlation** means nearby nodes belong to the same group/category.
{: .prompt-tip }

Specifically, Correlation involve two concepts - **Homophily** and **Influence**

- Homophily: **similar** nodes tend to connect.

- Influence: connected nodes affect each other over time.

> $$
> \text{individual characteristic} \xrightleftharpoons[\text{Influence}]{\text{Homophily}} \text{social connections}
> $$
> {: .prompt-info }

Under the Markov Assumption, the label $Y_v$ of node $v$ depends on labels of it's first-degree neighbours $N_v$:

$$
P(Y_v) = P(Y_v \mid N_v)
$$

All above motivate the *"guilt-by-association"* principle. If most of $v$'s neighbours carry label $L$, then $v$ is likely labelled as $L$ as well.

<br>

### 7.2 Collective Classification

Collective Classification assign labels to all unlabelled nodes simultaneously, **iteratively** refining predictions using network structure.

Three steps of the framework:
1. **Local Classifier**: Assign initial labels based on node features alone. No network information used.
2. **Relational Classifier**: Predict a node $v$'s label from labels and/or features of $v$'s neighbours. Network information entered.
3. **Collective Inference**: Iteratively apply the Relational Classifier across the graph, propogating label information beyond immediate neighbours, until predictions *stabilise (converged)* or reaching preset *maximum* iteration numbers.


### Prbabilistic Relational Classifier

The class probability of node $v$ is the weighted average of class probabilities of its neighbours.

For a node $v$ with neighbours $N_v$, the probability that $v$ belongs to class $c$ is updated as

<a id="eq-prc"></a>
$$
P(Y_v = c) = \frac{1}{\sum_{(v,u) \in E} W(v,u)} \sum_{(v,u) \in E} W(v,u) \cdot P(Y_u = c) \tag{1}\label{eq-prc}
$$

, where $W(v,u)$ is the edge weight between $v$ and $u$.

> $$P(Y_v = c)$$ is the probability of node $v$ with label $c$.
>
> $W(v,u)$ becomes entry of a simple adjacency matrix $$A_{v,u}$$ if the graph is unweighted.
{: .prompt-tip }

**Architecture:**
1. **Initialise**: 
    - Labelled nodes keep their ground-truth labels (fixed throughout).
    - Unlabelled nodes get uniform proability over all classes (or a prior if available).

2. **Iterate**: Update every unlabelled node in randome order using [Equation 1](#eq-prc).

3. **Repeat** until convergence and stabilised or the iteration budget is exhausted.

> Converge is not always guaranteed!
>
> This method replies solely on the graph sructure and initial labels, without any node feature information.
>
> Update ordering affects the result, especially on small graphs. Large graphs are less sensitive.
{: .prompt-tip }

> Iteration Stops When:
> - Converge: Numerical probability stops making any significant change
>   In maths, we can say
>   $$
>   \forall \epsilon > 0, \exists \delta \in \mathbb{N}, s.t. \vert P_{\delta + 1}(Y_v = c) - P_{\delta}(Y_v = c) \vert < \epsilon
>   $$
>
> - Stabilise: Classification stops flipping back and forth, where the decision is locked in.
> {: .prompt-info }

### Iterative Classification

Iterative Classification incorporates node features alongside neighbour label inforamtion.

Each node $v$ is described by a vector $$\mathbf{a}_v$$, consisting of 2 parts

$$
\mathbf{a}_v = [\mathbf{f}_v, \mathbf{z}_v]
$$

, where 
- $$\mathbf{f}_v$$ is the node $v$'s own feature vector
- $$\mathbf{z}_v$$ is a summary vector of labels/features of $v$'s neighbours, computed via aggregation

**Architecture:**
- Phase 1 [training]

