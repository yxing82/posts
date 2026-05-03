---
title: "Notes on ML with Graphs (Updating)"
date: 2026-05-03
categories: [Notes, Machine Learning]
tags: [graphs, ml]
math: true
---

## Traditional Machine learning

For traditional machine learning, people basically work on capturing features to make predictions. Specifically, researchers use hand-designed features.

In a Graph with ML, given $G = (V, E)$, we want to know how to learn a function $f: V \rightarrow \mathbb{R}$

In this post, we will talk about 3 types of predictions about the Graph:

- Node-level prediction
- Link-level prediction
- Graph-level prediction

## Node-level Prediction

- Task: Mainly focus on node classifications based on features.
- Goal: 
    - Characterise network structure
    - Identify the node position in the network

Researches are focused on 2 features: 
1. Importance-based features
2. Structure-based features

### Importance-based features

For importance-based features, the main question is "_How to identify the importance of a certain node?_"


