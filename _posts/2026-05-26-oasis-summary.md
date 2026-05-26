---
title: "Summary: OASIS – Optimisation-based Activity Scheduling with Integrated Simultaneous Choice Dimensions"
date: 2026-05-26
description: "Reading notes on Pougala et al. (2023) — an integrated framework for simultaneous activity-based modelling using Metropolis-Hastings sampling and maximum likelihood estimation."
categories: [Literature Notes, Transport Modelling]
tags: [activity-based modelling, discrete choice, MCMC, parameter estimation, literature review]
math: true
---

## Full reference

```bibtex
@article{pougala2023oasis,
  title={OASIS: Optimisation-based Activity Scheduling with Integrated Simultaneous choice dimensions},
  author={Pougala, Janody and Hillel, Tim and Bierlaire, Michel},
  journal={Transportation Research Part C: Emerging Technologies},
  volume={155},
  pages={104291},
  year={2023},
  publisher={Elsevier},
  doi={10.1016/j.trc.2023.104291}
}
```

## Questions

### What is the article about?

[This directs to the original article](https://www-sciencedirect-com.libproxy.ucl.ac.uk/science/article/pii/S0968090X23002802?ref=pdf_download&fr=RR-2&rr=9c3f42011f1efd7e) 

It is research on **OASIS**, an integrated framework for **activity-based** modelling, which considers all choice dimensions simultaneously. 

The authors also present a methodology to estimate the behavioural parameters from historical data to generate realistic daily schedules. 

OASIS: Optimization-based Activity Scheduling with Integrated Simultaneous choice dimensions

### What kind of contribution is the paper trying to make?

* The formulation of an integrated framework of simultaneous activity-travel simulation, more flexibility to accommodate different scenarios
* A methodology to effectively sample unchosen daily schedules
* Sampling a finite choice set, essential for the application of MLE
    * Simplify the estimation procedures
    * Formally and explicitly links behaviours and activity schedules - interpretable parameter estimates
    * Provide extensive econometric theory to support analysis

### What gap(s) does it identify/tackle?

* Existing activity-based models rely on **sequential choice models** between subsequent choices, oversimplifying the scheduling process and missing trade-offs between choice dimensions
* In the authors' own prior work [(Pougala et al., 2022)](https://www.sciencedirect.com/science/article/pii/S1755534522000124), the simultaneous model's parameters were **not estimated** — instead, fixed values from the literature were used, limiting behavioural realism
* **Parameter estimation** in activity-based models is generally challenging due to the size of the problem, spatio-temporal complexity, and lack of appropriate data
* Sequential models can estimate parameters in stages (e.g. [Bowman and Ben-Akiva, 2001](https://www.sciencedirect.com/science/article/pii/S0965856499000439); [Chen et al., 2020](https://www.sciencedirect.com/science/article/pii/S0968090X20305659)), but this comes at the expense of **model flexibility and behavioural realism**
* **Choice sets** are usually treated as given or constructed with arbitrary decision rules; when all choice dimensions are considered simultaneously, the combinatorial nature of the solution space makes full enumeration infeasible
* Prior approaches to choice set generation and parameter estimation each have shortcomings:
    * Deterministic approaches (e.g. [Bowman and Ben-Akiva, 2001](https://www.sciencedirect.com/science/article/pii/S0965856499000439); [Arentze and Timmermans, 2000](https://www.sciencedirect.com/science/article/pii/S0191261503000948)) predefine or enumerate choice sets using domain-specific rules — limited flexibility
    * Heuristic estimation methods (e.g. [Recker et al., 2008](https://www.sciencedirect.com/science/article/pii/S109396872601409X) using genetic algorithms) lack econometric properties
    * [Xu et al. (2017)](https://www.sciencedirect.com/science/article/pii/S235214651730323X) use pattern clustering with path-size logit, but do not correct for importance sampling bias and their methodology creates endogeneity, leading to overfitting

### Main research question(s) or argument(s) 

* Can the behavioural parameters of a **simultaneous** activity-based model be estimated from historical data using maximum likelihood estimation, given the combinatorial and multidimensional nature of the problem?
* Can the **Metropolis-Hastings algorithm** be applied to generate a choice set of competitive alternative schedules that is both informative enough for stable parameter estimates and varied enough to minimise bias?
* Does estimating parameters (as opposed to borrowing from the literature) improve the realism of simulated activity schedules?
* Does strategic sampling (MH algorithm) outperform random or clustering-based choice set generation for estimation quality?
* How do different utility specifications (flexibility-level vs. activity-specific vs. S-shaped MATSim scoring) compare in capturing observed scheduling behaviour?

### How will they test and/or make use of these ideas as part of their research?

The methodology consists of two core elements: **(i) choice set generation** and **(ii) discrete choice parameter estimation**.

#### Scheduling Framework

* A schedule is a sequence of activities starting and ending at home over a 24-hour time horizon $T$
* Each activity $a$ is characterised by: location $\ell_a$, start time $x_a$, duration $\tau_a$, cost $c_a$, and outbound travel mode $m_a$
* Two utility specifications are tested:
    * **Linear penalties**: utility captures participation, deviations from desired start time and duration (penalised linearly), and travel costs
    * **S-shape (MATSim)**: duration utility follows an asymmetric S-shaped curve with an inflection point — captures satiation effects without requiring explicit desired durations

#### Metropolis-Hastings Algorithm

* Goal: To generate a choice set of realistic unchosen daily schedules
* Core Components:
    * The State ($X_t$): Each state in the Markov Chain is a full 24-hour activity schedule
    * Target Distribution: To sample schedules proportional to their utility ($U_s$)
    * Transitions (Operators): When the algorithm decides to modify a schedule, it randomly selects an operator
        * Swap: Switching two adjacent activity blocks
        * Inflate/Deflate: Increasing the duration of one activity while decreasing another
        * Assign: Assigning a new activity type to a block
        * Anchor, Block, & Meta: Structural operators to manage where and how changes occur
* Process:
    1. **Initialization**: The chain starts with the observed schedule ($X_0$) as the initial state
    2. **Iteration**: For each step t:
        * An operator is selected to create a candidate schedule $X^*$
        * The **Acceptance Probability ($\alpha$)** is calculated based on the ratio of utilities of the new schedule and the old one
        * If accepted, the chain moved to $X^*$, otherwise stays at $X_t$
* Sampling Strategy:
    * Total Iterations: The algorithm runs for 100,000 iterations
    * Warm-up: The first 50,000 iterations are discarded to reach a stationary distribution
    * Thinning: After the warm-up, the algorithm only saves 1 out of 20 schedules - reduce autocorrelations; prevent the sample schedules are too similar
    * Final Selection: From the saved schedules, 9 alternatives are drawn and combined with 1 observed schedule to form the final 10 schedules.
* Final Output: The final outputs are the schedules, and a Sampling Correction Term ($\ln P(C_n|i)$) - Added to the utility function during estimation to correct any bias introduced by sampling

#### Parameter Estimation

* The scheduling process is formulated as a **discrete choice model** where alternatives are full daily schedules
* MLE on a sampled choice set requires a **correction term** $\ln P_n(\tilde{C}_n|i)$ to account for sampling bias [(Ben-Akiva and Lerman, 1985)](https://books.google.co.uk/books?hl=zh-CN&lr=&id=oLC6ZYPs9UoC&oi=fnd&pg=PR11&ots=nPgznY6pEd&sig=mN6-rhvaZWJtTPeCm_976qNbX08&redir_esc=y#v=onepage&q&f=false)
* Error terms assumed i.i.d. Extreme Value --> reduces to a **logit model** (IIA assumption)
* Models estimated using **PandasBiogeme**, with 70% of observations for training

#### Model Specifications Compared

| Model | Choice Set | Utility Specification |
|-------|-----------|----------------------|
| Benchmark 1 | N/A (literature params) | Flexibility-level, linear penalties |
| Benchmark 2 | Random generation | Activity-specific, linear penalties |
| Benchmark 3 | Empirical clustering | Activity-specific, linear penalties |
| Model 1 | MH-sampled | Flexibility-level (F / NF), linear penalties |
| Model 2 | MH-sampled | Activity-specific, linear penalties |
| Model 3 | MH-sampled | S-shaped MATSim scoring |

### How are data used to test them?

* **Data source**: Swiss Mobility and Transport Microcensus (MTMC), 2015 edition
    * A nationwide survey of mobility behaviours of Swiss residents
    * Contains 57,090 individuals and 43,630 trip diaries
    * Records socio-economic characteristics (age, gender, income) and detailed trip records for one reference day
* **Sample**: For illustration purposes, the study focuses on **236 full-time students residing in Lausanne**
* **Activities modelled**: home, work, education, leisure, and shopping
* **Scheduling preferences** (desired start times and durations) are derived from the dataset by fitting normal or log-normal distributions across the student population
* **Travel parameters** are not estimated in this study and are set to null. Travel is treated as associated with the origin activity, not as a standalone activity
* **Limitations of the data**:
    * Small sample size (236 individuals from one specific population who are students in Lausanne) limits generalisability
    * Only one reference day per person; no multi-day behaviour captured
    * Desired times are approximated from distributional fits, which impose unimodal assumptions, where the paper later acknowledges this is too restrictive (e.g. education start times are clearly bimodal in reality)
    * No individual-level constraint information available (unlike [Xu et al. (2017)](https://www.sciencedirect.com/science/article/pii/S235214651730323X))

### Main findings

* **Estimated parameters outperform literature values**: Models with estimated parameters (Models 1–3) generate average out-of-home durations closer to the observed data than Benchmark 1 (literature parameters). The simulated distribution improves with respect to activity participation and duration.
* **Strategic sampling (MH) outperforms random and empirical choice sets**:
    * The MH-sampled choice set produces more significant and behaviourally consistent parameters, even with only 10 alternatives per choice set
    * Random choice sets (Benchmark 2) require a much larger number of alternatives to achieve comparable results
    * The empirical (clustering) choice set (Benchmark 3) produces counterintuitive results (e.g. positive penalty for short work duration), suggesting inappropriate alternatives
* **Activity-specific parameters (Model 2) improve upon flexibility-level (Model 1)**: Removing the aggregation into "flexible" vs "non-flexible" categories yields better fits; notably, shopping was found to have high scheduling penalties, contradicting the assumption that it is a flexible activity
* **S-shaped MATSim specification (Model 3) has mixed results**: All parameters are significant, and the model captures realistic proportions of activity participation; however, it fails to properly model time-of-day frequency for most activities because it does not include start time preferences — start time effects are clearly significant
* **All models over-generate fully-at-home days** (~5x more than observed): This is attributed to the IIA assumption of the logit model, which does not account for correlations between alternatives or unobserved factors influencing the decision to leave home
* **Education start time bimodality not captured**: All estimated models assume unimodal desired start times, but the observed distribution is clearly bimodal; the OASIS models peak at ~9:00 instead of the observed ~11:00 peak
* **Leisure parameters are largely insignificant** (Model 2), suggesting leisure is not a strongly time-constrained activity for students

## Discuss!

* The paper makes a clear methodological contribution by integrating the MH algorithm for choice set generation with MLE for parameter estimation in a simultaneous activity-based framework. This is a meaningful step forward from prior work where parameters were either borrowed from literature or estimated with biased choice sets.
* The comparison across 6 model specifications is well-structured and provides useful insights into the trade-offs between model complexity, data requirements, and estimation quality.
* The finding that strategic sampling (MH) achieves good results even with a small choice set (10 alternatives) is practically important. It suggests the approach can work with limited computational budgets.
* The tension between the linear and S-shaped specifications is interesting: the S-shape is more behaviourally realistic for duration but loses start-time effects. A combined specification could be promising.
* The framework's modularity is a key strength: new operators, constraints, or utility specifications can be added without changing the core methodology. This makes OASIS potentially adaptable to different domains (energy, epidemiology) as claimed.

## Still confused about? - Limitation part

* Why are travel parameters set to null? This seems like a major simplification given that travel time/cost is a critical component of scheduling trade-offs. How much does this affect the validity of the estimated activity parameters?
* The logit model (IIA assumption) is acknowledged as too restrictive — but how far can mixed logit or latent class models actually go in addressing the over-generation of at-home schedules? Is this a specification issue or a more fundamental problem with the utility framework?
* The sample is restricted to 236 students in Lausanne. How well does this methodology scale to larger, more heterogeneous populations? The paper mentions computational costs briefly (2.22 min for all 236 individuals) but does not discuss scalability in depth.
* The unimodal distributional assumptions for desired times are clearly a limitation — would a mixture distribution or a non-parametric approach be feasible within the existing framework?
* The sampling correction term ($\ln P(C_n|i)$) derivation is relatively complex — how sensitive are the parameter estimates to errors or approximations in this correction?

## Based on the above, why was I encouraged to read this?

* This paper sits at the intersection of **activity-based modelling**, **discrete choice theory**, and **MCMC methods**, understanding it provides a foundation for how advanced transport demand models are being developed
* It demonstrates a practical methodology for **parameter estimation in combinatorial choice problems** where full enumeration is infeasible — relevant beyond transport (e.g. energy scheduling, urban simulation)
* The use of the **Metropolis-Hastings algorithm for choice set generation** is a transferable technique that connects to broader literature on sampling methods in econometrics
* It highlights the importance of **choice set construction** on parameter estimation quality, which is a fundamental lesson for any discrete choice modelling application
* The OASIS framework is open-source (GitHub), which means the methodology can be inspected, replicated, and extended — making it a useful reference for research projects
* Tim Hillel (UCL) is one of the co-authors, which likely makes it directly relevant to the module or research group context
