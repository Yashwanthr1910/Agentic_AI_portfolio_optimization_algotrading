# Agent 1 — Stock Selection Agent

## Overview

Agent 1 is the **Stock Selection Agent** of the Agentic AI Portfolio Optimization and Algorithmic Trading system.

Its purpose is to reduce the broad NIFTY-500 stock universe to a smaller set of candidate stocks that can be passed to Agent 2 for deep-learning-based trend prediction.

The architecture is inspired by the research paper:

**“Agentic AI-driven portfolio optimization: a hybrid approach for optimized stock selection and deep learning in algorithmic trading”**

The central paper-inspired Agent-1 idea is:

Decision Tree
→ Binary Genetic Siberian Tiger Optimization
→ Selected Stocks

In our implementation, the optimization stage is a reproducible **BGSTO-style approximation/reconstruction** rather than a claim of exact reproduction of every equation and search operator in the paper.

---

# Position in the Complete System

```text
NIFTY-500 Historical Data
          |
          v
       AGENT 1
    STOCK SELECTION
          |
          | Decision Tree
          | +
          | BGSTO-style optimization
          |
          v
       AGENT 2
    TREND PREDICTION
          |
          v
       AGENT 3
    RISK MANAGEMENT
          |
          v
       AGENT 4
    DQN TRADE EXECUTION
          |
          v
       AGENT 5
 PORTFOLIO REBALANCING
          |
          v
    FINAL PORTFOLIO