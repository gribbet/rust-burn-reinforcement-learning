# Rust Burn Reinforcement Learning

A high-performance Reinforcement Learning implementation in Rust using the [Burn](https://github.com/tracel-ai/burn) deep learning framework. This project implements the **Proximal Policy Optimization (PPO)** algorithm to solve the classic **CartPole** environment.

## Getting Started

### Training

To start the training process:

```bash
cargo run --release --bin train
```

### Running the simulation

Once you have trained a model, you can visualize the agent's behavior:

```bash
cargo run --release --bin viz
```