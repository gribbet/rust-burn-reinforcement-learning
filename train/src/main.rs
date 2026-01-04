mod distribution;
mod env;
mod ppo;

use crate::ppo::{train, ProximalPolicyOptimizationConfig};
use burn::backend::libtorch::LibTorchDevice;
use burn::backend::{Autodiff, LibTorch};

fn main() {
    let device = LibTorchDevice::Mps;
    let config = ProximalPolicyOptimizationConfig::new();
    let iterations = 100000;

    train::<Autodiff<LibTorch>>(device, config, iterations);
}
