use burn::prelude::*;
use std::f32::consts::PI;

pub struct DiagonalGaussian<B: Backend> {
    mean: Tensor<B, 2>,
    log_std: Tensor<B, 2>,
}

impl<B: Backend> DiagonalGaussian<B> {
    pub fn new(mean: Tensor<B, 2>, log_std: Tensor<B, 1>) -> Self {
        let log_std = log_std.unsqueeze_dim::<2>(0);
        Self { mean, log_std }
    }

    pub fn sample(&self, noise: Tensor<B, 2>) -> Tensor<B, 2> {
        self.mean.clone() + noise * self.log_std.clone().exp()
    }

    pub fn log_probability(&self, action: Tensor<B, 2>) -> Tensor<B, 1> {
        let variance = self.log_std.clone().mul_scalar(2.0).exp();
        let log_two_pi = (2.0 * PI).ln();

        let difference = action - self.mean.clone();
        (difference.powf_scalar(2.0) / variance + self.log_std.clone().mul_scalar(2.0))
            .add_scalar(log_two_pi)
            .mul_scalar(-0.5)
            .sum_dim(1)
            .squeeze_dim(1)
    }

    pub fn entropy(&self) -> Tensor<B, 1> {
        self.log_std
            .clone()
            .mul_scalar(2.0)
            .add_scalar((2.0 * PI).ln())
            .add_scalar(1.0)
            .mul_scalar(0.5)
            .mean_dim(1)
            .squeeze_dim(1)
    }
}
