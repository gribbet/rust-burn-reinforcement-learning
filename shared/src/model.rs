use burn::{
    module::Param,
    nn::{Linear, LinearConfig},
    prelude::*,
};

#[derive(Module, Debug)]
pub struct ActorCritic<B: Backend> {
    pub actor: MultiLayerPerceptron<B>,
    pub critic: MultiLayerPerceptron<B>,
    pub log_standard_deviation: Param<Tensor<B, 1>>,
}

impl<B: Backend> ActorCritic<B> {
    pub fn new(input_dimension: usize, action_dimension: usize, device: &B::Device) -> Self {
        let hidden_dimension = 128;
        Self {
            actor: MultiLayerPerceptron::new(
                input_dimension,
                action_dimension,
                hidden_dimension,
                device,
            ),
            critic: MultiLayerPerceptron::new(input_dimension, 1, hidden_dimension, device),
            log_standard_deviation: Param::from_tensor(Tensor::zeros([action_dimension], device)),
        }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 1>, Tensor<B, 2>) {
        let mean = self.actor.forward(x.clone());
        let value = self.critic.forward(x);
        (mean, self.log_standard_deviation.val(), value)
    }
}

#[derive(Module, Debug)]
pub struct MultiLayerPerceptron<B: Backend> {
    layer_1: Linear<B>,
    layer_2: Linear<B>,
    layer_3: Linear<B>,
}

impl<B: Backend> MultiLayerPerceptron<B> {
    pub fn new(input: usize, output: usize, hidden: usize, device: &B::Device) -> Self {
        Self {
            layer_1: LinearConfig::new(input, hidden).init(device),
            layer_2: LinearConfig::new(hidden, hidden).init(device),
            layer_3: LinearConfig::new(hidden, output).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.layer_1.forward(x).tanh();
        let x = self.layer_2.forward(x).tanh();
        self.layer_3.forward(x)
    }
}
