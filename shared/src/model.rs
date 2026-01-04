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
pub struct Normalizer<B: Backend> {
    pub mean: Param<Tensor<B, 2>>,
    pub var: Param<Tensor<B, 2>>,
    pub count: Param<Tensor<B, 1>>,
}

impl<B: Backend> Normalizer<B> {
    pub fn new(input_dimension: usize, device: &B::Device) -> Self {
        Self {
            mean: Param::from_tensor(Tensor::zeros([1, input_dimension], device)),
            var: Param::from_tensor(Tensor::ones([1, input_dimension], device)),
            count: Param::from_tensor(Tensor::from_floats([1e-4], device)),
        }
    }

    pub fn normalize(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let mean = self.mean.val();
        let var = self.var.val();
        (x - mean.detach()) / (var.detach().sqrt().add_scalar(1e-8))
    }
}

#[derive(Module, Debug)]
pub struct Agent<B: Backend> {
    pub model: ActorCritic<B>,
    pub normalizer: Normalizer<B>,
}

impl<B: Backend> Agent<B> {
    pub fn new(input_dimension: usize, action_dimension: usize, device: &B::Device) -> Self {
        Self {
            model: ActorCritic::new(input_dimension, action_dimension, device),
            normalizer: Normalizer::new(input_dimension, device),
        }
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
