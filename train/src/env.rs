use burn::prelude::*;
use burn::tensor::Distribution;
use burn::tensor::{backend::AutodiffBackend, Int};
use shared::physics::{PhysicsState, Walker, WalkerConfig};

pub trait EnvironmentState<B: Backend> {
    fn inner<AD: AutodiffBackend<InnerBackend = B>>(state: PhysicsState<AD>) -> Self;
    fn from_inner<AD: AutodiffBackend<InnerBackend = B>>(state: Self) -> PhysicsState<AD>;
    fn mask_where(self, mask: Tensor<B, 1, Bool>, other: Self) -> Self;
}

impl<B: Backend> EnvironmentState<B> for PhysicsState<B> {
    fn inner<AD: AutodiffBackend<InnerBackend = B>>(state: PhysicsState<AD>) -> Self {
        Self {
            positions: state.positions.inner(),
            velocities: state.velocities.inner(),
            time: state.time.inner(),
            target_velocity: state.target_velocity.inner(),
            fallen_time: state.fallen_time.inner(),
        }
    }

    fn from_inner<AD: AutodiffBackend<InnerBackend = B>>(state: Self) -> PhysicsState<AD> {
        PhysicsState {
            positions: Tensor::from_inner(state.positions),
            velocities: Tensor::from_inner(state.velocities),
            time: Tensor::from_inner(state.time),
            target_velocity: Tensor::from_inner(state.target_velocity),
            fallen_time: Tensor::from_inner(state.fallen_time),
        }
    }

    fn mask_where(self, mask: Tensor<B, 1, Bool>, other: Self) -> Self {
        let mask_3d = mask.clone().unsqueeze_dim::<2>(1).unsqueeze_dim::<3>(2);
        let pos_shape = self.positions.shape();
        let vel_shape = self.velocities.shape();

        Self {
            positions: self
                .positions
                .mask_where(mask_3d.clone().expand(pos_shape), other.positions),
            velocities: self.velocities.mask_where(mask_3d.expand(vel_shape), other.velocities),
            time: self.time.mask_where(mask.clone(), other.time),
            target_velocity: self.target_velocity.mask_where(mask.clone(), other.target_velocity),
            fallen_time: self.fallen_time.mask_where(mask, other.fallen_time),
        }
    }
}

pub struct TrainingEnv<B: Backend> {
    walker: Walker<B>,
    max_time: f32,
}

pub struct TrainingStep<B: Backend> {
    pub observation: Tensor<B, 2>,
    pub state: PhysicsState<B>,
    pub reward: Tensor<B, 1>,
    pub done: Tensor<B, 1, Int>,
    pub is_fallen: Tensor<B, 1, Int>,
}

impl<B: Backend> TrainingEnv<B> {
    pub fn new(device: &B::Device, max_time: f32) -> Self {
        let config = WalkerConfig::default();
        let walker = Walker::new(config, device);
        Self { walker, max_time }
    }

    pub fn action_dim(&self) -> usize {
        self.walker.action_dim()
    }

    pub fn reset(
        &self,
        environments_count: usize,
        device: &B::Device,
    ) -> (Tensor<B, 2>, PhysicsState<B>) {
        let mut state = self.walker.initial_state(environments_count, device);

        state.target_velocity =
            Tensor::<B, 1>::random([environments_count], Distribution::Uniform(1.0, 1.0), device);

        (self.walker.get_observation(&state), state)
    }

    pub fn step(&self, state: PhysicsState<B>, action: Tensor<B, 2>) -> TrainingStep<B> {
        let action = action.tanh();

        // Run physics step (internally handles sub-steps)
        let next_state = self.walker.step(state.clone(), action.clone());

        // Done conditions
        let is_fallen = next_state.fallen_time.clone().greater_equal_elem(2.0);

        // Reward function
        let batch_size = next_state.positions.dims()[0];
        let root_pos_next =
            next_state.positions.clone().slice([0..batch_size, 0..1, 0..2]).squeeze_dim::<2>(1);
        let root_y = root_pos_next.clone().slice([0..batch_size, 1..2]).squeeze_dim::<1>(1);
        let is_currently_fallen = root_y.lower_equal_elem(self.walker.config.fall_y);

        let root_x_next = root_pos_next.clone().slice([0..batch_size, 0..1]).squeeze_dim::<1>(1);
        let root_x_prev = state
            .positions
            .clone()
            .slice([0..batch_size, 0..1, 0..1])
            .squeeze_dim::<2>(1)
            .squeeze_dim::<1>(1);

        let distance = root_x_next - root_x_prev;
        let progress = (distance.clone() * state.target_velocity.clone()) * 1.0;
        let progress = progress.mask_where(is_currently_fallen, Tensor::zeros_like(&distance));

        let torque_penalty = action.powf_scalar(2.0).sum_dim(1).squeeze_dim::<1>(1) * -0.1; // Efficiency penalty

        let reward = (progress + torque_penalty).clamp(-10.0, 10.0);

        // Sanitize reward: replace NaN with 0.0
        let is_finite = reward.clone().equal(reward.clone());
        let reward = reward.clone().mask_where(is_finite.bool_not(), Tensor::zeros_like(&reward));

        let is_max_time = next_state.time.clone().greater_equal_elem(self.max_time);

        let done = is_fallen.clone().bool_or(is_max_time).int();

        TrainingStep {
            observation: self.walker.get_observation(&next_state),
            state: next_state,
            reward,
            done,
            is_fallen: is_fallen.int(),
        }
    }
}
