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
            last_action: state.last_action.inner(),
        }
    }

    fn from_inner<AD: AutodiffBackend<InnerBackend = B>>(state: Self) -> PhysicsState<AD> {
        PhysicsState {
            positions: Tensor::from_inner(state.positions),
            velocities: Tensor::from_inner(state.velocities),
            time: Tensor::from_inner(state.time),
            target_velocity: Tensor::from_inner(state.target_velocity),
            last_action: Tensor::from_inner(state.last_action),
        }
    }

    fn mask_where(self, mask: Tensor<B, 1, Bool>, other: Self) -> Self {
        let mask_2d = mask.clone().unsqueeze_dim::<2>(1);
        let mask_3d = mask_2d.clone().unsqueeze_dim::<3>(2);
        let pos_shape = self.positions.shape();
        let vel_shape = self.velocities.shape();
        let action_shape = self.last_action.shape();

        Self {
            positions: self
                .positions
                .mask_where(mask_3d.clone().expand(pos_shape), other.positions),
            velocities: self.velocities.mask_where(mask_3d.expand(vel_shape), other.velocities),
            time: self.time.mask_where(mask.clone(), other.time),
            target_velocity: self.target_velocity.mask_where(mask.clone(), other.target_velocity),
            last_action: self
                .last_action
                .mask_where(mask_2d.expand(action_shape), other.last_action),
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
            Tensor::random([environments_count], Distribution::Uniform(-0.5, 1.5), device);

        (self.walker.get_observation(&state), state)
    }

    pub fn step(&self, state: PhysicsState<B>, action: Tensor<B, 2>) -> TrainingStep<B> {
        let action = action.tanh();
        let WalkerConfig { time_step, fall_y, .. } = self.walker.config;

        // Run physics step (internally handles sub-steps)
        let next_state = self.walker.step(state.clone(), action.clone());

        // Done conditions
        let batch_size = next_state.positions.dims()[0];
        let root_pos_next =
            next_state.positions.clone().slice([0..batch_size, 0..1, 0..2]).squeeze_dim::<2>(1);
        let root_y = root_pos_next.clone().slice([0..batch_size, 1..2]).squeeze_dim::<1>(1);
        let is_fallen = root_y.lower_equal_elem(fall_y);

        let com_next = self.walker.center_of_mass(next_state.positions.clone());
        let com_prev = self.walker.center_of_mass(state.positions.clone());

        let com_x_next = com_next.clone().slice([0..batch_size, 0..1]).squeeze_dim::<1>(1);
        let com_x_prev = com_prev.clone().slice([0..batch_size, 0..1]).squeeze_dim::<1>(1);

        let distance = com_x_next.clone() - com_x_prev;
        let velocity = distance / time_step;

        let velocity_reward =
            (velocity - state.target_velocity.clone()).powf_scalar(2.0).mul_scalar(-4.0).exp();

        let torque_penalty = action.powf_scalar(2.0).sum_dim(1).squeeze_dim::<1>(1) * -0.1; // Increased efficiency penalty

        let reward = velocity_reward + torque_penalty;

        // Sanitize reward: replace NaN with 0.0
        let is_finite = reward.clone().equal(reward.clone());
        let reward = reward.clone().mask_where(is_finite.bool_not(), Tensor::zeros_like(&reward));

        let is_max_time = next_state.time.clone().greater_equal_elem(self.max_time);
        let is_goal = com_x_next.greater_equal_elem(10.0);

        let done = is_fallen.clone().bool_or(is_max_time).bool_or(is_goal).int();

        TrainingStep {
            observation: self.walker.get_observation(&next_state),
            state: next_state,
            reward,
            done,
            is_fallen: is_fallen.int(),
        }
    }
}
