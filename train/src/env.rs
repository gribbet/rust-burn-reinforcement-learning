use burn::prelude::*;
use burn::tensor::Distribution;
use burn::tensor::{backend::AutodiffBackend, Int};
use shared::physics::{PhysicsState, WalkerPhysics};

pub trait EnvironmentState<B: Backend> {
    fn inner<AD: AutodiffBackend<InnerBackend = B>>(state: PhysicsState<AD>) -> Self;
    fn from_inner<AD: AutodiffBackend<InnerBackend = B>>(state: Self) -> PhysicsState<AD>;
    fn mask_where(self, mask: Tensor<B, 1, Bool>, other: Self) -> Self;
}

impl<B: Backend> EnvironmentState<B> for PhysicsState<B> {
    fn inner<AD: AutodiffBackend<InnerBackend = B>>(state: PhysicsState<AD>) -> Self {
        Self {
            x: state.x.inner(),
            y: state.y.inner(),
            angles: state.angles.inner(),
            vx: state.vx.inner(),
            vy: state.vy.inner(),
            v_angles: state.v_angles.inner(),
            time: state.time.inner(),
            target_velocity: state.target_velocity.inner(),
        }
    }

    fn from_inner<AD: AutodiffBackend<InnerBackend = B>>(state: Self) -> PhysicsState<AD> {
        PhysicsState {
            x: Tensor::from_inner(state.x),
            y: Tensor::from_inner(state.y),
            angles: Tensor::from_inner(state.angles),
            vx: Tensor::from_inner(state.vx),
            vy: Tensor::from_inner(state.vy),
            v_angles: Tensor::from_inner(state.v_angles),
            time: Tensor::from_inner(state.time),
            target_velocity: Tensor::from_inner(state.target_velocity),
        }
    }

    fn mask_where(self, mask: Tensor<B, 1, Bool>, other: Self) -> Self {
        let mask_2d = mask.clone().unsqueeze_dim::<2>(1);
        let angles_shape = self.angles.shape();
        Self {
            x: self.x.mask_where(mask.clone(), other.x),
            y: self.y.mask_where(mask.clone(), other.y),
            angles: self
                .angles
                .mask_where(mask_2d.clone().expand(angles_shape.clone()), other.angles),
            vx: self.vx.mask_where(mask.clone(), other.vx),
            vy: self.vy.mask_where(mask.clone(), other.vy),
            v_angles: self.v_angles.mask_where(mask_2d.expand(angles_shape), other.v_angles),
            time: self.time.mask_where(mask.clone(), other.time),
            target_velocity: self.target_velocity.mask_where(mask, other.target_velocity),
        }
    }
}

pub struct TrainingEnv {
    pub physics: WalkerPhysics,
    pub max_steps: i32,
}

pub struct TrainingStep<B: Backend> {
    pub observation: Tensor<B, 2>,
    pub state: PhysicsState<B>,
    pub reward: Tensor<B, 1>,
    pub done: Tensor<B, 1, Int>,
    pub is_fallen: Tensor<B, 1, Int>, // Add this
}

impl Default for TrainingEnv {
    fn default() -> Self {
        Self { physics: WalkerPhysics::default(), max_steps: 1024 }
    }
}

impl TrainingEnv {
    pub fn reset<B: Backend>(
        &self,
        environments_count: usize,
        device: &B::Device,
    ) -> (Tensor<B, 2>, PhysicsState<B>) {
        let mut state = self.physics.initial_state(environments_count, device);

        state.target_velocity =
            Tensor::<B, 1>::random([environments_count], Distribution::Uniform(1.0, 1.0), device);

        (self.physics.get_observation(&state), state)
    }

    pub fn step<B: Backend>(
        &self,
        state: PhysicsState<B>,
        action: Tensor<B, 2>,
    ) -> TrainingStep<B> {
        let action = action.clamp(-1.0, 1.0);

        // Run physics step (internally handles sub-steps)
        let next_state = self.physics.step(state.clone(), action.clone());

        // Done conditions
        let is_fallen = next_state.y.clone().lower_equal_elem(1.0);

        // Reward function
        let distance = next_state.x.clone() - state.x.clone();
        let progress = (distance.clone() * state.target_velocity.clone()) * 100.0; // Reward based on target direction
        let progress = progress.mask_where(is_fallen.clone(), Tensor::zeros_like(&distance));

        let torque_penalty = action.powf_scalar(2.0).sum_dim(1).squeeze_dim(1) * -1.0; // Efficiency penalty

        let survival = (Tensor::ones_like(&progress) * 1.0)
            .mask_where(is_fallen.clone(), Tensor::ones_like(&progress) * -100.0);
        let reward = progress + torque_penalty;

        let is_max_steps = next_state.time.clone().greater_equal_elem(self.max_steps);

        let done = is_fallen.clone().bool_or(is_max_steps).int();

        TrainingStep {
            observation: self.physics.get_observation(&next_state),
            state: next_state,
            reward,
            done,
            is_fallen: is_fallen.int(),
        }
    }
}
