use burn::prelude::*;
use burn::tensor::Distribution;
use burn::tensor::{backend::AutodiffBackend, Int};
use shared::physics::{BipedalWalkerPhysics, PhysicsState};

pub trait EnvironmentState<B: Backend> {
    fn inner<AD: AutodiffBackend<InnerBackend = B>>(state: PhysicsState<AD>) -> Self;
    fn from_inner<AD: AutodiffBackend<InnerBackend = B>>(state: Self) -> PhysicsState<AD>;
    fn mask_where(self, mask: Tensor<B, 1, Bool>, other: Self) -> Self;
}

impl<B: Backend> EnvironmentState<B> for PhysicsState<B> {
    fn inner<AD: AutodiffBackend<InnerBackend = B>>(state: PhysicsState<AD>) -> Self {
        Self {
            hull_x: state.hull_x.inner(),
            hull_y: state.hull_y.inner(),
            hull_angle: state.hull_angle.inner(),
            hull_vx: state.hull_vx.inner(),
            hull_vy: state.hull_vy.inner(),
            hull_v_angle: state.hull_v_angle.inner(),
            left_hip_angle: state.left_hip_angle.inner(),
            left_hip_v: state.left_hip_v.inner(),
            left_knee_angle: state.left_knee_angle.inner(),
            left_knee_v: state.left_knee_v.inner(),
            right_hip_angle: state.right_hip_angle.inner(),
            right_hip_v: state.right_hip_v.inner(),
            right_knee_angle: state.right_knee_angle.inner(),
            right_knee_v: state.right_knee_v.inner(),
            left_contact: state.left_contact.inner(),
            right_contact: state.right_contact.inner(),
            time: state.time.inner(),
            target_velocity: state.target_velocity.inner(),
        }
    }

    fn from_inner<AD: AutodiffBackend<InnerBackend = B>>(state: Self) -> PhysicsState<AD> {
        PhysicsState {
            hull_x: Tensor::from_inner(state.hull_x),
            hull_y: Tensor::from_inner(state.hull_y),
            hull_angle: Tensor::from_inner(state.hull_angle),
            hull_vx: Tensor::from_inner(state.hull_vx),
            hull_vy: Tensor::from_inner(state.hull_vy),
            hull_v_angle: Tensor::from_inner(state.hull_v_angle),
            left_hip_angle: Tensor::from_inner(state.left_hip_angle),
            left_hip_v: Tensor::from_inner(state.left_hip_v),
            left_knee_angle: Tensor::from_inner(state.left_knee_angle),
            left_knee_v: Tensor::from_inner(state.left_knee_v),
            right_hip_angle: Tensor::from_inner(state.right_hip_angle),
            right_hip_v: Tensor::from_inner(state.right_hip_v),
            right_knee_angle: Tensor::from_inner(state.right_knee_angle),
            right_knee_v: Tensor::from_inner(state.right_knee_v),
            left_contact: Tensor::from_inner(state.left_contact),
            right_contact: Tensor::from_inner(state.right_contact),
            time: Tensor::from_inner(state.time),
            target_velocity: Tensor::from_inner(state.target_velocity),
        }
    }

    fn mask_where(self, mask: Tensor<B, 1, Bool>, other: Self) -> Self {
        let mask = &mask;
        Self {
            hull_x: self.hull_x.mask_where(mask.clone(), other.hull_x),
            hull_y: self.hull_y.mask_where(mask.clone(), other.hull_y),
            hull_angle: self.hull_angle.mask_where(mask.clone(), other.hull_angle),
            hull_vx: self.hull_vx.mask_where(mask.clone(), other.hull_vx),
            hull_vy: self.hull_vy.mask_where(mask.clone(), other.hull_vy),
            hull_v_angle: self.hull_v_angle.mask_where(mask.clone(), other.hull_v_angle),
            left_hip_angle: self.left_hip_angle.mask_where(mask.clone(), other.left_hip_angle),
            left_hip_v: self.left_hip_v.mask_where(mask.clone(), other.left_hip_v),
            left_knee_angle: self.left_knee_angle.mask_where(mask.clone(), other.left_knee_angle),
            left_knee_v: self.left_knee_v.mask_where(mask.clone(), other.left_knee_v),
            right_hip_angle: self.right_hip_angle.mask_where(mask.clone(), other.right_hip_angle),
            right_hip_v: self.right_hip_v.mask_where(mask.clone(), other.right_hip_v),
            right_knee_angle: self
                .right_knee_angle
                .mask_where(mask.clone(), other.right_knee_angle),
            right_knee_v: self.right_knee_v.mask_where(mask.clone(), other.right_knee_v),
            left_contact: self.left_contact.mask_where(mask.clone(), other.left_contact),
            right_contact: self.right_contact.mask_where(mask.clone(), other.right_contact),
            time: self.time.mask_where(mask.clone(), other.time),
            target_velocity: self.target_velocity.mask_where(mask.clone(), other.target_velocity),
        }
    }
}

pub struct TrainingEnv {
    pub physics: BipedalWalkerPhysics,
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
        Self { physics: BipedalWalkerPhysics::default(), max_steps: 1024 }
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
            (Tensor::<B, 1>::random([environments_count], Distribution::Uniform(1.0, 1.0), device));

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
        let is_fallen = next_state.hull_y.clone().lower_equal_elem(0.8);

        // Reward function
        let distance = next_state.hull_x.clone() - state.hull_x.clone();
        let progress = (distance.clone() * state.target_velocity.clone()) * 100.0; // Reward based on target direction
        let progress = progress.mask_where(is_fallen.clone(), Tensor::zeros_like(&distance));
        let still_penalty =
            distance.powf_scalar(2.0) * (1.0 - state.target_velocity.clone().abs()) * -0.5;

        let torque_penalty = action.powf_scalar(2.0).sum_dim(1).squeeze_dim(1) * 0.0; // Efficiency penalty

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
