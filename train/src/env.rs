use burn::prelude::*;
use burn::tensor::{backend::AutodiffBackend, Distribution, Int};
use shared::physics::{CartPolePhysics, PhysicsState};
use std::f32::consts::PI;

pub trait EnvironmentState<B: Backend> {
    fn inner<AD: AutodiffBackend<InnerBackend = B>>(state: PhysicsState<AD>) -> Self;
    fn from_inner<AD: AutodiffBackend<InnerBackend = B>>(state: Self) -> PhysicsState<AD>;
    fn mask_where(self, mask: Tensor<B, 1, Bool>, other: Self) -> Self;
}

impl<B: Backend> EnvironmentState<B> for PhysicsState<B> {
    fn inner<AD: AutodiffBackend<InnerBackend = B>>(state: PhysicsState<AD>) -> Self {
        Self {
            cart_position: state.cart_position.inner(),
            cart_velocity: state.cart_velocity.inner(),
            pole_angle: state.pole_angle.inner(),
            pole_angular_velocity: state.pole_angular_velocity.inner(),
            time: state.time.inner(),
        }
    }

    fn from_inner<AD: AutodiffBackend<InnerBackend = B>>(state: Self) -> PhysicsState<AD> {
        PhysicsState {
            cart_position: Tensor::from_inner(state.cart_position),
            cart_velocity: Tensor::from_inner(state.cart_velocity),
            pole_angle: Tensor::from_inner(state.pole_angle),
            pole_angular_velocity: Tensor::from_inner(state.pole_angular_velocity),
            time: Tensor::from_inner(state.time),
        }
    }

    fn mask_where(self, mask: Tensor<B, 1, Bool>, other: Self) -> Self {
        let mask = &mask;
        Self {
            cart_position: self.cart_position.mask_where(mask.clone(), other.cart_position),
            cart_velocity: self.cart_velocity.mask_where(mask.clone(), other.cart_velocity),
            pole_angle: self.pole_angle.mask_where(mask.clone(), other.pole_angle),
            pole_angular_velocity: self
                .pole_angular_velocity
                .mask_where(mask.clone(), other.pole_angular_velocity),
            time: self.time.mask_where(mask.clone(), other.time),
        }
    }
}

pub struct TrainingEnv {
    pub physics: CartPolePhysics,
    pub position_threshold: f32,
    pub max_steps: i32,
}

pub struct TrainingStep<B: Backend> {
    pub observation: Tensor<B, 2>,
    pub state: PhysicsState<B>,
    pub reward: Tensor<B, 1>,
    pub done: Tensor<B, 1, Int>,
    pub success: Tensor<B, 1, Int>,
}

impl TrainingEnv {
    pub fn new() -> Self {
        Self { physics: CartPolePhysics::default(), position_threshold: 2.4, max_steps: 500 }
    }

    pub fn reset<B: Backend>(
        &self,
        environments_count: usize,
        device: &B::Device,
    ) -> (Tensor<B, 2>, PhysicsState<B>) {
        let cart_position = Tensor::<B, 1>::random(
            [environments_count],
            Distribution::Uniform(-0.05, 0.05),
            device,
        );
        let cart_velocity = Tensor::<B, 1>::random(
            [environments_count],
            Distribution::Uniform(-0.05, 0.05),
            device,
        );
        let pole_angle = Tensor::<B, 1>::random(
            [environments_count],
            Distribution::Uniform(-(PI as f64), PI as f64),
            device,
        );
        let pole_angular_velocity =
            Tensor::<B, 1>::random([environments_count], Distribution::Uniform(-0.5, 0.5), device);
        let time = Tensor::<B, 1, Int>::zeros([environments_count], device);

        let state =
            PhysicsState { cart_position, cart_velocity, pole_angle, pole_angular_velocity, time };

        (self.physics.get_observation(&state), state)
    }

    pub fn step<B: Backend>(
        &self,
        state: PhysicsState<B>,
        action: Tensor<B, 1>,
    ) -> TrainingStep<B> {
        let clamped_action = action.clone().clamp(-1.0, 1.0);
        let next_state = self.physics.step(state, action);

        let reward_upright = next_state.pole_angle.clone().cos();
        let reward_center = next_state.cart_position.clone().powf_scalar(2.0) * -0.5;
        let reward_effort = clamped_action.powf_scalar(2.0) * -0.1;
        let reward = reward_upright + reward_center + reward_effort;

        let done_position =
            next_state.cart_position.clone().abs().greater_elem(self.position_threshold);
        let done_time = next_state.time.clone().greater_equal_elem(self.max_steps);
        let done = done_position.bool_or(done_time);

        let upright_threshold_radians = 12.0 * PI / 180.0;
        let success = next_state
            .pole_angle
            .clone()
            .abs()
            .lower_elem(upright_threshold_radians)
            .bool_and(next_state.pole_angular_velocity.clone().abs().lower_elem(1.0))
            .bool_and(next_state.time.clone().greater_elem(self.max_steps - 50))
            .bool_and(done.clone());

        TrainingStep {
            observation: self.physics.get_observation(&next_state),
            state: next_state,
            reward,
            done: done.int(),
            success: success.int(),
        }
    }
}
