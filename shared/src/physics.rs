use burn::prelude::*;
use burn::tensor::Int;
use std::f32::consts::PI;

#[derive(Clone, Debug)]
pub struct PhysicsState<B: Backend> {
    pub cart_position: Tensor<B, 1>,
    pub cart_velocity: Tensor<B, 1>,
    pub pole_angle: Tensor<B, 1>,
    pub pole_angular_velocity: Tensor<B, 1>,
    pub time: Tensor<B, 1, Int>,
}

pub struct CartPolePhysics {
    pub gravity: f32,
    pub pole_mass: f32,
    pub cart_mass: f32,
    pub pole_length: f32,
    pub force_magnitude: f32,
    pub time_step: f32,
}

impl Default for CartPolePhysics {
    fn default() -> Self {
        Self {
            gravity: 9.8,
            pole_mass: 0.1,
            cart_mass: 1.0,
            pole_length: 1.0,
            force_magnitude: 10.0,
            time_step: 0.02,
        }
    }
}

impl CartPolePhysics {
    pub fn step<B: Backend>(
        &self,
        state: PhysicsState<B>,
        action: Tensor<B, 1>,
    ) -> PhysicsState<B> {
        let total_mass = self.pole_mass + self.cart_mass;
        let half_pole_length = self.pole_length / 2.0;
        let pole_mass_length = self.pole_mass * half_pole_length;

        let cart_position = state.cart_position;
        let cart_velocity = state.cart_velocity;
        let pole_angle = state.pole_angle;
        let pole_angular_velocity = state.pole_angular_velocity;

        let force = action.clamp(-1.0, 1.0) * self.force_magnitude;
        let cos_theta = pole_angle.clone().cos();
        let sin_theta = pole_angle.clone().sin();

        let temp = (force
            + pole_angular_velocity.clone().powf_scalar(2.0)
                * pole_mass_length
                * sin_theta.clone())
            / total_mass;

        let theta_acceleration_numerator =
            sin_theta.clone() * self.gravity - cos_theta.clone() * temp.clone();
        let theta_acceleration_denominator = (cos_theta.clone().powf_scalar(2.0)
            * -(self.pole_mass / total_mass))
            .add_scalar(4.0 / 3.0)
            * half_pole_length;
        let theta_acceleration = theta_acceleration_numerator / theta_acceleration_denominator;

        let x_acceleration =
            temp - theta_acceleration.clone() * pole_mass_length * cos_theta.clone() / total_mass;

        let next_cart_position = cart_position + cart_velocity.clone() * self.time_step;
        let next_cart_velocity = cart_velocity + x_acceleration * self.time_step;
        let next_pole_angle = pole_angle + pole_angular_velocity.clone() * self.time_step;
        let next_pole_angular_velocity =
            pole_angular_velocity + theta_acceleration * self.time_step;

        let next_pole_angle =
            (next_pole_angle.add_scalar(PI)).remainder_scalar(2.0 * PI).sub_scalar(PI);

        let next_time = state.time.add_scalar(1);

        PhysicsState {
            cart_position: next_cart_position,
            cart_velocity: next_cart_velocity,
            pole_angle: next_pole_angle,
            pole_angular_velocity: next_pole_angular_velocity,
            time: next_time,
        }
    }

    pub fn get_observation<B: Backend>(&self, state: &PhysicsState<B>) -> Tensor<B, 2> {
        Tensor::cat(
            vec![
                state.cart_position.clone().unsqueeze_dim(1),
                state.cart_velocity.clone().unsqueeze_dim(1),
                state.pole_angle.clone().sin().unsqueeze_dim(1),
                state.pole_angle.clone().cos().unsqueeze_dim(1),
                state.pole_angular_velocity.clone().unsqueeze_dim(1),
            ],
            1,
        )
    }
}
