use burn::prelude::*;
use burn::tensor::{Distribution, Int};

#[derive(Clone, Debug)]
pub struct PhysicsState<B: Backend> {
    pub hull_x: Tensor<B, 1>,
    pub hull_y: Tensor<B, 1>,
    pub hull_angle: Tensor<B, 1>,
    pub hull_vx: Tensor<B, 1>,
    pub hull_vy: Tensor<B, 1>,
    pub hull_v_angle: Tensor<B, 1>,
    pub left_hip_angle: Tensor<B, 1>,
    pub left_hip_v: Tensor<B, 1>,
    pub left_knee_angle: Tensor<B, 1>,
    pub left_knee_v: Tensor<B, 1>,
    pub right_hip_angle: Tensor<B, 1>,
    pub right_hip_v: Tensor<B, 1>,
    pub right_knee_angle: Tensor<B, 1>,
    pub right_knee_v: Tensor<B, 1>,
    pub left_contact: Tensor<B, 1>,
    pub right_contact: Tensor<B, 1>,
    pub time: Tensor<B, 1, Int>,
    pub target_velocity: Tensor<B, 1>,
}

pub struct BipedalWalkerPhysics {
    pub gravity: f32,
    pub hull_mass: f32,
    pub leg_mass: f32,
    pub leg_length: f32,
    pub time_step: f32,
    pub friction: f32,
    pub torque_magnitude: f32,
    pub joint_damping: f32,
    pub sub_steps: usize,
}

impl Default for BipedalWalkerPhysics {
    fn default() -> Self {
        Self {
            gravity: 9.8,
            hull_mass: 10.0,
            leg_mass: 1.0,
            leg_length: 1.0,
            time_step: 0.02, // Control interval (50Hz)
            friction: 0.3,
            torque_magnitude: 20.0, // Increased from 10.0
            joint_damping: 0.1,
            sub_steps: 2, // Increased from
        }
    }
}

impl BipedalWalkerPhysics {
    pub fn step<B: Backend>(
        &self,
        state: PhysicsState<B>,
        action: Tensor<B, 2>, // [batch, 4] torques
    ) -> PhysicsState<B> {
        let mut current_state = state;
        for _ in 0..self.sub_steps {
            current_state = self.step_internal(current_state, action.clone());
        }
        current_state
    }

    fn step_internal<B: Backend>(
        &self,
        state: PhysicsState<B>,
        action: Tensor<B, 2>, // [batch, 4] torques
    ) -> PhysicsState<B> {
        let batch_size = action.dims()[0];

        // --- 1. Initialize Particles from State ---
        // Hull Center (HC)
        let hc_x = state.hull_x.clone();
        let hc_y = state.hull_y.clone();

        // Hull Head (HH) - for orientation
        let body_len = 0.5;
        let hh_x = hc_x.clone() - state.hull_angle.clone().sin() * body_len;
        let hh_y = hc_y.clone() + state.hull_angle.clone().cos() * body_len;

        // Leg segments
        let thigh_len = self.leg_length * 0.5;
        let shank_len = self.leg_length * 0.5;

        let get_leg_points = |hip_angle: Tensor<B, 1>, knee_angle: Tensor<B, 1>| {
            let abs_hip = state.hull_angle.clone() + hip_angle;
            let knee_x = hc_x.clone() + abs_hip.clone().sin() * thigh_len;
            let knee_y = hc_y.clone() - abs_hip.clone().cos() * thigh_len;

            let abs_knee = abs_hip + knee_angle;
            let foot_x = knee_x.clone() + abs_knee.clone().sin() * shank_len;
            let foot_y = knee_y.clone() - abs_knee.clone().cos() * shank_len;

            (knee_x, knee_y, foot_x, foot_y)
        };

        let (lk_x, lk_y, lf_x, lf_y) =
            get_leg_points(state.left_hip_angle.clone(), state.left_knee_angle.clone());
        let (rk_x, rk_y, rf_x, rf_y) =
            get_leg_points(state.right_hip_angle.clone(), state.right_knee_angle.clone());

        // Velocities
        let hc_vx = state.hull_vx.clone();
        let hc_vy = state.hull_vy.clone();

        // Angular velocity contribution to points
        let get_point_vel = |px: Tensor<B, 1>, py: Tensor<B, 1>| {
            let rx = px - hc_x.clone();
            let ry = py - hc_y.clone();
            let vx = hc_vx.clone() - state.hull_v_angle.clone() * ry;
            let vy = hc_vy.clone() + state.hull_v_angle.clone() * rx;
            (vx, vy)
        };

        let (_hh_vx, _hh_vy) = get_point_vel(hh_x.clone(), hh_y.clone());

        // --- 2. Calculate Forces on Particles ---

        // Gravity
        let g = self.gravity;
        let m_hull = self.hull_mass;
        let m_leg = self.leg_mass; // Lumped mass for knee/foot

        // Ground Contact (Penalty)
        let k_g = 1000.0; // Significantly increased stiffness
        let d_g = 100.0; // Increased damping

        let ground_force = |y: Tensor<B, 1>, vx: Tensor<B, 1>, vy: Tensor<B, 1>| {
            let penetration = y.clone().mul_scalar(-1.0).clamp_min(0.0);
            let is_contact = y.lower_equal_elem(0.0);
            let fy = (penetration * k_g - vy * d_g).clamp_min(0.0);
            let fy = Tensor::zeros_like(&fy).mask_where(is_contact.clone(), fy);
            let fx = vx.tanh() * fy.clone() * self.friction * -1.0;
            (fx, fy, is_contact)
        };

        // We need velocities of knees and feet for damping
        // V_knee = V_hull + V_rot_hull + V_rot_hip
        let get_abs_vel = |hip_angle: Tensor<B, 1>,
                           knee_angle: Tensor<B, 1>,
                           hip_v: Tensor<B, 1>,
                           knee_v: Tensor<B, 1>| {
            let abs_hip = state.hull_angle.clone() + hip_angle;
            let abs_knee = abs_hip.clone() + knee_angle;

            let v_hip_rot = state.hull_v_angle.clone() + hip_v.clone();
            let v_knee_rot = v_hip_rot.clone() + knee_v;

            let knee_vx = hc_vx.clone() + v_hip_rot.clone() * abs_hip.clone().cos() * thigh_len;
            let knee_vy = hc_vy.clone() + v_hip_rot.clone() * abs_hip.clone().sin() * thigh_len;

            let foot_vx = knee_vx.clone() + v_knee_rot.clone() * abs_knee.clone().cos() * shank_len;
            let foot_vy = knee_vy.clone() + v_knee_rot.clone() * abs_knee.clone().sin() * shank_len;

            (knee_vx, knee_vy, foot_vx, foot_vy)
        };

        let (lk_vx, lk_vy, lf_vx, lf_vy) = get_abs_vel(
            state.left_hip_angle.clone(),
            state.left_knee_angle.clone(),
            state.left_hip_v.clone(),
            state.left_knee_v.clone(),
        );
        let (rk_vx, rk_vy, rf_vx, rf_vy) = get_abs_vel(
            state.right_hip_angle.clone(),
            state.right_knee_angle.clone(),
            state.right_hip_v.clone(),
            state.right_knee_v.clone(),
        );

        let (lf_fx, lf_fy, l_contact) = ground_force(lf_y.clone(), lf_vx.clone(), lf_vy.clone());
        let (rf_fx, rf_fy, r_contact) = ground_force(rf_y.clone(), rf_vx.clone(), rf_vy.clone());
        let (lk_fx, lk_fy, _lk_contact) = ground_force(lk_y.clone(), lk_vx.clone(), lk_vy.clone());
        let (rk_fx, rk_fy, _rk_contact) = ground_force(rk_y.clone(), rk_vx.clone(), rk_vy.clone());

        // Hull Collision (Hip and Head)
        let (hc_fx, hc_fy, _hc_contact) = ground_force(hc_y.clone(), hc_vx.clone(), hc_vy.clone());
        let (hh_vx, hh_vy) = get_point_vel(hh_x.clone(), hh_y.clone());
        let (hh_fx, hh_fy, _hh_contact) = ground_force(hh_y.clone(), hh_vx, hh_vy);

        // --- 3. Convert Forces to Generalized Accelerations ---
        // This is the "Articulated Body" part simplified.
        // We sum forces and torques on the Hull.
        // We calculate torques on joints from the forces on distal points.

        // Hull Forces
        let hull_fx_total = lf_fx.clone()
            + rf_fx.clone()
            + lk_fx.clone()
            + rk_fx.clone()
            + hc_fx.clone()
            + hh_fx.clone();
        let hull_fy_total = lf_fy.clone()
            + rf_fy.clone()
            + lk_fy.clone()
            + rk_fy.clone()
            + hc_fy.clone()
            + hh_fy.clone()
            - (m_hull + 4.0 * m_leg) * g; // Gravity on total mass (simplified)

        // Hull Torque from Ground Forces
        // T = r x F
        let cross_product = |rx: Tensor<B, 1>,
                             ry: Tensor<B, 1>,
                             fx: Tensor<B, 1>,
                             fy: Tensor<B, 1>| { rx * fy - ry * fx };

        // Gravity Torques on Legs (Torque = -rx * mg)
        let t_gravity = |rx: Tensor<B, 1>| rx * (-m_leg * g);

        let t_l_knee_grav = t_gravity(lf_x.clone() - lk_x.clone());
        let t_r_knee_grav = t_gravity(rf_x.clone() - rk_x.clone());

        let t_l_hip_grav =
            t_gravity(lk_x.clone() - hc_x.clone()) + t_gravity(lf_x.clone() - hc_x.clone());
        let t_r_hip_grav =
            t_gravity(rk_x.clone() - hc_x.clone()) + t_gravity(rf_x.clone() - hc_x.clone());

        // Joint Torques (Actions)
        let t_l_hip =
            action.clone().slice([0..batch_size, 0..1]).squeeze_dim(1) * self.torque_magnitude;
        let t_l_knee =
            action.clone().slice([0..batch_size, 1..2]).squeeze_dim(1) * self.torque_magnitude;
        let t_r_hip =
            action.clone().slice([0..batch_size, 2..3]).squeeze_dim(1) * self.torque_magnitude;
        let t_r_knee =
            action.clone().slice([0..batch_size, 3..4]).squeeze_dim(1) * self.torque_magnitude;

        // External Torques on Joints (Ground + Gravity)
        let t_l_knee_ext = cross_product(
            lf_x.clone() - lk_x.clone(),
            lf_y.clone() - lk_y.clone(),
            lf_fx.clone(),
            lf_fy.clone(),
        ) + t_l_knee_grav;
        let t_r_knee_ext = cross_product(
            rf_x.clone() - rk_x.clone(),
            rf_y.clone() - rk_y.clone(),
            rf_fx.clone(),
            rf_fy.clone(),
        ) + t_r_knee_grav;

        let t_l_hip_ext = cross_product(
            lk_x.clone() - hc_x.clone(),
            lk_y.clone() - hc_y.clone(),
            lk_fx.clone(),
            lk_fy.clone(),
        ) + cross_product(
            lf_x.clone() - hc_x.clone(),
            lf_y.clone() - hc_y.clone(),
            lf_fx.clone(),
            lf_fy.clone(),
        ) + t_l_hip_grav;

        let t_r_hip_ext = cross_product(
            rk_x.clone() - hc_x.clone(),
            rk_y.clone() - hc_y.clone(),
            rk_fx.clone(),
            rk_fy.clone(),
        ) + cross_product(
            rf_x.clone() - hc_x.clone(),
            rf_y.clone() - hc_y.clone(),
            rf_fx.clone(),
            rf_fy.clone(),
        ) + t_r_hip_grav;

        // Internal Torques (Motor + Limit + Damping)

        // Damping (Explicit)
        let damp_coef = 0.5;
        let t_l_hip_damp = state.left_hip_v.clone() * -damp_coef;
        let t_r_hip_damp = state.right_hip_v.clone() * -damp_coef;
        let t_l_knee_damp = state.left_knee_v.clone() * -damp_coef;
        let t_r_knee_damp = state.right_knee_v.clone() * -damp_coef;

        // Limits
        let limit_torque = |angle: Tensor<B, 1>, v: Tensor<B, 1>, min: f32, max: f32| {
            // Lower limit
            let diff_min = min - angle.clone();
            let mask_min = diff_min.clone().greater_equal_elem(0.0);
            let t_min = diff_min.clamp_min(0.0) * 1000.0 - v.clone() * 50.0;

            // Upper limit
            let diff_max = angle.clone() - max;
            let mask_max = diff_max.clone().greater_equal_elem(0.0);
            let t_max = (diff_max.clamp_min(0.0) * 1000.0 + v.clone() * 50.0) * -1.0;

            let t =
                Tensor::zeros_like(&angle).mask_where(mask_min, t_min).mask_where(mask_max, t_max);
            t
        };

        // Hips: -0.8 to 0.8 rad (~ +/- 45 deg)
        let t_l_hip_limit =
            limit_torque(state.left_hip_angle.clone(), state.left_hip_v.clone(), -0.8, 0.8);
        let t_r_hip_limit =
            limit_torque(state.right_hip_angle.clone(), state.right_hip_v.clone(), -0.8, 0.8);

        // Knees: -2.5 to 0.0 rad (-140 to 0 deg) - Knees don't bend backwards
        let t_l_knee_limit =
            limit_torque(state.left_knee_angle.clone(), state.left_knee_v.clone(), -2.5, 0.0);
        let t_r_knee_limit =
            limit_torque(state.right_knee_angle.clone(), state.right_knee_v.clone(), -2.5, 0.0);

        // Total Internal Torque (Acts on Leg, Reaction acts on Hull)
        let t_l_hip_internal = t_l_hip.clone() + t_l_hip_limit.clone() + t_l_hip_damp.clone();
        let t_r_hip_internal = t_r_hip.clone() + t_r_hip_limit.clone() + t_r_hip_damp.clone();
        let t_l_knee_internal = t_l_knee.clone() + t_l_knee_limit.clone() + t_l_knee_damp.clone();
        let t_r_knee_internal = t_r_knee.clone() + t_r_knee_limit.clone() + t_r_knee_damp.clone();

        // Hull Torque = - Sum(Hip Internal Torques) + External Hull Torques (Collision)
        // Note: Knee torques are internal to the leg (between thigh and shank), they don't act on Hull directly.
        let t_hull_collision = cross_product(
            hh_x.clone() - hc_x.clone(),
            hh_y.clone() - hc_y.clone(),
            hh_fx.clone(),
            hh_fy.clone(),
        );
        let t_hull_total =
            (t_l_hip_internal.clone() + t_r_hip_internal.clone()) * -1.0 + t_hull_collision;

        // --- 4. Integration (Semi-Implicit Euler) ---
        let dt = self.time_step / self.sub_steps as f32;

        // Hull
        let next_hull_vx = state.hull_vx + (hull_fx_total / (m_hull + 4.0 * m_leg)) * dt;
        let next_hull_vy = state.hull_vy + (hull_fy_total / (m_hull + 4.0 * m_leg)) * dt;
        let next_hull_v_angle = state.hull_v_angle + (t_hull_total / (m_hull * 0.5)) * dt;

        // Use NEW velocities for position updates
        let next_hull_x = state.hull_x + next_hull_vx.clone() * dt;
        let next_hull_y = state.hull_y + next_hull_vy.clone() * dt;
        let next_hull_angle = state.hull_angle + next_hull_v_angle.clone() * dt;

        // Joints
        let i_leg = 0.5;

        let next_l_hip_v =
            state.left_hip_v + ((t_l_hip_internal + t_l_hip_ext).clamp(-50.0, 50.0) / i_leg) * dt;
        let next_r_hip_v =
            state.right_hip_v + ((t_r_hip_internal + t_r_hip_ext).clamp(-50.0, 50.0) / i_leg) * dt;
        let next_l_knee_v = state.left_knee_v
            + ((t_l_knee_internal + t_l_knee_ext).clamp(-50.0, 50.0) / i_leg) * dt;
        let next_r_knee_v = state.right_knee_v
            + ((t_r_knee_internal + t_r_knee_ext).clamp(-50.0, 50.0) / i_leg) * dt;

        // Use NEW velocities for angle updates
        let next_l_hip_angle = state.left_hip_angle + next_l_hip_v.clone() * dt;
        let next_r_hip_angle = state.right_hip_angle + next_r_hip_v.clone() * dt;
        let next_l_knee_angle = state.left_knee_angle + next_l_knee_v.clone() * dt;
        let next_r_knee_angle = state.right_knee_angle + next_r_knee_v.clone() * dt;

        PhysicsState {
            hull_x: next_hull_x,
            hull_y: next_hull_y,
            hull_angle: next_hull_angle,
            hull_vx: next_hull_vx,
            hull_vy: next_hull_vy,
            hull_v_angle: next_hull_v_angle,
            left_hip_angle: next_l_hip_angle,
            left_hip_v: next_l_hip_v,
            left_knee_angle: next_l_knee_angle,
            left_knee_v: next_l_knee_v,
            right_hip_angle: next_r_hip_angle,
            right_hip_v: next_r_hip_v,
            right_knee_angle: next_r_knee_angle,
            right_knee_v: next_r_knee_v,
            left_contact: l_contact.float(),
            right_contact: r_contact.float(),
            time: state.time.add_scalar(1),
            target_velocity: state.target_velocity,
        }
    }

    pub fn get_observation<B: Backend>(&self, state: &PhysicsState<B>) -> Tensor<B, 2> {
        Tensor::cat(
            vec![
                (state.hull_y.clone() - 0.8).unsqueeze_dim(1),
                state.hull_angle.clone().unsqueeze_dim(1),
                (state.hull_vx.clone() * 0.1).unsqueeze_dim(1),
                (state.hull_vy.clone() * 0.1).unsqueeze_dim(1),
                (state.hull_v_angle.clone() * 0.1).unsqueeze_dim(1),
                state.left_hip_angle.clone().unsqueeze_dim(1),
                (state.left_hip_v.clone() * 0.1).unsqueeze_dim(1),
                state.left_knee_angle.clone().unsqueeze_dim(1),
                (state.left_knee_v.clone() * 0.1).unsqueeze_dim(1),
                state.right_hip_angle.clone().unsqueeze_dim(1),
                (state.right_hip_v.clone() * 0.1).unsqueeze_dim(1),
                state.right_knee_angle.clone().unsqueeze_dim(1),
                (state.right_knee_v.clone() * 0.1).unsqueeze_dim(1),
                state.left_contact.clone().unsqueeze_dim(1),
                state.right_contact.clone().unsqueeze_dim(1),
                state.target_velocity.clone().unsqueeze_dim(1),
            ],
            1,
        )
    }

    pub fn initial_state<B: Backend>(
        &self,
        batch_size: usize,
        device: &B::Device,
    ) -> PhysicsState<B> {
        let random = |min: f32, max: f32| {
            Tensor::<B, 1>::random(
                [batch_size],
                Distribution::Uniform(min as f64, max as f64),
                device,
            )
        };

        PhysicsState {
            hull_x: Tensor::zeros([batch_size], device),
            hull_y: Tensor::ones([batch_size], device) * 1.05,
            hull_angle: random(-0.1, 0.1),
            hull_vx: random(-0.2, 0.2),
            hull_vy: random(-0.2, 0.2),
            hull_v_angle: random(-0.2, 0.2),
            left_hip_angle: random(-0.5, 0.5),
            left_hip_v: Tensor::zeros([batch_size], device),
            left_knee_angle: random(-0.6, 0.0),
            left_knee_v: Tensor::zeros([batch_size], device),
            right_hip_angle: random(-0.5, 0.5),
            right_hip_v: Tensor::zeros([batch_size], device),
            right_knee_angle: random(-0.6, 0.0),
            right_knee_v: Tensor::zeros([batch_size], device),
            left_contact: Tensor::zeros([batch_size], device),
            target_velocity: Tensor::zeros([batch_size], device),
            right_contact: Tensor::zeros([batch_size], device),
            time: Tensor::zeros([batch_size], device),
        }
    }
}
