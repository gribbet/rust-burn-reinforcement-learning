use burn::prelude::*;
use burn::tensor::{Distribution, Int, TensorData};

#[derive(Clone, Debug)]
pub struct Segment {
    pub length: f32,
    pub angle_min: f32,
    pub angle_max: f32,
    pub children: Vec<Segment>,
}

#[derive(Clone, Debug)]
pub struct Morphology {
    pub root: Segment,
}

pub struct InternalSegment<'a> {
    pub segment: &'a Segment,
    pub parent_idx: Option<usize>,
}

impl Morphology {
    pub fn humanoid() -> Self {
        Self {
            root: Segment {
                length: 0.6, // Torso
                angle_min: -0.5,
                angle_max: 0.5,
                children: vec![
                    // Left Leg
                    Segment {
                        length: 0.4,
                        angle_min: -1.0,
                        angle_max: 1.0,
                        children: vec![Segment {
                            length: 0.4,
                            angle_min: -2.0,
                            angle_max: 0.0,
                            children: vec![],
                        }],
                    },
                    // Right Leg
                    Segment {
                        length: 0.4,
                        angle_min: -1.0,
                        angle_max: 1.0,
                        children: vec![Segment {
                            length: 0.4,
                            angle_min: -2.0,
                            angle_max: 0.0,
                            children: vec![],
                        }],
                    },
                ],
            },
        }
    }

    pub fn num_joints(&self) -> usize {
        self.flatten_segments().len()
    }

    pub fn flatten_segments(&self) -> Vec<&Segment> {
        let mut segments = Vec::new();
        self.flatten_recursive(&self.root, &mut segments);
        segments
    }

    fn flatten_recursive<'a>(&self, segment: &'a Segment, segments: &mut Vec<&'a Segment>) {
        segments.push(segment);
        for child in &segment.children {
            self.flatten_recursive(child, segments);
        }
    }
}

#[derive(Clone, Debug)]
pub struct PhysicsState<B: Backend> {
    pub x: Tensor<B, 1>,
    pub y: Tensor<B, 1>,
    pub angles: Tensor<B, 2>, // [batch, num_segments]
    pub vx: Tensor<B, 1>,
    pub vy: Tensor<B, 1>,
    pub v_angles: Tensor<B, 2>, // [batch, num_segments]
    pub time: Tensor<B, 1, Int>,
    pub target_velocity: Tensor<B, 1>,
}

pub struct Kinematics<B: Backend> {
    pub root_x: Tensor<B, 1>,
    pub root_y: Tensor<B, 1>,
    pub root_vx: Tensor<B, 1>,
    pub root_vy: Tensor<B, 1>,
    pub rel_com_x: Tensor<B, 1>,
    pub rel_com_y: Tensor<B, 1>,
    pub end_x: Tensor<B, 2>,
    pub end_y: Tensor<B, 2>,
    pub end_vx: Tensor<B, 2>,
    pub end_vy: Tensor<B, 2>,
}

pub struct WalkerPhysics {
    pub gravity: f32,
    pub morphology: Morphology,
    pub time_step: f32,
    pub friction: f32,
    pub torque_magnitude: f32,
    pub joint_damping: f32,
    pub sub_steps: usize,
    pub ground_stiffness: f32,
    pub joint_limit_stiffness: f32,
    pub friction_sharpness: f32,
    pub mass_density: f32,
}

impl Default for WalkerPhysics {
    fn default() -> Self {
        Self {
            gravity: 9.8,
            morphology: Morphology::humanoid(),
            time_step: 0.02,
            friction: 1.0,
            torque_magnitude: 40.0,
            joint_damping: 0.1,
            sub_steps: 4,
            ground_stiffness: 5000.0,
            joint_limit_stiffness: 1000.0,
            friction_sharpness: 10.0,
            mass_density: 5.0,
        }
    }
}

impl WalkerPhysics {
    fn get_flat_segments(&self) -> Vec<InternalSegment<'_>> {
        let mut flat_segments = Vec::new();
        fn flatten_with_parents<'a>(
            s: &'a Segment,
            p: Option<usize>,
            list: &mut Vec<InternalSegment<'a>>,
        ) {
            let idx = list.len();
            list.push(InternalSegment { segment: s, parent_idx: p });
            for child in &s.children {
                flatten_with_parents(child, Some(idx), list);
            }
        }
        flatten_with_parents(&self.morphology.root, None, &mut flat_segments);
        flat_segments
    }

    fn get_morphology_matrices<B: Backend>(
        &self,
        device: &B::Device,
    ) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>, Vec<Option<usize>>, Vec<f32>, Vec<f32>) {
        let flat_segments = self.get_flat_segments();
        let num_segments = flat_segments.len();
        let mut ancestor_data = vec![0.0; num_segments * num_segments];
        let mut lengths_data = vec![0.0; num_segments];
        let mut masses_data = vec![0.0; num_segments];
        let mut parent_indices = Vec::with_capacity(num_segments);

        for i in 0..num_segments {
            lengths_data[i] = flat_segments[i].segment.length;
            masses_data[i] = flat_segments[i].segment.length * self.mass_density;
            parent_indices.push(flat_segments[i].parent_idx);
            let mut curr = Some(i);
            while let Some(idx) = curr {
                ancestor_data[i * num_segments + idx] = 1.0;
                curr = flat_segments[idx].parent_idx;
            }
        }

        let ancestor_matrix = Tensor::<B, 2>::from_data(
            TensorData::new(ancestor_data, [num_segments, num_segments]),
            device,
        );
        let lengths = Tensor::<B, 2>::from_data(
            TensorData::new(lengths_data.clone(), [1, num_segments]),
            device,
        );
        let masses = Tensor::<B, 2>::from_data(
            TensorData::new(masses_data.clone(), [1, num_segments]),
            device,
        );

        (ancestor_matrix, lengths, masses, parent_indices, lengths_data, masses_data)
    }

    pub fn calculate_kinematics<B: Backend>(&self, state: &PhysicsState<B>) -> Kinematics<B> {
        let _batch_size = state.x.dims()[0];
        let device = &state.x.device();
        let (ancestor_matrix, lengths, masses, _, _, masses_data) =
            self.get_morphology_matrices::<B>(device);

        let abs_angles = state.angles.clone().matmul(ancestor_matrix.clone().transpose());
        let abs_v_angles = state.v_angles.clone().matmul(ancestor_matrix.clone().transpose());

        let rel_vec_x = abs_angles.clone().sin() * lengths.clone();
        let rel_vec_y = abs_angles.clone().cos() * lengths.clone() * -1.0;

        let rel_end_x = rel_vec_x.clone().matmul(ancestor_matrix.clone().transpose());
        let rel_end_y = rel_vec_y.clone().matmul(ancestor_matrix.clone().transpose());

        let rel_vec_vx = abs_v_angles.clone() * abs_angles.clone().cos() * lengths.clone();
        let rel_vec_vy = abs_v_angles.clone() * abs_angles.clone().sin() * lengths.clone();

        let rel_end_vx = rel_vec_vx.clone().matmul(ancestor_matrix.clone().transpose());
        let rel_end_vy = rel_vec_vy.clone().matmul(ancestor_matrix.clone().transpose());

        let rel_start_x = rel_end_x.clone() - rel_vec_x;
        let rel_start_y = rel_end_y.clone() - rel_vec_y;
        let rel_start_vx = rel_end_vx.clone() - rel_vec_vx;
        let rel_start_vy = rel_end_vy.clone() - rel_vec_vy;

        let rel_com_x_segs = (rel_start_x + rel_end_x.clone()) * 0.5;
        let rel_com_y_segs = (rel_start_y + rel_end_y.clone()) * 0.5;
        let rel_com_vx_segs = (rel_start_vx + rel_end_vx.clone()) * 0.5;
        let rel_com_vy_segs = (rel_start_vy + rel_end_vy.clone()) * 0.5;

        let total_mass: f32 = masses_data.iter().sum();
        let rel_com_x = (rel_com_x_segs * masses.clone()).sum_dim(1).squeeze_dim(1) / total_mass;
        let rel_com_y = (rel_com_y_segs * masses.clone()).sum_dim(1).squeeze_dim(1) / total_mass;
        let rel_com_vx = (rel_com_vx_segs * masses.clone()).sum_dim(1).squeeze_dim(1) / total_mass;
        let rel_com_vy = (rel_com_vy_segs * masses.clone()).sum_dim(1).squeeze_dim(1) / total_mass;

        let root_x = state.x.clone() - rel_com_x.clone();
        let root_y = state.y.clone() - rel_com_y.clone();
        let root_vx = state.vx.clone() - rel_com_vx;
        let root_vy = state.vy.clone() - rel_com_vy;

        let end_x = rel_end_x + root_x.clone().unsqueeze_dim(1);
        let end_y = rel_end_y + root_y.clone().unsqueeze_dim(1);
        let end_vx = rel_end_vx + root_vx.clone().unsqueeze_dim(1);
        let end_vy = rel_end_vy + root_vy.clone().unsqueeze_dim(1);

        Kinematics {
            root_x,
            root_y,
            root_vx,
            root_vy,
            rel_com_x,
            rel_com_y,
            end_x,
            end_y,
            end_vx,
            end_vy,
        }
    }

    pub fn step<B: Backend>(
        &self,
        state: PhysicsState<B>,
        action: Tensor<B, 2>, // [batch, num_joints]
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
        action: Tensor<B, 2>,
    ) -> PhysicsState<B> {
        let batch_size = action.dims()[0];
        let device = &action.device();
        let (ancestor_matrix, lengths, masses, parent_indices, lengths_data, masses_data) =
            self.get_morphology_matrices::<B>(device);
        let num_segments = parent_indices.len();
        let g = self.gravity;

        let kin = self.calculate_kinematics(&state);

        // --- 4. Forces and Torques (Vectorized) ---
        let k_g = self.ground_stiffness;

        // Combine root and end points for contact resolution
        let combined_y =
            Tensor::cat(vec![kin.root_y.clone().unsqueeze_dim(1), kin.end_y.clone()], 1);
        let combined_vx =
            Tensor::cat(vec![kin.root_vx.clone().unsqueeze_dim(1), kin.end_vx.clone()], 1);
        let combined_vy =
            Tensor::cat(vec![kin.root_vy.clone().unsqueeze_dim(1), kin.end_vy.clone()], 1);

        let root_mass = masses_data[0] * 0.5;
        let combined_masses = Tensor::cat(
            vec![Tensor::<B, 1>::from_floats([root_mass], device).unsqueeze_dim(0), masses.clone()],
            1,
        );

        let combined_d_gs = (combined_masses.clone() * k_g).sqrt() * 2.0;

        let penetration = combined_y.clone().mul_scalar(-1.0).clamp_min(0.0);
        let is_contact = combined_y.clone().lower_equal_elem(0.0);

        // Normal force for integration (spring only, damping is handled implicitly in step 8 for stability)
        let combined_fy = Tensor::zeros_like(&combined_y)
            .mask_where(is_contact.clone(), penetration.clone() * k_g);

        // Effective normal force for friction calculation (includes damping for physical realism)
        let fy_damping = (combined_vy.clone() * -1.0) * combined_d_gs.clone();
        let fy_eff = (penetration * k_g + fy_damping).clamp_min(0.0);
        let combined_fy_eff =
            Tensor::zeros_like(&combined_y).mask_where(is_contact.clone(), fy_eff);

        // Friction force: Stribeck model (Static friction > Kinetic friction)
        // mu(v) = mu_k + (mu_s - mu_k) * exp(-(v/v_s)^2)
        let mu_k = self.friction;
        let mu_s = mu_k * 1.5; // Static friction is typically higher than kinetic
        let v_s = 0.1; // Transition velocity (m/s)
        let v_rel = combined_vx.clone().abs();
        let mu_v = mu_k + (mu_s - mu_k) * (v_rel.powf_scalar(2.0).neg() / (v_s * v_s)).exp();

        let combined_fx =
            (combined_vx.clone() * self.friction_sharpness).tanh() * combined_fy_eff * mu_v * -1.0;

        // Split back for segment-specific calculations
        let root_fy = combined_fy.clone().slice([0..batch_size, 0..1]).squeeze_dim(1);
        let root_fx = combined_fx.clone().slice([0..batch_size, 0..1]).squeeze_dim(1);
        let fy_val = combined_fy.slice([0..batch_size, 1..num_segments + 1]);
        let fx_val = combined_fx.slice([0..batch_size, 1..num_segments + 1]);

        let total_mass: f32 = masses_data.iter().sum();
        let hull_fx_total = fx_val.clone().sum_dim(1).squeeze_dim(1) + root_fx.clone();
        let hull_fy_total =
            fy_val.clone().sum_dim(1).squeeze_dim(1) + root_fy.clone() - total_mass * g;

        let total_d_g = (is_contact.float() * combined_d_gs).sum_dim(1).squeeze_dim(1);

        // --- 5. Subtree Inertia and Mass ---
        let mut subtree_mass = vec![0.0; num_segments];
        let mut subtree_inertia = vec![0.0; num_segments];

        for i in (0..num_segments).rev() {
            let m = masses_data[i];
            let l = lengths_data[i];
            let i_self = (m * l.powi(2)) / 3.0;

            subtree_mass[i] += m;
            subtree_inertia[i] += i_self;

            if let Some(p_idx) = parent_indices[i] {
                let p_l = lengths_data[p_idx];
                subtree_mass[p_idx] += subtree_mass[i];
                subtree_inertia[p_idx] += subtree_inertia[i] + subtree_mass[i] * p_l.powi(2);
            }
        }
        let subtree_inertia_tensor: Tensor<B, 1> =
            Tensor::from_floats(subtree_inertia.as_slice(), device);

        // Total body inertia about COM (Vectorized)
        let i_self_com = (masses.clone() * lengths.clone().powf_scalar(2.0)) / 12.0; // [1, num_segments]

        let mut pkx_vec = Vec::with_capacity(num_segments);
        let mut pky_vec = Vec::with_capacity(num_segments);
        for i in 0..num_segments {
            match parent_indices[i] {
                Some(p_idx) => {
                    pkx_vec.push(kin.end_x.clone().slice([0..batch_size, p_idx..p_idx + 1]));
                    pky_vec.push(kin.end_y.clone().slice([0..batch_size, p_idx..p_idx + 1]));
                }
                None => {
                    pkx_vec.push(kin.root_x.clone().unsqueeze_dim::<2>(1));
                    pky_vec.push(kin.root_y.clone().unsqueeze_dim::<2>(1));
                }
            }
        }
        let pkx = Tensor::cat(pkx_vec, 1);
        let pky = Tensor::cat(pky_vec, 1);

        let scx = (pkx.clone() + kin.end_x.clone()) * 0.5;
        let scy = (pky.clone() + kin.end_y.clone()) * 0.5;
        let dx = scx - state.x.clone().unsqueeze_dim::<2>(1);
        let dy = scy - state.y.clone().unsqueeze_dim::<2>(1);
        let i_com: Tensor<B, 1> = (i_self_com
            + (dx.powf_scalar(2.0) + dy.powf_scalar(2.0)) * masses.clone())
        .sum_dim(1)
        .squeeze_dim(1)
        .clamp_min(0.01);

        // --- 6. Recursive Torque Calculation ---
        let f_ext_x = fx_val.clone();
        let f_ext_y = fy_val.clone() - masses.clone() * g;

        let t_contact =
            (kin.end_x.clone() - pkx.clone()) * fy_val - (kin.end_y.clone() - pky.clone()) * fx_val;
        let t_grav =
            ((pkx.clone() + kin.end_x.clone()) * 0.5 - pkx.clone()) * (masses.clone() * -g);

        let mut subtree_t_ext = t_contact + t_grav;

        let abs_angles = state.angles.clone().matmul(ancestor_matrix.clone().transpose());
        let subtree_fx = f_ext_x.matmul(ancestor_matrix.clone());
        let subtree_fy = f_ext_y.matmul(ancestor_matrix.clone());

        for i in (1..num_segments).rev() {
            let p_idx = parent_indices[i].unwrap();
            let l_p = lengths_data[p_idx];
            let vec_p_x = abs_angles.clone().slice([0..batch_size, p_idx..p_idx + 1]).sin() * l_p;
            let vec_p_y =
                abs_angles.clone().slice([0..batch_size, p_idx..p_idx + 1]).cos() * (l_p * -1.0);

            let t_child = subtree_t_ext.clone().slice([0..batch_size, i..i + 1]);
            let f_child_x = subtree_fx.clone().slice([0..batch_size, i..i + 1]);
            let f_child_y = subtree_fy.clone().slice([0..batch_size, i..i + 1]);

            let cross = vec_p_x * f_child_y - vec_p_y * f_child_x;

            let current_p_t = subtree_t_ext.clone().slice([0..batch_size, p_idx..p_idx + 1]);
            subtree_t_ext = subtree_t_ext
                .slice_assign([0..batch_size, p_idx..p_idx + 1], current_p_t + t_child + cross);
        }

        // Shift root torque from head-relative to COM-relative: tau_com = tau_head + (head - com) x F_total
        let dx_head = kin.root_x.clone() - state.x.clone();
        let dy_head = kin.root_y.clone() - state.y.clone();
        let t_shift = dx_head * hull_fy_total.clone() - dy_head * hull_fx_total.clone();

        let current_root_t = subtree_t_ext.clone().slice([0..batch_size, 0..1]).squeeze_dim(1);
        subtree_t_ext = subtree_t_ext
            .slice_assign([0..batch_size, 0..1], (current_root_t + t_shift).unsqueeze_dim(1));

        // --- 7. Internal Torques (Joint Limits, Actions) ---
        let k_limit_base = self.joint_limit_stiffness;

        let i_sub =
            subtree_inertia_tensor.unsqueeze_dim::<2>(0).repeat(&[batch_size, 1]).clamp_min(0.01);
        let i_sub = i_sub.slice_assign([0..batch_size, 0..1], i_com.unsqueeze_dim::<2>(1));

        let act = action * self.torque_magnitude * i_sub.clone();
        let act = act.slice_assign([0..batch_size, 0..1], Tensor::zeros([batch_size, 1], device));

        let k_limit = i_sub.clone() * k_limit_base;
        let d_limit = (k_limit.clone() * i_sub.clone()).sqrt() * 2.0;

        let mins: Tensor<B, 2> = Tensor::<B, 1>::from_floats(
            self.morphology
                .flatten_segments()
                .iter()
                .map(|s| s.angle_min)
                .collect::<Vec<_>>()
                .as_slice(),
            device,
        )
        .unsqueeze_dim::<2>(0);
        let maxs: Tensor<B, 2> = Tensor::<B, 1>::from_floats(
            self.morphology
                .flatten_segments()
                .iter()
                .map(|s| s.angle_max)
                .collect::<Vec<_>>()
                .as_slice(),
            device,
        )
        .unsqueeze_dim::<2>(0);

        let diff_min = state.angles.clone().neg().add_scalar(0.0) + mins;
        let t_min = (diff_min.clone().clamp_min(0.0) * k_limit.clone()
            - state.v_angles.clone() * d_limit.clone())
        .mask_where(diff_min.lower_elem(0.0), Tensor::zeros_like(&state.angles));

        let diff_max = state.angles.clone() - maxs;
        let t_max =
            (diff_max.clone().clamp_min(0.0) * k_limit + state.v_angles.clone() * d_limit) * -1.0;
        let t_max = t_max.mask_where(diff_max.lower_elem(0.0), Tensor::zeros_like(&state.angles));

        let joint_t_int = act + t_min + t_max;

        // --- 8. Integration ---
        let dt = self.time_step / self.sub_steps as f32;
        let next_com_vx = (state.vx + (hull_fx_total / total_mass) * dt)
            / (Tensor::ones_like(&state.y) + (total_d_g.clone() / total_mass) * dt);
        let next_com_vy = (state.vy + (hull_fy_total / total_mass) * dt)
            / (Tensor::ones_like(&state.y) + (total_d_g / total_mass) * dt);
        let next_com_x = state.x + next_com_vx.clone() * dt;
        let next_com_y = state.y + next_com_vy.clone() * dt;

        let d_joint = i_sub.clone() * self.joint_damping * 10.0;

        let next_segment_vs = (state.v_angles.clone()
            + ((joint_t_int + subtree_t_ext) / i_sub.clone()) * dt)
            / (Tensor::ones_like(&i_sub) + (d_joint / i_sub) * dt);
        let next_segment_angles = state.angles.clone() + next_segment_vs.clone() * dt;

        PhysicsState {
            x: next_com_x,
            y: next_com_y,
            angles: next_segment_angles,
            vx: next_com_vx,
            vy: next_com_vy,
            v_angles: next_segment_vs,
            time: state.time.add_scalar(1),
            target_velocity: state.target_velocity,
        }
    }

    pub fn get_observation<B: Backend>(&self, state: &PhysicsState<B>) -> Tensor<B, 2> {
        let kin = self.calculate_kinematics(state);
        let segment_angles = state.angles.clone();
        let segment_vs = state.v_angles.clone() * 0.1;

        // Contacts (Head + All segment ends)
        let contacts = Tensor::cat(
            vec![
                kin.root_y.clone().lower_equal_elem(0.0).float().unsqueeze_dim(1),
                kin.end_y.lower_equal_elem(0.0).float(),
            ],
            1,
        );

        let hull_y = state.y.clone().unsqueeze_dim(1);
        let hull_vx = state.vx.clone().unsqueeze_dim(1) * 0.1;
        let hull_vy = state.vy.clone().unsqueeze_dim(1) * 0.1;

        let mut obs = vec![hull_y - 0.8, segment_angles, hull_vx, hull_vy, segment_vs, contacts];
        obs.push(state.target_velocity.clone().unsqueeze_dim(1));
        Tensor::cat(obs, 1)
    }

    pub fn initial_state<B: Backend>(
        &self,
        batch_size: usize,
        device: &B::Device,
    ) -> PhysicsState<B> {
        let num_segments = self.morphology.num_joints();
        let segments = self.morphology.flatten_segments();

        let mut angles = Vec::with_capacity(num_segments);
        for s in segments {
            let angle = Tensor::<B, 1>::random(
                [batch_size],
                Distribution::Uniform((s.angle_min * 0.1) as f64, (s.angle_max * 0.1) as f64),
                device,
            );
            angles.push(angle.unsqueeze_dim(1));
        }

        PhysicsState {
            x: Tensor::zeros([batch_size], device),
            y: Tensor::ones([batch_size], device) * 1.0,
            angles: Tensor::cat(angles, 1),
            vx: Tensor::zeros([batch_size], device),
            vy: Tensor::zeros([batch_size], device),
            v_angles: Tensor::zeros([batch_size, num_segments], device),
            time: Tensor::<B, 1, Int>::zeros([batch_size], device),
            target_velocity: Tensor::zeros([batch_size], device),
        }
    }
}
