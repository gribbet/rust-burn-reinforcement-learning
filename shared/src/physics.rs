use burn::prelude::*;
use burn::tensor::Int;
use rand::Rng;

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

impl Morphology {
    pub fn humanoid() -> Self {
        Self {
            root: Segment {
                length: 0.6, // Torso
                angle_min: -1e9,
                angle_max: 1e9,
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
}

pub struct WalkerConfig {
    pub gravity: f32,
    pub morphology: Morphology,
    pub time_step: f32,
    pub friction: f32,
    pub torque_magnitude: f32,
    pub mass_density: f32,
    pub fall_y: f32,
}

impl Default for WalkerConfig {
    fn default() -> Self {
        Self {
            gravity: 9.8,
            morphology: Morphology::humanoid(),
            time_step: 0.02,
            friction: 1.0,
            torque_magnitude: 20.0,
            mass_density: 5.0,
            fall_y: 0.75,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PhysicsState<B: Backend> {
    pub positions: Tensor<B, 3>,  // [batch, n_particles, 2]
    pub velocities: Tensor<B, 3>, // [batch, n_particles, 2]
    pub time: Tensor<B, 1>,
    pub target_velocity: Tensor<B, 1>,
}

pub struct Walker<B: Backend> {
    pub config: WalkerConfig,
    pub edges: Vec<(usize, usize, f32)>,         // p1, p2, length
    pub joint_pairs: Vec<(usize, usize, usize)>, // p_parent_start, p_joint, p_child_end

    pub edge_idx_1: Tensor<B, 1, Int>,
    pub edge_idx_2: Tensor<B, 1, Int>,
    pub edge_lengths: Tensor<B, 1>,
    pub edge_map_1: Tensor<B, 2>,
    pub edge_map_2: Tensor<B, 2>,

    pub joint_idx_p: Tensor<B, 1, Int>,
    pub joint_idx_j: Tensor<B, 1, Int>,
    pub joint_idx_c: Tensor<B, 1, Int>,
    pub joint_map_p: Tensor<B, 2>,
    pub joint_map_j: Tensor<B, 2>,
    pub joint_map_c: Tensor<B, 2>,

    pub joint_angle_min: Tensor<B, 1>,
    pub joint_angle_max: Tensor<B, 1>,

    pub masses: Tensor<B, 1>,
    pub inv_masses: Tensor<B, 1>,
    pub n_particles: usize,
    _marker: std::marker::PhantomData<B>,
}

impl<B: Backend> Walker<B> {
    pub fn new(config: WalkerConfig, device: &B::Device) -> Self {
        let mut edges = Vec::new();
        let mut joint_pairs = Vec::new();
        let mut joint_limits = Vec::new();

        // First pass: count particles and build structure
        fn count_segments(seg: &Segment) -> usize {
            1 + seg.children.iter().map(count_segments).sum::<usize>()
        }
        let n_segments = count_segments(&config.morphology.root);
        let n_particles = n_segments + 1;
        let mut masses_vec = vec![0.0; n_particles];

        // Stack for DFS: (segment, start_node_idx, parent_start_node_idx)
        let mut stack = vec![(&config.morphology.root, 0usize, None::<usize>)];
        let mut next_particle_id = 1;

        while let Some((seg, start_idx, parent_start_opt)) = stack.pop() {
            let end_idx = next_particle_id;
            next_particle_id += 1;

            edges.push((start_idx, end_idx, seg.length));

            let seg_mass = seg.length * config.mass_density;
            masses_vec[start_idx] += seg_mass * 0.5;
            masses_vec[end_idx] += seg_mass * 0.5;

            if let Some(parent_start) = parent_start_opt {
                // Joint at start_idx
                joint_pairs.push((parent_start, start_idx, end_idx));
                joint_limits.push((seg.angle_min, seg.angle_max));
            }

            for child in &seg.children {
                stack.push((child, end_idx, Some(start_idx)));
            }
        }

        let masses = Tensor::from_floats(masses_vec.as_slice(), device);
        let inv_masses_vec: Vec<f32> =
            masses_vec.iter().map(|&m| if m > 0.0 { 1.0 / m } else { 0.0 }).collect();
        let inv_masses = Tensor::from_floats(inv_masses_vec.as_slice(), device);

        let n_edges = edges.len();
        let n_joints = joint_pairs.len();

        let edge_idx_1_vec: Vec<i32> = edges.iter().map(|e| e.0 as i32).collect();
        let edge_idx_2_vec: Vec<i32> = edges.iter().map(|e| e.1 as i32).collect();
        let edge_lengths_vec: Vec<f32> = edges.iter().map(|e| e.2).collect();

        let edge_idx_1 = Tensor::from_ints(edge_idx_1_vec.as_slice(), device);
        let edge_idx_2 = Tensor::from_ints(edge_idx_2_vec.as_slice(), device);
        let edge_lengths = Tensor::from_floats(edge_lengths_vec.as_slice(), device);

        // Build edge maps
        let mut edge_map_1_data = vec![0.0; n_particles * n_edges];
        let mut edge_map_2_data = vec![0.0; n_particles * n_edges];
        for (i, &(p1, p2, _)) in edges.iter().enumerate() {
            edge_map_1_data[p1 * n_edges + i] = 1.0;
            edge_map_2_data[p2 * n_edges + i] = 1.0;
        }
        let edge_map_1 = Tensor::<B, 1>::from_floats(edge_map_1_data.as_slice(), device)
            .reshape([n_particles, n_edges]);
        let edge_map_2 = Tensor::<B, 1>::from_floats(edge_map_2_data.as_slice(), device)
            .reshape([n_particles, n_edges]);

        // Joints
        let joint_idx_p_vec: Vec<i32> = joint_pairs.iter().map(|j| j.0 as i32).collect();
        let joint_idx_j_vec: Vec<i32> = joint_pairs.iter().map(|j| j.1 as i32).collect();
        let joint_idx_c_vec: Vec<i32> = joint_pairs.iter().map(|j| j.2 as i32).collect();

        let joint_idx_p = Tensor::from_ints(joint_idx_p_vec.as_slice(), device);
        let joint_idx_j = Tensor::from_ints(joint_idx_j_vec.as_slice(), device);
        let joint_idx_c = Tensor::from_ints(joint_idx_c_vec.as_slice(), device);

        let mut joint_map_p_data = vec![0.0; n_particles * n_joints];
        let mut joint_map_j_data = vec![0.0; n_particles * n_joints];
        let mut joint_map_c_data = vec![0.0; n_particles * n_joints];

        for (i, &(p, j, c)) in joint_pairs.iter().enumerate() {
            joint_map_p_data[p * n_joints + i] = 1.0;
            joint_map_j_data[j * n_joints + i] = 1.0;
            joint_map_c_data[c * n_joints + i] = 1.0;
        }
        let joint_map_p = Tensor::<B, 1>::from_floats(joint_map_p_data.as_slice(), device)
            .reshape([n_particles, n_joints]);
        let joint_map_j = Tensor::<B, 1>::from_floats(joint_map_j_data.as_slice(), device)
            .reshape([n_particles, n_joints]);
        let joint_map_c = Tensor::<B, 1>::from_floats(joint_map_c_data.as_slice(), device)
            .reshape([n_particles, n_joints]);

        let joint_angle_min_vec: Vec<f32> = joint_limits.iter().map(|l| l.0).collect();
        let joint_angle_max_vec: Vec<f32> = joint_limits.iter().map(|l| l.1).collect();
        let joint_angle_min = Tensor::from_floats(joint_angle_min_vec.as_slice(), device);
        let joint_angle_max = Tensor::from_floats(joint_angle_max_vec.as_slice(), device);

        Self {
            config,
            edges,
            joint_pairs,
            edge_idx_1,
            edge_idx_2,
            edge_lengths,
            edge_map_1,
            edge_map_2,
            joint_idx_p,
            joint_idx_j,
            joint_idx_c,
            joint_map_p,
            joint_map_j,
            joint_map_c,
            joint_angle_min,
            joint_angle_max,
            masses,
            inv_masses,
            n_particles,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn action_dim(&self) -> usize {
        self.joint_pairs.len()
    }

    pub fn initial_state(&self, batch_size: usize, device: &B::Device) -> PhysicsState<B> {
        let mut rng = rand::thread_rng();
        let mut all_positions = Vec::with_capacity(batch_size * self.n_particles * 2);

        for _ in 0..batch_size {
            let mut positions = vec![[0.0; 2]; self.n_particles];
            positions[0] = [0.0, 1.5]; // Root start (Shoulders)

            // Stack: (segment, start_idx, start_x, start_y, angle)
            // Root segment is fixed at -PI/2 (Down) - Shoulders to Hips
            let mut stack = vec![(
                &self.config.morphology.root,
                0usize,
                0.0f32,
                1.5f32,
                -std::f32::consts::PI / 2.0,
            )];

            let mut next_particle_id = 1;

            while let Some((seg, _start_idx, x, y, angle)) = stack.pop() {
                let end_idx = next_particle_id;
                next_particle_id += 1;

                let end_x = x + angle.cos() * seg.length;
                let end_y = y + angle.sin() * seg.length;

                positions[end_idx] = [end_x, end_y];

                for child in &seg.children {
                    // Children continue relative to parent
                    let base_angle = angle;

                    let angle_offset = rng.gen_range(child.angle_min..=child.angle_max);
                    let child_angle = base_angle + angle_offset;

                    stack.push((child, end_idx, end_x, end_y, child_angle));
                }
            }

            for p in positions {
                all_positions.push(p[0]);
                all_positions.push(p[1]);
            }
        }

        let positions = Tensor::<B, 1>::from_floats(all_positions.as_slice(), device).reshape([
            batch_size,
            self.n_particles,
            2,
        ]);

        let velocities = Tensor::zeros([batch_size, self.n_particles, 2], device);
        let time = Tensor::zeros([batch_size], device);
        let target_velocity = Tensor::zeros([batch_size], device);

        PhysicsState { positions, velocities, time, target_velocity }
    }

    pub fn get_observation(&self, state: &PhysicsState<B>) -> Tensor<B, 2> {
        let batch_size = state.positions.dims()[0];

        // Observation:
        // Root y, vx, vy
        // For each joint: relative angle, angular velocity?
        // Or just positions and velocities of all particles relative to root?

        // Let's return relative positions and velocities to make it general.
        // Relative to root start (particle 0).

        let root_pos = state.positions.clone().slice([0..batch_size, 0..1, 0..2]); // [B, 1, 2]
        let rel_pos = state.positions.clone() - root_pos; // [B, N, 2]

        let flat_pos = rel_pos.reshape([batch_size, self.n_particles * 2]);
        let flat_vel = state.velocities.clone().reshape([batch_size, self.n_particles * 2]);

        Tensor::cat(vec![flat_pos, flat_vel], 1)
    }

    pub fn step(&self, state: PhysicsState<B>, action: Tensor<B, 2>) -> PhysicsState<B> {
        let batch_size = state.positions.dims()[0];

        let mut positions = state.positions;
        let mut velocities = state.velocities;

        let inv_masses_expanded = self
            .inv_masses
            .clone()
            .reshape([1, self.n_particles, 1])
            .expand([batch_size, self.n_particles, 1]);

        // 1. External Forces (Gravity)
        // F = m * g
        // a = g
        let mut acc = Tensor::<B, 3>::zeros([batch_size, self.n_particles, 2], &positions.device());
        let gravity_vec =
            Tensor::<B, 1>::from_floats([0.0, -self.config.gravity], &positions.device())
                .reshape([1, 1, 2])
                .expand([batch_size, self.n_particles, 2]);
        acc = acc + gravity_vec;

        // 2. Actuation Forces (Vectorized)
        let pos_p = positions.clone().select(1, self.joint_idx_p.clone()); // [B, n_joints, 2]
        let pos_j = positions.clone().select(1, self.joint_idx_j.clone());
        let pos_c = positions.clone().select(1, self.joint_idx_c.clone());

        let r_parent = pos_j.clone() - pos_p.clone();
        let r_child = pos_c.clone() - pos_j.clone();

        let r_parent_norm = r_parent.clone().powf_scalar(2.0).sum_dim(2).sqrt().clamp_min(1e-6);
        let r_child_norm = r_child.clone().powf_scalar(2.0).sum_dim(2).sqrt().clamp_min(1e-6);

        let u_parent = r_parent.clone() / r_parent_norm.clone();
        let u_child = r_child.clone() / r_child_norm.clone();

        let n_joints = self.joint_pairs.len();

        let perp_parent = Tensor::cat(
            vec![
                u_parent.clone().slice([0..batch_size, 0..n_joints, 1..2]).neg(),
                u_parent.clone().slice([0..batch_size, 0..n_joints, 0..1]),
            ],
            2,
        );

        let perp_child = Tensor::cat(
            vec![
                u_child.clone().slice([0..batch_size, 0..n_joints, 1..2]).neg(),
                u_child.clone().slice([0..batch_size, 0..n_joints, 0..1]),
            ],
            2,
        );

        let torque = action.clone().unsqueeze_dim::<3>(2); // [B, n_joints, 1]

        let f_parent_mag = torque.clone() / r_parent_norm * self.config.torque_magnitude;
        let f_child_mag = torque.clone() / r_child_norm * self.config.torque_magnitude;

        let f_child = perp_child * f_child_mag;
        let f_parent = perp_parent * f_parent_mag;
        let f_joint = (f_child.clone() + f_parent.clone()).neg();

        let map_p = self.joint_map_p.clone().unsqueeze::<3>().expand([
            batch_size,
            self.n_particles,
            n_joints,
        ]);
        let map_j = self.joint_map_j.clone().unsqueeze::<3>().expand([
            batch_size,
            self.n_particles,
            n_joints,
        ]);
        let map_c = self.joint_map_c.clone().unsqueeze::<3>().expand([
            batch_size,
            self.n_particles,
            n_joints,
        ]);

        let actuation_forces =
            map_p.matmul(f_parent) + map_j.matmul(f_joint) + map_c.matmul(f_child);

        acc = acc + actuation_forces * inv_masses_expanded.clone();

        // 3. Integration (Semi-implicit Euler)
        velocities = velocities + acc * self.config.time_step;
        let mut predicted = positions.clone() + velocities.clone() * self.config.time_step;

        // 4. Constraints (Distance + Angular)
        let n_edges = self.edges.len();
        let lengths = self.edge_lengths.clone().reshape([1, n_edges, 1]);
        let w1 =
            self.inv_masses.clone().select(0, self.edge_idx_1.clone()).reshape([1, n_edges, 1]);
        let w2 =
            self.inv_masses.clone().select(0, self.edge_idx_2.clone()).reshape([1, n_edges, 1]);
        let w_sum = (w1.clone() + w2.clone()).clamp_min(1e-6);

        let map_1 = self.edge_map_1.clone().unsqueeze::<3>().expand([
            batch_size,
            self.n_particles,
            n_edges,
        ]);
        let map_2 = self.edge_map_2.clone().unsqueeze::<3>().expand([
            batch_size,
            self.n_particles,
            n_edges,
        ]);

        let min_limit = self.joint_angle_min.clone().reshape([1, n_joints, 1]);
        let max_limit = self.joint_angle_max.clone().reshape([1, n_joints, 1]);

        let w_p =
            self.inv_masses.clone().select(0, self.joint_idx_p.clone()).reshape([1, n_joints, 1]);
        let w_j =
            self.inv_masses.clone().select(0, self.joint_idx_j.clone()).reshape([1, n_joints, 1]);
        let w_c =
            self.inv_masses.clone().select(0, self.joint_idx_c.clone()).reshape([1, n_joints, 1]);

        let map_p = self.joint_map_p.clone().unsqueeze::<3>().expand([
            batch_size,
            self.n_particles,
            n_joints,
        ]);
        let map_j = self.joint_map_j.clone().unsqueeze::<3>().expand([
            batch_size,
            self.n_particles,
            n_joints,
        ]);
        let map_c = self.joint_map_c.clone().unsqueeze::<3>().expand([
            batch_size,
            self.n_particles,
            n_joints,
        ]);

        let old_x = positions.clone().slice([0..batch_size, 0..self.n_particles, 0..1]);

        for _ in 0..2 {
            let x1 = predicted.clone().select(1, self.edge_idx_1.clone()); // [B, n_edges, 2]
            let x2 = predicted.clone().select(1, self.edge_idx_2.clone());

            let delta = x2.clone() - x1.clone();
            let dist = delta.clone().powf_scalar(2.0).sum_dim(2).sqrt().clamp_min(1e-6);

            let diff = (dist.clone() - lengths.clone()) / dist;

            let correction = delta * diff;
            let c1 = correction.clone() * (w1.clone() / w_sum.clone());
            let c2 = correction * (w2.clone() / w_sum.clone());

            let correction_total = map_1.clone().matmul(c1) - map_2.clone().matmul(c2);
            predicted = predicted + correction_total;

            // --- Angular Constraints (XPBD) ---
            // Re-fetch positions after distance constraints
            let pos_p = predicted.clone().select(1, self.joint_idx_p.clone());
            let pos_j = predicted.clone().select(1, self.joint_idx_j.clone());
            let pos_c = predicted.clone().select(1, self.joint_idx_c.clone());

            // Vectors: u = J - P, v = C - J
            let u = pos_j.clone() - pos_p.clone();
            let v = pos_c.clone() - pos_j.clone();

            let u_len_sq = u.clone().powf_scalar(2.0).sum_dim(2).clamp_min(1e-6);
            let v_len_sq = v.clone().powf_scalar(2.0).sum_dim(2).clamp_min(1e-6);

            // Normals (perpendiculars in 2D: -y, x)
            let u_x = u.clone().slice([0..batch_size, 0..n_joints, 0..1]);
            let u_y = u.clone().slice([0..batch_size, 0..n_joints, 1..2]);
            let u_perp = Tensor::cat(vec![u_y.clone().neg(), u_x.clone()], 2);

            let v_x = v.clone().slice([0..batch_size, 0..n_joints, 0..1]);
            let v_y = v.clone().slice([0..batch_size, 0..n_joints, 1..2]);
            let v_perp = Tensor::cat(vec![v_y.clone().neg(), v_x.clone()], 2);

            // Calculate current angle
            // Cross product (z): u_x * v_y - u_y * v_x
            let cross = u_x.clone() * v_y.clone() - u_y.clone() * v_x.clone();
            // Dot product: u . v
            let dot = (u.clone() * v.clone()).sum_dim(2);

            let angle = Self::approx_atan2(cross, dot);

            // Clamping
            let clamped_angle =
                angle.clone().max_pair(min_limit.clone()).min_pair(max_limit.clone());

            // C = angle - clamped_angle (We want C = 0)
            let c_val = angle.clone() - clamped_angle;

            // Gradients
            // grad_p = u_perp / u_len_sq
            // grad_c = v_perp / v_len_sq
            // grad_j = -(grad_p + grad_c)

            let grad_p = u_perp.clone() / u_len_sq.clone();
            let grad_c = v_perp.clone() / v_len_sq.clone();
            let grad_j = (grad_p.clone() + grad_c.clone()).neg();

            // Lambda denominator
            // sum(w * |grad|^2)
            let term_p = w_p.clone() * grad_p.clone().powf_scalar(2.0).sum_dim(2);
            let term_c = w_c.clone() * grad_c.clone().powf_scalar(2.0).sum_dim(2);
            let term_j = w_j.clone() * grad_j.clone().powf_scalar(2.0).sum_dim(2);

            // Compliance alpha (small value for stability)
            let alpha = 1e-6;
            let denom = term_p + term_c + term_j + alpha;

            // Lagrange multiplier
            let delta_lambda = c_val.neg() / denom;

            // Position corrections
            let dp = grad_p * delta_lambda.clone() * w_p.clone();
            let dc = grad_c * delta_lambda.clone() * w_c.clone();
            let dj = grad_j * delta_lambda.clone() * w_j.clone();

            let correction_total =
                map_p.clone().matmul(dp) + map_j.clone().matmul(dj) + map_c.clone().matmul(dc);

            predicted = predicted + correction_total;

            // --- Ground Collision ---
            // y < 0 -> y = 0
            let y = predicted.clone().slice([0..batch_size, 0..self.n_particles, 1..2]);
            let penetration = y.clone().neg().clamp_min(0.0);

            // Project out of ground
            let correction_y = penetration.clone();

            // PBD Friction
            // Apply tangential correction opposite to movement, limited by normal impulse * friction
            let current_x = predicted.clone().slice([0..batch_size, 0..self.n_particles, 0..1]);
            let diff_x = current_x - old_x.clone();

            // Normal impulse is proportional to correction_y
            let max_friction = correction_y.clone() * self.config.friction;

            // Clamp correction_x to [-max, max]
            // We want correction_x = -diff_x, clamped.
            let correction_x =
                diff_x.neg().max_pair(max_friction.clone().neg()).min_pair(max_friction);

            let correction_vec = Tensor::cat(vec![correction_x, correction_y.clone()], 2);

            predicted = predicted + correction_vec;
        }

        // Update velocity based on position change (PBD velocity update)
        // v = (p_new - p_old) / dt
        velocities = (predicted.clone() - positions) / self.config.time_step;
        positions = predicted;

        PhysicsState {
            positions,
            velocities,
            time: state.time + self.config.time_step,
            target_velocity: state.target_velocity,
        }
    }

    // Approximation of atan(z) for z in [-1, 1]
    // Polynomial: z * (0.99997726 - 0.33262347 * z^2 + 0.19354346 * z^4 - 0.11643287 * z^6 + 0.05265332 * z^8 - 0.01172120 * z^10)
    fn approx_atan_core(z: Tensor<B, 3>) -> Tensor<B, 3> {
        let z2 = z.clone().powf_scalar(2.0);
        let z4 = z2.clone().powf_scalar(2.0);
        let z6 = z4.clone() * z2.clone();
        let z8 = z4.clone().powf_scalar(2.0);
        let z10 = z8.clone() * z2.clone();

        z.clone()
            * (0.99997726 - 0.33262347 * z2 + 0.19354346 * z4 - 0.11643287 * z6 + 0.05265332 * z8
                - 0.01172120 * z10)
    }

    fn approx_atan(z: Tensor<B, 3>) -> Tensor<B, 3> {
        let mask = z.clone().abs().greater_elem(1.0);
        let z_safe = z.clone().mask_fill(mask.clone().bool_not(), 1.0);
        let z_inv = 1.0 / z_safe;
        let z_core = z.clone().mask_where(mask.clone(), z_inv);

        let res = Self::approx_atan_core(z_core);

        let pi_2 = std::f32::consts::PI / 2.0;
        let sign = z
            .clone()
            .mask_fill(z.clone().lower_elem(0.0), -1.0)
            .mask_fill(z.clone().greater_elem(0.0), 1.0);

        let res_inv = sign * pi_2 - res.clone();

        res.mask_where(mask, res_inv)
    }

    fn approx_atan2(y: Tensor<B, 3>, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x_zero = x.clone().equal_elem(0.0);
        let x_safe = x.clone().mask_fill(x_zero.clone(), 1.0);
        let z = y.clone() / x_safe;

        let atan_z = Self::approx_atan(z);

        let pi = std::f32::consts::PI;
        let pi_2 = pi / 2.0;

        let x_neg = x.clone().lower_elem(0.0);
        let y_neg = y.clone().lower_elem(0.0);

        let mut res = atan_z;

        // Add PI where x < 0
        let res_plus_pi = res.clone() + pi;
        res = res.mask_where(x_neg.clone(), res_plus_pi);

        // If y < 0 and we added PI (meaning x < 0), we are at > PI/2.
        // We want -PI relative to the original, or subtract 2PI from current.
        // If x < 0, y < 0 -> atan(z) > 0. res = atan(z) + PI > PI.
        // We want atan(z) - PI.
        // So subtract 2PI.

        let res_y_neg =
            res.clone().mask_where(res.clone().greater_elem(pi_2), res.clone() - 2.0 * pi);

        res = res.mask_where(y_neg, res_y_neg);

        // Handle x = 0
        let res_x_zero = Tensor::zeros_like(&res)
            .mask_fill(y.clone().greater_elem(0.0), pi_2)
            .mask_fill(y.clone().lower_elem(0.0), -pi_2);

        res = res.mask_where(x_zero, res_x_zero);

        res
    }
}
