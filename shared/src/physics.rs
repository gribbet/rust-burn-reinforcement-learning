use burn::prelude::*;
use burn::tensor::Int;
use std::f32::consts::PI;

#[derive(Clone, Debug)]
pub struct Segment {
    pub length: f32,
    pub mass: f32,
    pub angle_min: f32,
    pub angle_max: f32,
    pub max_torque: f32,
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
                length: 0.2, // Head
                mass: 5.0,
                angle_min: 0.0,
                angle_max: 0.0,
                max_torque: 0.0,
                children: vec![
                    // Torso
                    Segment {
                        length: 0.6,
                        mass: 35.0,
                        angle_min: -0.1, // Slight lean back
                        angle_max: 0.8,  // Lean forward into the walk
                        max_torque: 150.0,
                        children: vec![
                            // Left Leg
                            Segment {
                                length: 0.4,
                                mass: 10.0,
                                angle_min: -0.7, // Swing back
                                angle_max: 1.4,  // Swing forward (right)
                                max_torque: 120.0,
                                children: vec![Segment {
                                    length: 0.4,
                                    mass: 3.5,
                                    angle_min: -2.3, // Knee bend (backwards)
                                    angle_max: 0.0,  // Straight
                                    max_torque: 80.0,
                                    children: vec![],
                                }],
                            },
                            // Right Leg
                            Segment {
                                length: 0.4,
                                mass: 10.0,
                                angle_min: -0.7,
                                angle_max: 1.4,
                                max_torque: 120.0,
                                children: vec![Segment {
                                    length: 0.4,
                                    mass: 3.5,
                                    angle_min: -2.3,
                                    angle_max: 0.0,
                                    max_torque: 80.0,
                                    children: vec![],
                                }],
                            },
                        ],
                    },
                    // Left Arm
                    Segment {
                        length: 0.3,
                        mass: 2.5,
                        angle_min: -1.2,
                        angle_max: 1.2,
                        max_torque: 20.0,
                        children: vec![Segment {
                            length: 0.3,
                            mass: 1.5,
                            angle_min: 0.0,
                            angle_max: 2.2, // Elbow bend forward
                            max_torque: 10.0,
                            children: vec![],
                        }],
                    },
                    // Right Arm
                    Segment {
                        length: 0.3,
                        mass: 2.5,
                        angle_min: -1.2,
                        angle_max: 1.2,
                        max_torque: 20.0,
                        children: vec![Segment {
                            length: 0.3,
                            mass: 1.5,
                            angle_min: 0.0,
                            angle_max: 2.2,
                            max_torque: 10.0,
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
    pub fall_y: f32,
    pub constraint_iterations: usize,
}

impl Default for WalkerConfig {
    fn default() -> Self {
        Self {
            gravity: 9.8,
            morphology: Morphology::humanoid(),
            time_step: 1.0 / 60.0,
            friction: 1.0,
            fall_y: 1.3,
            constraint_iterations: 4,
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

    // Simplified mapping tensors
    pub edge_map_combined: Tensor<B, 3>,
    pub joint_map_combined: Tensor<B, 3>,

    pub joint_idx_p: Tensor<B, 1, Int>,
    pub joint_idx_j: Tensor<B, 1, Int>,
    pub joint_idx_c: Tensor<B, 1, Int>,

    pub joint_angle_min: Tensor<B, 1>,
    pub joint_angle_max: Tensor<B, 1>,

    pub joint_parent_lengths: Tensor<B, 3>,
    pub joint_child_lengths: Tensor<B, 3>,
    pub joint_max_torques: Tensor<B, 3>,

    pub masses: Tensor<B, 1>,
    pub inv_masses: Tensor<B, 1>,
    pub n_particles: usize,

    // Initialization data (stored on CPU to drive GPU-side initialization)
    pub init_p1: Vec<usize>,
    pub init_p2: Vec<usize>,
    pub init_lengths: Vec<f32>,
    pub init_angle_min: Vec<f32>,
    pub init_angle_max: Vec<f32>,
    pub init_parent_edge: Vec<Option<usize>>,

    pub init_eff_min: Tensor<B, 1>,
    pub init_eff_max: Tensor<B, 1>,
    pub init_lengths_tensor: Tensor<B, 1>,

    pub gravity_vec: Tensor<B, 3>,
    pub inv_masses_reshaped: Tensor<B, 3>,
    pub edge_lengths_reshaped: Tensor<B, 3>,
    pub edge_w1: Tensor<B, 3>,
    pub edge_w2: Tensor<B, 3>,
    pub edge_w_sum: Tensor<B, 3>,
    pub joint_min_limit: Tensor<B, 3>,
    pub joint_max_limit: Tensor<B, 3>,
    pub joint_w_p: Tensor<B, 3>,
    pub joint_w_j: Tensor<B, 3>,
    pub joint_w_c: Tensor<B, 3>,

    _marker: std::marker::PhantomData<B>,
}

impl<B: Backend> Walker<B> {
    pub fn center_of_mass(&self, positions: Tensor<B, 3>) -> Tensor<B, 2> {
        let masses = self.masses.clone().reshape([1, self.n_particles, 1]);
        let total_mass = masses.clone().sum_dim(1).squeeze_dim::<2>(1); // [1, 1] -> [1]

        let weighted_pos = positions * masses;
        let com = weighted_pos.sum_dim(1).squeeze_dim::<2>(1) / total_mass;
        com // [batch, 2]
    }

    pub fn new(config: WalkerConfig, device: &B::Device) -> Self {
        let mut edges = Vec::new();
        let mut joint_pairs = Vec::new();
        let mut joint_limits = Vec::new();
        let mut joint_parent_lengths_vec = Vec::new();
        let mut joint_child_lengths_vec = Vec::new();
        let mut joint_max_torques_vec = Vec::new();

        // First pass: count particles and build structure
        fn count_segments(seg: &Segment) -> usize {
            1 + seg.children.iter().map(count_segments).sum::<usize>()
        }
        let n_segments = count_segments(&config.morphology.root);
        let n_particles = n_segments + 1;
        let mut masses_vec = vec![0.0; n_particles];

        let mut init_p1 = Vec::new();
        let mut init_p2 = Vec::new();
        let mut init_lengths = Vec::new();
        let mut init_angle_min = Vec::new();
        let mut init_angle_max = Vec::new();
        let mut init_parent_edge = Vec::new();

        // Stack for DFS: (segment, start_node_idx, parent_start_node_idx, parent_edge_idx)
        let mut stack = vec![(&config.morphology.root, 0usize, None::<usize>)];
        let mut next_particle_id = 1;

        while let Some((seg, start_idx, parent_edge_opt)) = stack.pop() {
            let end_idx = next_particle_id;
            let current_edge_idx = init_p1.len();
            next_particle_id += 1;

            edges.push((start_idx, end_idx, seg.length));

            init_p1.push(start_idx);
            init_p2.push(end_idx);
            init_lengths.push(seg.length);
            init_angle_min.push(seg.angle_min);
            init_angle_max.push(seg.angle_max);
            init_parent_edge.push(parent_edge_opt);

            let seg_mass = seg.mass;
            masses_vec[start_idx] += seg_mass * 0.5;
            masses_vec[end_idx] += seg_mass * 0.5;

            if let Some(parent_edge_idx) = parent_edge_opt {
                let parent_start = init_p1[parent_edge_idx];
                // Joint at start_idx
                joint_pairs.push((parent_start, start_idx, end_idx));
                joint_limits.push((seg.angle_min, seg.angle_max));
                joint_parent_lengths_vec.push(init_lengths[parent_edge_idx]);
                joint_child_lengths_vec.push(seg.length);
                joint_max_torques_vec.push(seg.max_torque);
            }

            for child in &seg.children {
                stack.push((child, end_idx, Some(current_edge_idx)));
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

        let joint_parent_lengths =
            Tensor::<B, 1>::from_floats(joint_parent_lengths_vec.as_slice(), device)
                .reshape([1, n_joints, 1]);
        let joint_child_lengths =
            Tensor::<B, 1>::from_floats(joint_child_lengths_vec.as_slice(), device)
                .reshape([1, n_joints, 1]);
        let joint_max_torques =
            Tensor::<B, 1>::from_floats(joint_max_torques_vec.as_slice(), device)
                .reshape([1, n_joints, 1]);

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

        let joint_map_combined =
            Tensor::cat(vec![joint_map_p, joint_map_j, joint_map_c], 1).unsqueeze::<3>();
        let edge_map_combined = Tensor::cat(vec![edge_map_1, edge_map_2.neg()], 1).unsqueeze::<3>();

        let joint_angle_min_vec: Vec<f32> = joint_limits.iter().map(|l| l.0).collect();
        let joint_angle_max_vec: Vec<f32> = joint_limits.iter().map(|l| l.1).collect();
        let joint_angle_min = Tensor::from_floats(joint_angle_min_vec.as_slice(), device);
        let joint_angle_max = Tensor::from_floats(joint_angle_max_vec.as_slice(), device);

        let mut init_eff_min_vec = Vec::new();
        let mut init_eff_max_vec = Vec::new();
        for i in 0..init_p1.len() {
            let angle_min = init_angle_min[i];
            let angle_max = init_angle_max[i];
            let parent_edge_idx = init_parent_edge[i];

            let (eff_min, eff_max) =
                if parent_edge_idx.is_none() { (-0.8, 0.0) } else { (angle_min, angle_max) };
            init_eff_min_vec.push(eff_min);
            init_eff_max_vec.push(eff_max);
        }
        let init_eff_min = Tensor::from_floats(init_eff_min_vec.as_slice(), device);
        let init_eff_max = Tensor::from_floats(init_eff_max_vec.as_slice(), device);
        let init_lengths_tensor = Tensor::from_floats(init_lengths.as_slice(), device);

        let gravity_vec =
            Tensor::<B, 1>::from_floats([0.0, -config.gravity], device).reshape([1, 1, 2]);
        let inv_masses_reshaped = inv_masses.clone().reshape([1, n_particles, 1]);
        let edge_lengths_reshaped = edge_lengths.clone().reshape([1, n_edges, 1]);
        let edge_w1 = inv_masses.clone().select(0, edge_idx_1.clone()).reshape([1, n_edges, 1]);
        let edge_w2 = inv_masses.clone().select(0, edge_idx_2.clone()).reshape([1, n_edges, 1]);
        let edge_w_sum = (edge_w1.clone() + edge_w2.clone()).clamp_min(1e-6);

        let joint_min_limit = joint_angle_min.clone().reshape([1, n_joints, 1]);
        let joint_max_limit = joint_angle_max.clone().reshape([1, n_joints, 1]);
        let joint_w_p = inv_masses.clone().select(0, joint_idx_p.clone()).reshape([1, n_joints, 1]);
        let joint_w_j = inv_masses.clone().select(0, joint_idx_j.clone()).reshape([1, n_joints, 1]);
        let joint_w_c = inv_masses.clone().select(0, joint_idx_c.clone()).reshape([1, n_joints, 1]);

        Self {
            config,
            edges,
            joint_pairs,
            edge_idx_1,
            edge_idx_2,
            edge_lengths,
            edge_map_combined,
            joint_idx_p,
            joint_idx_j,
            joint_idx_c,
            joint_map_combined,
            joint_angle_min,
            joint_angle_max,
            joint_parent_lengths,
            joint_child_lengths,
            joint_max_torques,
            masses,
            inv_masses,
            n_particles,
            init_p1,
            init_p2,
            init_lengths,
            init_angle_min,
            init_angle_max,
            init_parent_edge,
            init_eff_min,
            init_eff_max,
            init_lengths_tensor,
            gravity_vec,
            inv_masses_reshaped,
            edge_lengths_reshaped,
            edge_w1,
            edge_w2,
            edge_w_sum,
            joint_min_limit,
            joint_max_limit,
            joint_w_p,
            joint_w_j,
            joint_w_c,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn action_dim(&self) -> usize {
        self.joint_pairs.len()
    }

    pub fn initial_state(&self, batch_size: usize, device: &B::Device) -> PhysicsState<B> {
        let mut positions = Tensor::zeros([batch_size, self.n_particles, 2], device);

        // Root start (Top of Head) at [0, 1.6]
        let root_start = Tensor::<B, 1>::from_floats([0.0, 1.6], device)
            .reshape([1, 1, 2])
            .expand([batch_size, 1, 2]);
        positions = positions.slice_assign([0..batch_size, 0..1, 0..2], root_start);

        let n_edges = self.init_p1.len();
        let random_vals =
            Tensor::random([n_edges, batch_size], burn::tensor::Distribution::Default, device);
        let eff_min = self.init_eff_min.clone().reshape([n_edges, 1]);
        let eff_max = self.init_eff_max.clone().reshape([n_edges, 1]);
        let all_angle_offsets = random_vals * (eff_max - eff_min.clone()) + eff_min;

        let root_base_angle = Tensor::<B, 1>::from_floats([-PI / 2.0], device).expand([batch_size]);

        let mut edge_angles: Vec<Tensor<B, 1>> = Vec::with_capacity(n_edges);

        for i in 0..n_edges {
            let p1 = self.init_p1[i];
            let p2 = self.init_p2[i];
            let length = self.init_lengths_tensor.clone().slice([i..i + 1]).reshape([1, 1]);
            let parent_edge_idx = self.init_parent_edge[i];

            let base_angle = if let Some(idx) = parent_edge_idx {
                edge_angles[idx].clone()
            } else {
                root_base_angle.clone()
            };

            let angle_offset = all_angle_offsets.clone().slice([i..i + 1]).squeeze_dim(0);
            let angle = base_angle + angle_offset;
            edge_angles.push(angle.clone());

            let cos_angle = angle.clone().cos().reshape([batch_size, 1]);
            let sin_angle = angle.sin().reshape([batch_size, 1]);
            let delta = Tensor::cat(vec![cos_angle * length.clone(), sin_angle * length], 1)
                .reshape([batch_size, 1, 2]);

            let p1_pos = positions.clone().slice([0..batch_size, p1..p1 + 1, 0..2]);
            let p2_pos = p1_pos + delta;
            positions = positions.slice_assign([0..batch_size, p2..p2 + 1, 0..2], p2_pos);
        }

        let velocities = Tensor::zeros([batch_size, self.n_particles, 2], device);
        let time = Tensor::zeros([batch_size], device);
        let target_velocity = Tensor::zeros([batch_size], device);

        PhysicsState { positions, velocities, time, target_velocity }
    }

    pub fn get_observation(&self, state: &PhysicsState<B>) -> Tensor<B, 2> {
        let batch_size = state.positions.dims()[0];

        let com = self.center_of_mass(state.positions.clone());
        let com_x = com.slice([0..batch_size, 0..1]).unsqueeze_dim::<3>(1);
        let offset = Tensor::cat(vec![com_x.clone(), Tensor::zeros_like(&com_x)], 2);

        let rel_pos = state.positions.clone() - offset;

        let flat_pos = rel_pos.reshape([batch_size, self.n_particles * 2]);
        let flat_vel = state.velocities.clone().reshape([batch_size, self.n_particles * 2]);

        let obs = Tensor::cat(vec![flat_pos, flat_vel], 1);
        // Sanitize observation: replace NaN with 0.0
        let is_finite = obs.clone().equal(obs.clone());
        obs.clone().mask_where(is_finite.bool_not(), Tensor::zeros_like(&obs))
    }

    pub fn step(&self, state: PhysicsState<B>, action: Tensor<B, 2>) -> PhysicsState<B> {
        let batch_size = state.positions.dims()[0];

        let mut positions = state.positions;
        let mut velocities = state.velocities;

        // 1. External Forces (Gravity)
        // F = m * g
        // a = g
        let mut acc = self.gravity_vec.clone().expand([batch_size, self.n_particles, 2]);

        // 2. Actuation Forces (Vectorized)
        let pos_p = positions.clone().select(1, self.joint_idx_p.clone()); // [B, n_joints, 2]
        let pos_j = positions.clone().select(1, self.joint_idx_j.clone());
        let pos_c = positions.clone().select(1, self.joint_idx_c.clone());

        let r_parent = pos_j.clone() - pos_p;
        let r_child = pos_c - pos_j.clone();

        let r_parent_norm = (r_parent.clone() * r_parent.clone()).sum_dim(2).sqrt().clamp_min(1e-6);
        let r_child_norm = (r_child.clone() * r_child.clone()).sum_dim(2).sqrt().clamp_min(1e-6);

        let u_parent = r_parent / r_parent_norm;
        let u_child = r_child / r_child_norm;

        let n_joints = self.joint_pairs.len();

        let perp_parent = Tensor::cat(
            vec![
                u_parent.clone().slice([0..batch_size, 0..n_joints, 1..2]).neg(),
                u_parent.slice([0..batch_size, 0..n_joints, 0..1]),
            ],
            2,
        );

        let perp_child = Tensor::cat(
            vec![
                u_child.clone().slice([0..batch_size, 0..n_joints, 1..2]).neg(),
                u_child.slice([0..batch_size, 0..n_joints, 0..1]),
            ],
            2,
        );

        let torque = action.unsqueeze_dim::<3>(2); // [B, n_joints, 1]

        let f_parent_mag =
            torque.clone() / self.joint_parent_lengths.clone() * self.joint_max_torques.clone();
        let f_child_mag =
            torque / self.joint_child_lengths.clone() * self.joint_max_torques.clone();

        let f_child = perp_child * f_child_mag;
        let f_parent = perp_parent * f_parent_mag;
        let f_joint = (f_child.clone() + f_parent.clone()).neg();

        let combined_forces = Tensor::cat(vec![f_parent, f_joint, f_child], 1);
        let actuation_forces = self.joint_map_combined.clone().matmul(combined_forces);

        acc = acc + actuation_forces * self.inv_masses_reshaped.clone();

        // 3. Integration (Semi-implicit Euler)
        velocities = velocities + acc * self.config.time_step;
        let mut predicted = positions.clone() + velocities.clone() * self.config.time_step;

        // 4. Constraints (Distance + Angular)
        let old_x = positions.clone().slice([0..batch_size, 0..self.n_particles, 0..1]);

        for _ in 0..self.config.constraint_iterations {
            let x1 = predicted.clone().select(1, self.edge_idx_1.clone()); // [B, n_edges, 2]
            let x2 = predicted.clone().select(1, self.edge_idx_2.clone());

            let delta = x2 - x1;
            let dist = (delta.clone() * delta.clone()).sum_dim(2).sqrt().clamp_min(1e-6);

            let diff = (dist.clone() - self.edge_lengths_reshaped.clone()) / dist;

            let correction = delta * diff;
            let c1 = correction.clone() * (self.edge_w1.clone() / self.edge_w_sum.clone());
            let c2 = correction * (self.edge_w2.clone() / self.edge_w_sum.clone());

            let combined_corrections = Tensor::cat(vec![c1, c2], 1);
            let correction_total = self.edge_map_combined.clone().matmul(combined_corrections);
            predicted = predicted + correction_total;

            // --- Angular Constraints (XPBD) ---
            // Re-fetch positions after distance constraints
            let pos_p = predicted.clone().select(1, self.joint_idx_p.clone());
            let pos_j = predicted.clone().select(1, self.joint_idx_j.clone());
            let pos_c = predicted.clone().select(1, self.joint_idx_c.clone());

            // Vectors: u = J - P, v = C - J
            let u = pos_j.clone() - pos_p.clone();
            let v = pos_c.clone() - pos_j.clone();

            let u_len_sq = (u.clone() * u.clone()).sum_dim(2).clamp_min(1e-6);
            let v_len_sq = (v.clone() * v.clone()).sum_dim(2).clamp_min(1e-6);

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
            let dot = (u * v).sum_dim(2);

            let angle = crate::math::approx_atan2(cross, dot);

            // Clamping
            let clamped_angle = angle
                .clone()
                .max_pair(self.joint_min_limit.clone())
                .min_pair(self.joint_max_limit.clone());

            // C = angle - clamped_angle (We want C = 0)
            let c_val = angle - clamped_angle;

            // Gradients
            // grad_p = u_perp / u_len_sq
            // grad_c = v_perp / v_len_sq
            // grad_j = -(grad_p + grad_c)

            let grad_p = u_perp / u_len_sq;
            let grad_c = v_perp / v_len_sq;
            let grad_j = (grad_p.clone() + grad_c.clone()).neg();

            // Lambda denominator
            // sum(w * |grad|^2)
            let term_p = self.joint_w_p.clone() * (grad_p.clone() * grad_p.clone()).sum_dim(2);
            let term_c = self.joint_w_c.clone() * (grad_c.clone() * grad_c.clone()).sum_dim(2);
            let term_j = self.joint_w_j.clone() * (grad_j.clone() * grad_j.clone()).sum_dim(2);

            // Compliance alpha (small value for stability)
            let alpha = 1e-4;
            let denom = term_p + term_c + term_j + alpha;

            // Lagrange multiplier
            let delta_lambda = c_val.neg() / denom;

            // Position corrections
            let dp = grad_p * delta_lambda.clone() * self.joint_w_p.clone();
            let dc = grad_c * delta_lambda.clone() * self.joint_w_c.clone();
            let dj = grad_j * delta_lambda.clone() * self.joint_w_j.clone();

            let combined_corrections = Tensor::cat(vec![dp, dj, dc], 1);
            let correction_total = self.joint_map_combined.clone().matmul(combined_corrections);

            predicted = predicted + correction_total;

            // --- Ground Collision & Friction ---
            let y = predicted.clone().slice([0..batch_size, 0..self.n_particles, 1..2]);
            let is_grounded = y.clone().lower_equal_elem(0.01);

            // Project out of ground
            let correction_y = y.clone().neg().clamp_min(0.0);

            // PBD Friction
            // Apply tangential correction opposite to movement when grounded
            let current_x = predicted.clone().slice([0..batch_size, 0..self.n_particles, 0..1]);
            let diff_x = current_x - old_x.clone();

            let friction_correction = diff_x.neg() * self.config.friction;
            let correction_x = Tensor::zeros_like(&friction_correction)
                .mask_where(is_grounded, friction_correction);

            let correction_vec = Tensor::cat(vec![correction_x, correction_y], 2);

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
}
