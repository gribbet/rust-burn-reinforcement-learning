use burn::prelude::*;
use burn::tensor::{Distribution, Int};

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
            friction: 0.5,
            torque_magnitude: 40.0,
            joint_damping: 0.1,
            sub_steps: 2,
            ground_stiffness: 5000.0,
            joint_limit_stiffness: 1000.0,
            friction_sharpness: 1.0,
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

    pub fn calculate_kinematics<B: Backend>(&self, state: &PhysicsState<B>) -> Kinematics<B> {
        let batch_size = state.x.dims()[0];
        let device = &state.x.device();
        let num_segments = self.morphology.num_joints();

        let flat_segments = self.get_flat_segments();

        let mut rel_end_x_vec: Vec<Tensor<B, 1>> = Vec::with_capacity(num_segments);
        let mut rel_end_y_vec: Vec<Tensor<B, 1>> = Vec::with_capacity(num_segments);
        let mut rel_end_vx_vec: Vec<Tensor<B, 1>> = Vec::with_capacity(num_segments);
        let mut rel_end_vy_vec: Vec<Tensor<B, 1>> = Vec::with_capacity(num_segments);
        let mut abs_angle_vec: Vec<Tensor<B, 1>> = Vec::with_capacity(num_segments);
        let mut abs_v_angle_vec: Vec<Tensor<B, 1>> = Vec::with_capacity(num_segments);

        let mut total_mass = 0.0;
        for (i, is) in flat_segments.iter().enumerate() {
            let j_angle = state.angles.clone().slice([0..batch_size, i..i + 1]).squeeze_dim(1);
            let j_v = state.v_angles.clone().slice([0..batch_size, i..i + 1]).squeeze_dim(1);

            let (px, py, pa, pvx, pvy, pva) = match is.parent_idx {
                Some(p_idx) => (
                    rel_end_x_vec[p_idx].clone(),
                    rel_end_y_vec[p_idx].clone(),
                    abs_angle_vec[p_idx].clone(),
                    rel_end_vx_vec[p_idx].clone(),
                    rel_end_vy_vec[p_idx].clone(),
                    abs_v_angle_vec[p_idx].clone(),
                ),
                None => (
                    Tensor::zeros([batch_size], device),
                    Tensor::zeros([batch_size], device),
                    Tensor::zeros([batch_size], device),
                    Tensor::zeros([batch_size], device),
                    Tensor::zeros([batch_size], device),
                    Tensor::zeros([batch_size], device),
                ),
            };

            let a = pa + j_angle;
            let ex = px + a.clone().sin() * is.segment.length;
            let ey = py - a.clone().cos() * is.segment.length;

            let va = pva + j_v;
            let evx = pvx + va.clone() * a.clone().cos() * is.segment.length;
            let evy = pvy + va.clone() * a.clone().sin() * is.segment.length;

            rel_end_x_vec.push(ex);
            rel_end_y_vec.push(ey);
            rel_end_vx_vec.push(evx);
            rel_end_vy_vec.push(evy);
            abs_angle_vec.push(a);
            abs_v_angle_vec.push(va);
            total_mass += is.segment.length * self.mass_density;
        }

        let mut rel_com_x = Tensor::zeros([batch_size], device);
        let mut rel_com_y = Tensor::zeros([batch_size], device);
        let mut rel_com_vx = Tensor::zeros([batch_size], device);
        let mut rel_com_vy = Tensor::zeros([batch_size], device);

        for i in 0..flat_segments.len() {
            let (px, py, pvx, pvy) = match flat_segments[i].parent_idx {
                Some(p_idx) => (
                    rel_end_x_vec[p_idx].clone(),
                    rel_end_y_vec[p_idx].clone(),
                    rel_end_vx_vec[p_idx].clone(),
                    rel_end_vy_vec[p_idx].clone(),
                ),
                None => (
                    Tensor::zeros([batch_size], device),
                    Tensor::zeros([batch_size], device),
                    Tensor::zeros([batch_size], device),
                    Tensor::zeros([batch_size], device),
                ),
            };
            let cx = (px + rel_end_x_vec[i].clone()) * 0.5;
            let cy = (py + rel_end_y_vec[i].clone()) * 0.5;
            let cvx = (pvx + rel_end_vx_vec[i].clone()) * 0.5;
            let cvy = (pvy + rel_end_vy_vec[i].clone()) * 0.5;

            let m = flat_segments[i].segment.length * self.mass_density;
            rel_com_x = rel_com_x + cx * m;
            rel_com_y = rel_com_y + cy * m;
            rel_com_vx = rel_com_vx + cvx * m;
            rel_com_vy = rel_com_vy + cvy * m;
        }
        rel_com_x = rel_com_x / total_mass;
        rel_com_y = rel_com_y / total_mass;
        rel_com_vx = rel_com_vx / total_mass;
        rel_com_vy = rel_com_vy / total_mass;

        let root_x = state.x.clone() - rel_com_x.clone();
        let root_y = state.y.clone() - rel_com_y.clone();
        let root_vx = state.vx.clone() - rel_com_vx;
        let root_vy = state.vy.clone() - rel_com_vy;

        let end_x = Tensor::stack(
            rel_end_x_vec.into_iter().map(|t| t + root_x.clone()).collect::<Vec<_>>(),
            1,
        );
        let end_y = Tensor::stack(
            rel_end_y_vec.into_iter().map(|t| t + root_y.clone()).collect::<Vec<_>>(),
            1,
        );
        let end_vx = Tensor::stack(
            rel_end_vx_vec.into_iter().map(|t| t + root_vx.clone()).collect::<Vec<_>>(),
            1,
        );
        let end_vy = Tensor::stack(
            rel_end_vy_vec.into_iter().map(|t| t + root_vy.clone()).collect::<Vec<_>>(),
            1,
        );

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
        let num_segments = self.morphology.num_joints();
        let g = self.gravity;

        let kin = self.calculate_kinematics(&state);
        let flat_with_parents = self.get_flat_segments();

        // Pre-calculate morphology tensors
        let masses: Tensor<B, 1> = Tensor::from_floats(
            flat_with_parents
                .iter()
                .map(|is| is.segment.length * self.mass_density)
                .collect::<Vec<_>>()
                .as_slice(),
            device,
        );
        let lengths: Tensor<B, 1> = Tensor::from_floats(
            flat_with_parents.iter().map(|is| is.segment.length).collect::<Vec<_>>().as_slice(),
            device,
        );
        let total_mass: f32 =
            flat_with_parents.iter().map(|is| is.segment.length * self.mass_density).sum();

        // --- 4. Forces and Torques (Vectorized) ---
        let k_g = self.ground_stiffness;
        let d_gs = (masses.clone() * k_g).sqrt() * 2.0; // [num_segments]

        let penetration = kin.end_y.clone().mul_scalar(-1.0).clamp_min(0.0); // [batch, num_segments]
        let is_contact = kin.end_y.clone().lower_equal_elem(0.0); // [batch, num_segments]

        let fy_val =
            Tensor::zeros_like(&kin.end_y).mask_where(is_contact.clone(), penetration * k_g);
        let fx_val = (kin.end_vx.clone() * self.friction_sharpness).tanh()
            * fy_val.clone()
            * self.friction
            * -1.0;

        // Root point (top of head) contact force
        let root_mass = flat_with_parents[0].segment.length * self.mass_density * 0.5;
        let root_d_g = (k_g * root_mass).sqrt() * 2.0;
        let root_penetration = kin.root_y.clone().mul_scalar(-1.0).clamp_min(0.0);
        let root_is_contact = kin.root_y.clone().lower_equal_elem(0.0);
        let root_fy = Tensor::zeros_like(&kin.root_y)
            .mask_where(root_is_contact.clone(), root_penetration * k_g);
        let root_fx = (kin.root_vx.clone() * self.friction_sharpness).tanh()
            * root_fy.clone()
            * self.friction
            * -1.0;

        let hull_fx_total = fx_val.clone().sum_dim(1).squeeze_dim(1) + root_fx.clone();
        let hull_fy_total =
            fy_val.clone().sum_dim(1).squeeze_dim(1) + root_fy.clone() - total_mass * g;

        let total_d_g = (is_contact.float() * d_gs.unsqueeze_dim::<2>(0)).sum_dim(1).squeeze_dim(1)
            + root_is_contact.float() * root_d_g;

        // --- 5. Subtree Inertia and Mass ---
        let mut subtree_mass = vec![0.0; flat_with_parents.len()];
        let mut subtree_inertia = vec![0.0; flat_with_parents.len()];

        for i in (0..flat_with_parents.len()).rev() {
            let seg = flat_with_parents[i].segment;
            let m = seg.length * self.mass_density;
            let l = seg.length;
            let i_self = (m * l.powi(2)) / 3.0;

            subtree_mass[i] += m;
            subtree_inertia[i] += i_self;

            if let Some(p_idx) = flat_with_parents[i].parent_idx {
                let p_l = flat_with_parents[p_idx].segment.length;
                subtree_mass[p_idx] += subtree_mass[i];
                subtree_inertia[p_idx] += subtree_inertia[i] + subtree_mass[i] * p_l.powi(2);
            }
        }
        let subtree_inertia_tensor: Tensor<B, 1> =
            Tensor::from_floats(subtree_inertia.as_slice(), device);

        // Total body inertia about COM (Vectorized)
        let i_self_com = (masses.clone() * lengths.powf_scalar(2.0)) / 12.0; // [num_segments]

        let mut pkx_vec = Vec::with_capacity(num_segments);
        let mut pky_vec = Vec::with_capacity(num_segments);
        for i in 0..num_segments {
            match flat_with_parents[i].parent_idx {
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
        let i_com: Tensor<B, 1> = (i_self_com.unsqueeze_dim::<2>(0)
            + (dx.powf_scalar(2.0) + dy.powf_scalar(2.0)) * masses.clone().unsqueeze_dim::<2>(0))
        .sum_dim(1)
        .squeeze_dim(1)
        .clamp_min(0.01);

        // --- 6. Recursive Torque Calculation ---
        let cross_product_vec =
            |rx: Tensor<B, 2>, ry: Tensor<B, 2>, fx: Tensor<B, 2>, fy: Tensor<B, 2>| {
                rx * fy - ry * fx
            };

        let f_ext_x = fx_val.clone();
        let f_ext_y = fy_val.clone() - masses.clone().unsqueeze_dim::<2>(0) * g;

        let t_contact = cross_product_vec(
            kin.end_x.clone() - pkx.clone(),
            kin.end_y.clone() - pky.clone(),
            fx_val.clone(),
            fy_val.clone(),
        );
        let t_grav = cross_product_vec(
            (pkx.clone() + kin.end_x.clone()) * 0.5 - pkx.clone(),
            (pky.clone() + kin.end_y.clone()) * 0.5 - pky.clone(),
            Tensor::zeros_like(&f_ext_x),
            Tensor::ones_like(&f_ext_x) * (-masses.clone().unsqueeze_dim::<2>(0) * g),
        );

        let seg_t_ext = t_contact + t_grav;

        let mut subtree_fx = Vec::with_capacity(num_segments);
        let mut subtree_fy = Vec::with_capacity(num_segments);
        let mut subtree_t_ext = Vec::with_capacity(num_segments);

        for i in 0..num_segments {
            subtree_fx.push(f_ext_x.clone().slice([0..batch_size, i..i + 1]).squeeze_dim(1));
            subtree_fy.push(f_ext_y.clone().slice([0..batch_size, i..i + 1]).squeeze_dim(1));
            subtree_t_ext.push(seg_t_ext.clone().slice([0..batch_size, i..i + 1]).squeeze_dim(1));
        }

        // Add root point force to root segment
        let root_t_ext = (kin.root_x.clone() - state.x.clone()) * root_fy.clone()
            - (kin.root_y.clone() - state.y.clone()) * root_fx.clone();
        subtree_fx[0] = subtree_fx[0].clone() + root_fx;
        subtree_fy[0] = subtree_fy[0].clone() + root_fy;
        subtree_t_ext[0] = subtree_t_ext[0].clone() + root_t_ext;

        let cross_product = |rx: Tensor<B, 1>,
                             ry: Tensor<B, 1>,
                             fx: Tensor<B, 1>,
                             fy: Tensor<B, 1>| { rx * fy - ry * fx };

        for i in (1..num_segments).rev() {
            let p_idx = flat_with_parents[i].parent_idx.unwrap();

            let (pjx, pjy) = match flat_with_parents[p_idx].parent_idx {
                Some(pp_idx) => (
                    self.rel_end_x_from_kin(&kin, pp_idx, batch_size),
                    self.rel_end_y_from_kin(&kin, pp_idx, batch_size),
                ),
                None => (Tensor::zeros([batch_size], device), Tensor::zeros([batch_size], device)),
            };

            let rx_child = self.rel_end_x_from_kin(&kin, p_idx, batch_size) - pjx;
            let ry_child = self.rel_end_y_from_kin(&kin, p_idx, batch_size) - pjy;

            subtree_t_ext[p_idx] = subtree_t_ext[p_idx].clone()
                + subtree_t_ext[i].clone()
                + cross_product(rx_child, ry_child, subtree_fx[i].clone(), subtree_fy[i].clone());
            subtree_fx[p_idx] = subtree_fx[p_idx].clone() + subtree_fx[i].clone();
            subtree_fy[p_idx] = subtree_fy[p_idx].clone() + subtree_fy[i].clone();
        }

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
            flat_with_parents.iter().map(|is| is.segment.angle_min).collect::<Vec<_>>().as_slice(),
            device,
        )
        .unsqueeze_dim::<2>(0);
        let maxs: Tensor<B, 2> = Tensor::<B, 1>::from_floats(
            flat_with_parents.iter().map(|is| is.segment.angle_max).collect::<Vec<_>>().as_slice(),
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
        let subtree_t_ext_tensor = Tensor::stack(subtree_t_ext, 1);

        let next_segment_vs = (state.v_angles.clone()
            + ((joint_t_int + subtree_t_ext_tensor) / i_sub.clone()) * dt)
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

    fn rel_end_x_from_kin<B: Backend>(
        &self,
        kin: &Kinematics<B>,
        idx: usize,
        batch_size: usize,
    ) -> Tensor<B, 1> {
        kin.end_x.clone().slice([0..batch_size, idx..idx + 1]).squeeze_dim(1) - kin.root_x.clone()
    }

    fn rel_end_y_from_kin<B: Backend>(
        &self,
        kin: &Kinematics<B>,
        idx: usize,
        batch_size: usize,
    ) -> Tensor<B, 1> {
        kin.end_y.clone().slice([0..batch_size, idx..idx + 1]).squeeze_dim(1) - kin.root_y.clone()
    }

    pub fn get_observation<B: Backend>(&self, state: &PhysicsState<B>) -> Tensor<B, 2> {
        let kin = self.calculate_kinematics(state);
        let segment_angles = state.angles.clone();
        let segment_vs = state.v_angles.clone() * 0.1;

        // Contacts
        let contacts = kin.end_y.lower_equal_elem(0.0).float();

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
