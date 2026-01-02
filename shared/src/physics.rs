use burn::prelude::*;
use burn::tensor::{Distribution, Int};

#[derive(Clone, Debug)]
pub struct Segment {
    pub length: f32,
    pub angle_min: f32,
    pub angle_max: f32,
    pub children: Vec<Segment>,
}

impl Segment {
    pub fn mass(&self) -> f32 {
        self.length * 5.0
    }
}

#[derive(Clone, Debug)]
pub struct Morphology {
    pub root: Segment,
}

impl Morphology {
    pub fn humanoid() -> Self {
        Self {
            root: Segment {
                length: 0.2,      // Head
                angle_min: -3.14, // Full range
                angle_max: 3.14,
                children: vec![
                    // Torso starts at the neck (end of head)
                    Segment {
                        length: 0.5,
                        angle_min: -1.0,
                        angle_max: 1.0,
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
                    // Left Arm
                    Segment {
                        length: 0.3,
                        angle_min: -1.5,
                        angle_max: 1.5,
                        children: vec![Segment {
                            length: 0.3,
                            angle_min: -2.0,
                            angle_max: 0.0,
                            children: vec![],
                        }],
                    },
                    // Right Arm
                    Segment {
                        length: 0.3,
                        angle_min: -1.5,
                        angle_max: 1.5,
                        children: vec![Segment {
                            length: 0.3,
                            angle_min: 0.0,
                            angle_max: 2.0,
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
    pub end_x: Vec<Tensor<B, 1>>,
    pub end_y: Vec<Tensor<B, 1>>,
    pub end_vx: Vec<Tensor<B, 1>>,
    pub end_vy: Vec<Tensor<B, 1>>,
}

pub struct WalkerPhysics {
    pub gravity: f32,
    pub morphology: Morphology,
    pub time_step: f32,
    pub friction: f32,
    pub torque_magnitude: f32,
    pub joint_damping: f32,
    pub sub_steps: usize,
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
        }
    }
}

impl WalkerPhysics {
    pub fn calculate_kinematics<B: Backend>(&self, state: &PhysicsState<B>) -> Kinematics<B> {
        let batch_size = state.x.dims()[0];
        let device = &state.x.device();
        let num_segments = self.morphology.num_joints();

        let mut segment_angles = Vec::with_capacity(num_segments);
        let mut segment_vs = Vec::with_capacity(num_segments);
        for i in 0..num_segments {
            segment_angles
                .push(state.angles.clone().slice([0..batch_size, i..i + 1]).squeeze_dim(1));
            segment_vs.push(state.v_angles.clone().slice([0..batch_size, i..i + 1]).squeeze_dim(1));
        }

        struct InternalSegment<'a> {
            segment: &'a Segment,
            parent_idx: Option<usize>,
        }

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

        let mut rel_end_x: Vec<Tensor<B, 1>> = Vec::with_capacity(flat_segments.len());
        let mut rel_end_y: Vec<Tensor<B, 1>> = Vec::with_capacity(flat_segments.len());
        let mut rel_end_vx: Vec<Tensor<B, 1>> = Vec::with_capacity(flat_segments.len());
        let mut rel_end_vy: Vec<Tensor<B, 1>> = Vec::with_capacity(flat_segments.len());
        let mut abs_angle: Vec<Tensor<B, 1>> = Vec::with_capacity(flat_segments.len());
        let mut abs_v_angle: Vec<Tensor<B, 1>> = Vec::with_capacity(flat_segments.len());

        let mut total_mass = 0.0;
        for (i, is) in flat_segments.iter().enumerate() {
            let j_angle = segment_angles[i].clone();
            let j_v = segment_vs[i].clone();

            let (px, py, pa, pvx, pvy, pva) = match is.parent_idx {
                Some(p_idx) => (
                    rel_end_x[p_idx].clone(),
                    rel_end_y[p_idx].clone(),
                    abs_angle[p_idx].clone(),
                    rel_end_vx[p_idx].clone(),
                    rel_end_vy[p_idx].clone(),
                    abs_v_angle[p_idx].clone(),
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

            rel_end_x.push(ex);
            rel_end_y.push(ey);
            rel_end_vx.push(evx);
            rel_end_vy.push(evy);
            abs_angle.push(a);
            abs_v_angle.push(va);
            total_mass += is.segment.mass();
        }

        let mut rel_com_x = Tensor::zeros([batch_size], device);
        let mut rel_com_y = Tensor::zeros([batch_size], device);
        let mut rel_com_vx = Tensor::zeros([batch_size], device);
        let mut rel_com_vy = Tensor::zeros([batch_size], device);

        for i in 0..flat_segments.len() {
            let (px, py, pvx, pvy) = match flat_segments[i].parent_idx {
                Some(p_idx) => (
                    rel_end_x[p_idx].clone(),
                    rel_end_y[p_idx].clone(),
                    rel_end_vx[p_idx].clone(),
                    rel_end_vy[p_idx].clone(),
                ),
                None => (
                    Tensor::zeros([batch_size], device),
                    Tensor::zeros([batch_size], device),
                    Tensor::zeros([batch_size], device),
                    Tensor::zeros([batch_size], device),
                ),
            };
            let cx = (px + rel_end_x[i].clone()) * 0.5;
            let cy = (py + rel_end_y[i].clone()) * 0.5;
            let cvx = (pvx + rel_end_vx[i].clone()) * 0.5;
            let cvy = (pvy + rel_end_vy[i].clone()) * 0.5;

            rel_com_x = rel_com_x + cx * flat_segments[i].segment.mass();
            rel_com_y = rel_com_y + cy * flat_segments[i].segment.mass();
            rel_com_vx = rel_com_vx + cvx * flat_segments[i].segment.mass();
            rel_com_vy = rel_com_vy + cvy * flat_segments[i].segment.mass();
        }
        rel_com_x = rel_com_x / total_mass;
        rel_com_y = rel_com_y / total_mass;
        rel_com_vx = rel_com_vx / total_mass;
        rel_com_vy = rel_com_vy / total_mass;

        let root_x = state.x.clone() - rel_com_x.clone();
        let root_y = state.y.clone() - rel_com_y.clone();
        let root_vx = state.vx.clone() - rel_com_vx;
        let root_vy = state.vy.clone() - rel_com_vy;

        let mut end_x = Vec::with_capacity(flat_segments.len());
        let mut end_y = Vec::with_capacity(flat_segments.len());
        let mut end_vx = Vec::with_capacity(flat_segments.len());
        let mut end_vy = Vec::with_capacity(flat_segments.len());

        for i in 0..flat_segments.len() {
            end_x.push(rel_end_x[i].clone() + root_x.clone());
            end_y.push(rel_end_y[i].clone() + root_y.clone());
            end_vx.push(rel_end_vx[i].clone() + root_vx.clone());
            end_vy.push(rel_end_vy[i].clone() + root_vy.clone());
        }

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
        let end_y = kin.end_y.clone();
        let end_vx = kin.end_vx.clone();

        struct InternalSegment<'a> {
            segment: &'a Segment,
            parent_idx: Option<usize>,
        }
        let mut flat_with_parents = Vec::new();
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
        flatten_with_parents(&self.morphology.root, None, &mut flat_with_parents);

        let mut total_mass = 0.0;
        for is in &flat_with_parents {
            total_mass += is.segment.mass();
        }

        // --- 4. Forces and Torques ---
        let k_g = 5000.0;
        let mut fx = Vec::with_capacity(flat_with_parents.len());
        let mut fy = Vec::with_capacity(flat_with_parents.len());
        let mut hull_fx_total = Tensor::zeros([batch_size], device);
        let mut hull_fy_total = Tensor::zeros([batch_size], device);
        let mut total_d_g = Tensor::zeros([batch_size], device);

        // Root point (top of head) contact force
        let (root_fx, root_fy) = {
            let mass = flat_with_parents[0].segment.mass() * 0.5;
            let d_g = (k_g * mass).sqrt() * 2.0;
            let y = kin.root_y.clone();
            let vx = kin.root_vx.clone();
            let penetration = y.clone().mul_scalar(-1.0).clamp_min(0.0);
            let is_contact = y.lower_equal_elem(0.0);
            let fy_val = Tensor::zeros([batch_size], device)
                .mask_where(is_contact.clone(), penetration * k_g);
            let fx_val = vx.tanh() * fy_val.clone() * self.friction * -1.0;

            hull_fx_total = hull_fx_total + fx_val.clone();
            hull_fy_total = hull_fy_total + fy_val.clone();
            total_d_g = total_d_g
                + Tensor::zeros([batch_size], device)
                    .mask_where(is_contact, Tensor::ones([batch_size], device) * d_g);
            (fx_val, fy_val)
        };

        for i in 0..flat_with_parents.len() {
            let mass = flat_with_parents[i].segment.mass();
            let d_g = (k_g * mass).sqrt() * 2.0;
            let y = end_y[i].clone();
            let vx = end_vx[i].clone();

            let penetration = y.clone().mul_scalar(-1.0).clamp_min(0.0);
            let is_contact = y.lower_equal_elem(0.0);

            let fy_spring = penetration * k_g;
            let fy_val =
                Tensor::zeros([batch_size], device).mask_where(is_contact.clone(), fy_spring);

            let fx_val = vx.tanh() * fy_val.clone() * self.friction * -1.0;

            fx.push(fx_val.clone());
            fy.push(fy_val.clone());
            hull_fx_total = hull_fx_total + fx_val;
            hull_fy_total = hull_fy_total + fy_val;
            total_d_g = total_d_g
                + Tensor::zeros([batch_size], device)
                    .mask_where(is_contact, Tensor::ones([batch_size], device) * d_g);
        }
        hull_fy_total = hull_fy_total - total_mass * g;

        // --- 5. Subtree Inertia and Mass ---
        let mut subtree_mass = vec![0.0; flat_with_parents.len()];
        let mut subtree_inertia = vec![0.0; flat_with_parents.len()];

        for i in (0..flat_with_parents.len()).rev() {
            let seg = flat_with_parents[i].segment;
            let m = seg.mass();
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

        // Calculate total body inertia about the Center of Mass directly for stability
        let mut i_com = Tensor::zeros([batch_size], device);
        for i in 0..flat_with_parents.len() {
            let m = flat_with_parents[i].segment.mass();
            let l = flat_with_parents[i].segment.length;
            let i_self_com = (m * l.powi(2)) / 12.0;
            let (pkx, pky) = match flat_with_parents[i].parent_idx {
                Some(p_idx) => (kin.end_x[p_idx].clone(), kin.end_y[p_idx].clone()),
                None => (kin.root_x.clone(), kin.root_y.clone()),
            };
            let scx = (pkx + kin.end_x[i].clone()) * 0.5;
            let scy = (pky + kin.end_y[i].clone()) * 0.5;
            let dx = scx - state.x.clone();
            let dy = scy - state.y.clone();
            i_com = i_com + (i_self_com + (dx.powf_scalar(2.0) + dy.powf_scalar(2.0)) * m);
        }
        let i_com = i_com.clamp_min(0.01);

        let cross_product = |rx: Tensor<B, 1>,
                             ry: Tensor<B, 1>,
                             fx: Tensor<B, 1>,
                             fy: Tensor<B, 1>| { rx * fy - ry * fx };

        let mut segment_angles = Vec::with_capacity(num_segments);
        let mut segment_vs = Vec::with_capacity(num_segments);
        let mut actions = Vec::with_capacity(num_segments);
        for i in 0..num_segments {
            segment_angles
                .push(state.angles.clone().slice([0..batch_size, i..i + 1]).squeeze_dim(1));
            segment_vs.push(state.v_angles.clone().slice([0..batch_size, i..i + 1]).squeeze_dim(1));
            actions.push(action.clone().slice([0..batch_size, i..i + 1]).squeeze_dim(1));
        }

        // --- 6. Recursive Torque Calculation ---
        let mut seg_fx = Vec::with_capacity(num_segments);
        let mut seg_fy = Vec::with_capacity(num_segments);
        let mut seg_t_ext = Vec::with_capacity(num_segments);

        for i in 0..num_segments {
            let f_ext_x = fx[i].clone();
            let f_ext_y = fy[i].clone() - flat_with_parents[i].segment.mass() * g;

            seg_fx.push(f_ext_x.clone());
            seg_fy.push(f_ext_y.clone());

            let (pkx, pky) = match flat_with_parents[i].parent_idx {
                Some(p_idx) => (kin.end_x[p_idx].clone(), kin.end_y[p_idx].clone()),
                None => (state.x.clone(), state.y.clone()),
            };

            let t_contact = cross_product(
                kin.end_x[i].clone() - pkx.clone(),
                kin.end_y[i].clone() - pky.clone(),
                fx[i].clone(),
                fy[i].clone(),
            );
            let t_grav = cross_product(
                (pkx.clone() + kin.end_x[i].clone()) * 0.5 - pkx,
                (pky.clone() + kin.end_y[i].clone()) * 0.5 - pky,
                Tensor::zeros_like(&f_ext_x),
                Tensor::ones_like(&f_ext_x) * (-flat_with_parents[i].segment.mass() * g),
            );

            seg_t_ext.push(t_contact + t_grav);
        }

        // Add root point force to root segment (index 0)
        seg_fx[0] = seg_fx[0].clone() + root_fx.clone();
        seg_fy[0] = seg_fy[0].clone() + root_fy.clone();
        seg_t_ext[0] = seg_t_ext[0].clone()
            + cross_product(
                kin.root_x.clone() - state.x.clone(),
                kin.root_y.clone() - state.y.clone(),
                root_fx,
                root_fy,
            );

        let mut subtree_fx = seg_fx;
        let mut subtree_fy = seg_fy;
        let mut subtree_t_ext = seg_t_ext;

        for i in (1..num_segments).rev() {
            let p_idx = flat_with_parents[i].parent_idx.unwrap();

            let (pjx, pjy) = match flat_with_parents[p_idx].parent_idx {
                Some(pp_idx) => (kin.end_x[pp_idx].clone(), kin.end_y[pp_idx].clone()),
                None => (state.x.clone(), state.y.clone()),
            };

            let rx_child = kin.end_x[p_idx].clone() - pjx;
            let ry_child = kin.end_y[p_idx].clone() - pjy;

            subtree_t_ext[p_idx] = subtree_t_ext[p_idx].clone()
                + subtree_t_ext[i].clone()
                + cross_product(rx_child, ry_child, subtree_fx[i].clone(), subtree_fy[i].clone());
            subtree_fx[p_idx] = subtree_fx[p_idx].clone() + subtree_fx[i].clone();
            subtree_fy[p_idx] = subtree_fy[p_idx].clone() + subtree_fy[i].clone();
        }

        // --- 7. Internal Torques (Joint Limits, Actions) ---
        let mut joint_t_int = Vec::with_capacity(num_segments);
        let k_limit_base = 1000.0;

        for i in 0..num_segments {
            let is = &flat_with_parents[i];
            let i_sub = if i == 0 {
                i_com.clone()
            } else {
                Tensor::ones([batch_size], device) * subtree_inertia[i].max(0.01)
            };

            let j_angle = segment_angles[i].clone();
            let j_v = segment_vs[i].clone();

            let act = if i == 0 {
                Tensor::zeros([batch_size], device)
            } else {
                actions[i].clone() * self.torque_magnitude * i_sub.clone()
            };

            let k_limit = i_sub.clone() * k_limit_base;
            let d_limit = (k_limit.clone() * i_sub.clone()).sqrt() * 2.0;
            let diff_min = j_angle.clone().neg().add_scalar(is.segment.angle_min);
            let t_min = (diff_min.clone().clamp_min(0.0) * k_limit.clone()
                - j_v.clone() * d_limit.clone())
            .mask_where(diff_min.lower_elem(0.0), Tensor::zeros_like(&j_angle));
            let diff_max = j_angle.clone().add_scalar(-is.segment.angle_max);
            let t_max = (diff_max.clone().clamp_min(0.0) * k_limit + j_v.clone() * d_limit) * -1.0;
            let t_max = t_max.mask_where(diff_max.lower_elem(0.0), Tensor::zeros_like(&j_angle));

            joint_t_int.push(act + t_min + t_max);
        }

        // --- 8. Integration ---
        let dt = self.time_step / self.sub_steps as f32;
        let next_com_vx = (state.vx + (hull_fx_total / total_mass) * dt)
            / (Tensor::ones_like(&state.y) + (total_d_g.clone() / total_mass) * dt);
        let next_com_vy = (state.vy + (hull_fy_total / total_mass) * dt)
            / (Tensor::ones_like(&state.y) + (total_d_g / total_mass) * dt);
        let next_com_x = state.x + next_com_vx.clone() * dt;
        let next_com_y = state.y + next_com_vy.clone() * dt;

        let mut next_segment_vs = Vec::with_capacity(num_segments);
        for i in 0..num_segments {
            let i_sub = if i == 0 {
                i_com.clone()
            } else {
                Tensor::ones([batch_size], device) * subtree_inertia[i].max(0.01)
            };
            let d_joint = i_sub.clone() * self.joint_damping * 10.0;
            let nv = (segment_vs[i].clone()
                + ((joint_t_int[i].clone() + subtree_t_ext[i].clone()) / i_sub.clone()) * dt)
                / (Tensor::ones_like(&i_sub) + (d_joint / i_sub) * dt);
            next_segment_vs.push(nv.unsqueeze_dim(1));
        }
        let next_segment_vs = Tensor::cat(next_segment_vs, 1);
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
        let num_segments = self.morphology.num_joints();

        let kin = self.calculate_kinematics(state);
        let segment_angles = state.angles.clone();
        let segment_vs = state.v_angles.clone() * 0.1;

        // Contacts
        let mut contacts = Vec::with_capacity(num_segments);
        for i in 0..num_segments {
            let ey = kin.end_y[i].clone();
            contacts.push(ey.lower_equal_elem(0.0).float().unsqueeze_dim(1));
        }

        let hull_y = state.y.clone().unsqueeze_dim(1);
        let hull_vx = state.vx.clone().unsqueeze_dim(1) * 0.1;
        let hull_vy = state.vy.clone().unsqueeze_dim(1) * 0.1;

        let mut obs = vec![hull_y - 0.8, segment_angles, hull_vx, hull_vy, segment_vs];
        obs.extend(contacts);
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
