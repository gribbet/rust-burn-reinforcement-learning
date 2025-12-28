use burn::prelude::*;
use burn::tensor::{Distribution, Int};

#[derive(Clone, Debug)]
pub struct Segment {
    pub length: f32,
    pub mass: f32,
    pub angle_min: f32,
    pub angle_max: f32,
    pub children: Vec<Segment>,
}

#[derive(Clone, Debug)]
pub struct Morphology {
    pub root: Segment,
}

impl Morphology {
    pub fn biped() -> Self {
        let leg = |angle_min, angle_max| Segment {
            length: 0.5,
            mass: 1.0,
            angle_min,
            angle_max,
            children: vec![Segment {
                length: 0.5,
                mass: 1.0,
                angle_min: -2.5,
                angle_max: 0.0,
                children: vec![],
            }],
        };

        Self {
            root: Segment {
                length: 0.5,
                mass: 10.0,
                angle_min: -10.0,
                angle_max: 10.0,
                children: vec![
                    leg(-0.7, 1.0), // Left Leg
                    leg(-0.7, 1.0), // Right Leg
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
            morphology: Morphology::biped(),
            time_step: 0.02,
            friction: 0.3,
            torque_magnitude: 20.0,
            joint_damping: 0.1,
            sub_steps: 3,
        }
    }
}

impl WalkerPhysics {
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

        // --- 1. Forward Kinematics ---
        let hull_x = state.x.clone();
        let hull_y = state.y.clone();
        let segment_angles = state.angles.clone();

        let hull_vx = state.vx.clone();
        let hull_vy = state.vy.clone();
        let segment_vs = state.v_angles.clone();

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

        let mut end_x: Vec<Tensor<B, 1>> = Vec::with_capacity(flat_segments.len());
        let mut end_y: Vec<Tensor<B, 1>> = Vec::with_capacity(flat_segments.len());
        let mut abs_angle: Vec<Tensor<B, 1>> = Vec::with_capacity(flat_segments.len());
        let mut end_vx: Vec<Tensor<B, 1>> = Vec::with_capacity(flat_segments.len());
        let mut end_vy: Vec<Tensor<B, 1>> = Vec::with_capacity(flat_segments.len());
        let mut abs_v_angle: Vec<Tensor<B, 1>> = Vec::with_capacity(flat_segments.len());

        for (i, is) in flat_segments.iter().enumerate() {
            let j_angle = segment_angles.clone().slice([0..batch_size, i..i + 1]).squeeze_dim(1);
            let j_v = segment_vs.clone().slice([0..batch_size, i..i + 1]).squeeze_dim(1);

            let (px, py, pa, pvx, pvy, pva) = match is.parent_idx {
                Some(p_idx) => (
                    end_x[p_idx].clone(),
                    end_y[p_idx].clone(),
                    abs_angle[p_idx].clone(),
                    end_vx[p_idx].clone(),
                    end_vy[p_idx].clone(),
                    abs_v_angle[p_idx].clone(),
                ),
                None => (
                    hull_x.clone(),
                    hull_y.clone(),
                    Tensor::zeros_like(&hull_x),
                    hull_vx.clone(),
                    hull_vy.clone(),
                    Tensor::zeros_like(&hull_vx),
                ),
            };

            let a = pa + j_angle;
            let ex = px + a.clone().sin() * is.segment.length;
            let ey = py - a.clone().cos() * is.segment.length;

            let va = pva + j_v;
            let evx = pvx + va.clone() * a.clone().cos() * is.segment.length;
            let evy = pvy + va.clone() * a.clone().sin() * is.segment.length;

            end_x.push(ex);
            end_y.push(ey);
            abs_angle.push(a);
            end_vx.push(evx);
            end_vy.push(evy);
            abs_v_angle.push(va);
        }

        // --- 2. Forces (Unified Collision) ---
        let k_g = 4000.0;

        let mut fx = Vec::with_capacity(flat_segments.len());
        let mut fy = Vec::with_capacity(flat_segments.len());
        let mut segment_d_g = Vec::with_capacity(flat_segments.len());

        for i in 0..flat_segments.len() {
            let mass = flat_segments[i].segment.mass;
            let d_g = (k_g * mass).sqrt() * 2.0; // Critical damping for this segment

            let y = end_y[i].clone();
            let vx = end_vx[i].clone();

            let penetration = y.clone().mul_scalar(-1.0).clamp_min(0.0);
            let is_contact = y.lower_equal_elem(0.0);

            // Spring force (explicit)
            let fy_spring = penetration * k_g;
            let fy_val = Tensor::zeros_like(&fy_spring).mask_where(is_contact.clone(), fy_spring);

            // Friction (explicit part)
            let fx_val = vx.tanh() * fy_val.clone() * self.friction * -1.0;

            fx.push(fx_val);
            fy.push(fy_val);
            segment_d_g.push(d_g);
        }

        // Root head contact
        let root_mass = self.morphology.root.mass;
        let root_d_g = (k_g * root_mass).sqrt() * 2.0;
        let h_penetration = hull_y.clone().mul_scalar(-1.0).clamp_min(0.0);
        let h_is_contact = hull_y.clone().lower_equal_elem(0.0);
        let hc_fy = Tensor::zeros_like(&h_penetration)
            .mask_where(h_is_contact.clone(), h_penetration * k_g);
        let hc_fx = hull_vx.clone().tanh() * hc_fy.clone() * self.friction * -1.0;

        // --- 3. Torques ---
        let cross_product = |rx: Tensor<B, 1>,
                             ry: Tensor<B, 1>,
                             fx: Tensor<B, 1>,
                             fy: Tensor<B, 1>| { rx * fy - ry * fx };
        let t_gravity = |rx: Tensor<B, 1>, mass: f32| rx * (-mass * g);

        let mut hull_fx_total = hc_fx;
        let mut hull_fy_total = hc_fy;
        let mut total_mass = root_mass;
        let mut total_d_g = Tensor::zeros_like(&hull_y)
            .mask_where(h_is_contact, Tensor::ones_like(&hull_y) * root_d_g);

        for i in 0..flat_segments.len() {
            hull_fx_total = hull_fx_total + fx[i].clone();
            hull_fy_total = hull_fy_total + fy[i].clone();
            total_mass += flat_segments[i].segment.mass;

            let is_contact = end_y[i].clone().lower_equal_elem(0.0);
            total_d_g = total_d_g
                + Tensor::zeros_like(&hull_y)
                    .mask_where(is_contact, Tensor::ones_like(&hull_y) * segment_d_g[i]);
        }
        hull_fy_total = hull_fy_total - total_mass * g;

        let mut joint_t_ext = Vec::with_capacity(flat_segments.len());
        let mut joint_t_int = Vec::with_capacity(flat_segments.len());
        let k_limit = 1000.0;
        let d_limit = (k_limit * 1.0f32).sqrt() * 2.0;

        for i in 0..flat_segments.len() {
            let is = &flat_segments[i];
            let (jx, jy) = match is.parent_idx {
                Some(p_idx) => (end_x[p_idx].clone(), end_y[p_idx].clone()),
                None => (hull_x.clone(), hull_y.clone()),
            };

            let mut t_ext = Tensor::zeros([batch_size], device);
            for k in 0..flat_segments.len() {
                let mut in_subtree = false;
                let mut curr = Some(k);
                while let Some(curr_idx) = curr {
                    if curr_idx == i {
                        in_subtree = true;
                        break;
                    }
                    curr = flat_segments[curr_idx].parent_idx;
                }
                if in_subtree {
                    let rx = end_x[k].clone() - jx.clone();
                    let ry = end_y[k].clone() - jy.clone();
                    t_ext =
                        t_ext + cross_product(rx.clone(), ry.clone(), fx[k].clone(), fy[k].clone());
                    t_ext = t_ext + t_gravity(rx, flat_segments[k].segment.mass);
                }
            }
            joint_t_ext.push(t_ext);

            let j_angle = segment_angles.clone().slice([0..batch_size, i..i + 1]).squeeze_dim(1);
            let j_v = segment_vs.clone().slice([0..batch_size, i..i + 1]).squeeze_dim(1);
            let act = action.clone().slice([0..batch_size, i..i + 1]).squeeze_dim(1)
                * self.torque_magnitude;

            let t_damp = j_v.clone() * -self.joint_damping;
            let diff_min = is.segment.angle_min - j_angle.clone();
            let t_min = (diff_min.clone().clamp_min(0.0) * k_limit - j_v.clone() * d_limit)
                .mask_where(diff_min.lower_elem(0.0), Tensor::zeros_like(&j_angle));
            let diff_max = j_angle.clone() - is.segment.angle_max;
            let t_max = (diff_max.clone().clamp_min(0.0) * k_limit + j_v.clone() * d_limit) * -1.0;
            let t_max = t_max.mask_where(diff_max.lower_elem(0.0), Tensor::zeros_like(&j_angle));

            let t_int = act + t_min + t_max + t_damp;
            joint_t_int.push(t_int.clone());
        }

        // --- 4. Integration ---
        let dt = self.time_step / self.sub_steps as f32;
        let next_hull_vx = hull_vx + (hull_fx_total / total_mass) * dt;
        let next_hull_vy = (hull_vy + (hull_fy_total / total_mass) * dt)
            / (Tensor::ones_like(&hull_y) + (total_d_g / total_mass) * dt);

        let next_hull_x = hull_x + next_hull_vx.clone() * dt;
        let next_hull_y = hull_y + next_hull_vy.clone() * dt;

        let mut next_segment_vs = Vec::with_capacity(num_segments);
        for i in 0..num_segments {
            let i_seg = if i == 0 { self.morphology.root.mass * 0.5 } else { 0.5 };
            let d_joint = self.joint_damping;

            // Implicit joint damping update
            let nv = (segment_vs.clone().slice([0..batch_size, i..i + 1]).squeeze_dim(1)
                + ((joint_t_int[i].clone() + joint_t_ext[i].clone()) / i_seg) * dt)
                / (1.0 + (d_joint / i_seg) * dt);

            next_segment_vs.push(nv.unsqueeze_dim(1));
        }
        let next_segment_vs = Tensor::cat(next_segment_vs, 1);
        let next_segment_angles = segment_angles + next_segment_vs.clone() * dt;

        PhysicsState {
            x: next_hull_x,
            y: next_hull_y,
            angles: next_segment_angles,
            vx: next_hull_vx,
            vy: next_hull_vy,
            v_angles: next_segment_vs,
            time: state.time.add_scalar(1),
            target_velocity: state.target_velocity,
        }
    }

    pub fn get_observation<B: Backend>(&self, state: &PhysicsState<B>) -> Tensor<B, 2> {
        let batch_size = state.x.dims()[0];
        let num_segments = self.morphology.num_joints();

        let hull_y = state.y.clone().unsqueeze_dim(1);
        let hull_vx = state.vx.clone().unsqueeze_dim(1) * 0.1;
        let hull_vy = state.vy.clone().unsqueeze_dim(1) * 0.1;
        let segment_angles = state.angles.clone();
        let segment_vs = state.v_angles.clone() * 0.1;

        // Contacts are calculated on the fly for observation
        let mut contacts = Vec::with_capacity(num_segments);
        let mut end_y: Vec<Tensor<B, 1>> = Vec::with_capacity(num_segments);
        let mut abs_angle: Vec<Tensor<B, 1>> = Vec::with_capacity(num_segments);

        let flat_segments = self.morphology.flatten_segments();
        let mut segment_info = Vec::new();
        fn flatten_with_parents_info(
            s: &Segment,
            p: Option<usize>,
            list: &mut Vec<(usize, Option<usize>)>,
        ) {
            let idx = list.len();
            list.push((idx, p));
            for child in &s.children {
                flatten_with_parents_info(child, Some(idx), list);
            }
        }
        flatten_with_parents_info(&self.morphology.root, None, &mut segment_info);

        for (i, p_idx) in segment_info {
            let j_angle = segment_angles.clone().slice([0..batch_size, i..i + 1]).squeeze_dim(1);
            let (py, pa) = match p_idx {
                Some(pi) => (end_y[pi].clone(), abs_angle[pi].clone()),
                None => (hull_y.clone().squeeze_dim(1), Tensor::zeros_like(&hull_y).squeeze_dim(1)),
            };
            let a = pa + j_angle;
            let ey = py - a.clone().cos() * flat_segments[i].length;
            contacts.push(ey.clone().lower_equal_elem(0.0).float().unsqueeze_dim(1));
            end_y.push(ey);
            abs_angle.push(a);
        }

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
        for (i, s) in segments.into_iter().enumerate() {
            let angle = if i == 0 {
                Tensor::<B, 1>::zeros([batch_size], device)
            } else {
                Tensor::<B, 1>::random(
                    [batch_size],
                    Distribution::Uniform(s.angle_min as f64, s.angle_max as f64),
                    device,
                )
            };
            angles.push(angle.unsqueeze_dim(1));
        }

        PhysicsState {
            x: Tensor::zeros([batch_size], device),
            y: Tensor::ones([batch_size], device) * 2.0,
            angles: Tensor::cat(angles, 1),
            vx: Tensor::zeros([batch_size], device),
            vy: Tensor::zeros([batch_size], device),
            v_angles: Tensor::zeros([batch_size, num_segments], device),
            time: Tensor::zeros([batch_size], device),
            target_velocity: Tensor::zeros([batch_size], device),
        }
    }
}
