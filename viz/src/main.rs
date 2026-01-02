use burn::backend::libtorch::{LibTorch, LibTorchDevice};
use burn::prelude::*;
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use macroquad::prelude::*;
use shared::model::ActorCritic;
use shared::physics::{PhysicsState, Segment, WalkerPhysics};

fn window_conf() -> Conf {
    Conf {
        window_title: "Walker Visualization".to_owned(),
        sample_count: 4,
        high_dpi: true,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let device = LibTorchDevice::Mps;
    let physics = WalkerPhysics::default();

    // Initialize state with full random range for visualization
    let mut state = physics.initial_state(1, &device);

    let initial_obs = physics.get_observation(&state);
    let input_dim = initial_obs.dims()[1];
    let action_dim = physics.morphology.num_joints();
    let mut model = ActorCritic::<LibTorch>::new(input_dim, action_dim, &device);
    let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
    let mut model_loaded = false;

    match model.clone().load_file("model", &recorder, &device) {
        Ok(loaded_model) => {
            model = loaded_model;
            model_loaded = true;
            println!("Model loaded successfully.");
        }
        Err(e) => println!("Model file not found at model.bin. Using zero actions. Error: {:?}", e),
    }

    loop {
        clear_background(WHITE);

        let target_vel = 1.0;

        state.target_velocity = Tensor::from_floats([target_vel], &device);

        let action = if model_loaded {
            let obs = physics.get_observation(&state);
            let (mean, _, _) = model.forward(obs);
            mean.tanh()
        } else {
            Tensor::zeros([1, action_dim], &device)
        };

        state = physics.step(state, action.clone());

        draw_simulation(&state, &physics);

        next_frame().await;
    }
}

fn draw_simulation<B: Backend>(state: &PhysicsState<B>, physics: &WalkerPhysics) {
    let kin = physics.calculate_kinematics(state);

    let root_x = kin.root_x.to_data().as_slice::<f32>().unwrap()[0];
    let root_y = kin.root_y.to_data().as_slice::<f32>().unwrap()[0];

    let end_x: Vec<f32> =
        kin.end_x.iter().map(|t| t.to_data().as_slice::<f32>().unwrap()[0]).collect();
    let end_y: Vec<f32> =
        kin.end_y.iter().map(|t| t.to_data().as_slice::<f32>().unwrap()[0]).collect();

    let screen_w = screen_width();
    let screen_h = screen_height();
    let scale = 150.0;

    let ground_y = screen_h * 0.8;
    let draw_x_offset = screen_w / 2.0;

    // Ground
    draw_line(0.0, ground_y, screen_w, ground_y, 2.0, BLACK);

    // Helper to get parent index
    let mut flat_with_parents = Vec::new();
    struct InternalSegment {
        parent_idx: Option<usize>,
    }
    fn flatten_with_parents(s: &Segment, p: Option<usize>, list: &mut Vec<InternalSegment>) {
        let idx = list.len();
        list.push(InternalSegment { parent_idx: p });
        for child in &s.children {
            flatten_with_parents(child, Some(idx), list);
        }
    }
    flatten_with_parents(&physics.morphology.root, None, &mut flat_with_parents);

    for i in 0..flat_with_parents.len() {
        let (px, py) = match flat_with_parents[i].parent_idx {
            Some(p_idx) => (end_x[p_idx], end_y[p_idx]),
            None => (root_x, root_y),
        };
        let ex = end_x[i];
        let ey = end_y[i];

        let s_px = draw_x_offset + (px - root_x) * scale;
        let s_py = ground_y - py * scale;
        let s_ex = draw_x_offset + (ex - root_x) * scale;
        let s_ey = ground_y - ey * scale;

        draw_line(s_px, s_py, s_ex, s_ey, 4.0, BLACK);
        draw_circle(s_ex, s_ey, 3.0, RED);
    }

    // Draw CoM
    let com_x = state.x.clone().to_data().as_slice::<f32>().unwrap()[0];
    let com_y = state.y.clone().to_data().as_slice::<f32>().unwrap()[0];
    let s_com_x = draw_x_offset + (com_x - root_x) * scale;
    let s_com_y = ground_y - com_y * scale;
    draw_circle(s_com_x, s_com_y, 5.0, BLUE);

    draw_text(&format!("X: {:.2}", com_x), 20.0, 20.0, 20.0, BLACK);
    if com_y < 0.4 {
        draw_text("FALLEN", 20.0, 50.0, 30.0, RED);
    }
}
