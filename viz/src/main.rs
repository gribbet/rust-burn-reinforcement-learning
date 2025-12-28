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

    // Manually initialize state to be upright and deterministic for visualization
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
            mean.clamp(-1.0, 1.0)
        } else {
            Tensor::zeros([1, action_dim], &device)
        };

        state = physics.step(state, action.clone());

        draw_simulation(&state, &physics);

        next_frame().await;
    }
}

fn draw_simulation<B: Backend>(state: &PhysicsState<B>, physics: &WalkerPhysics) {
    let x_data = state.x.clone().into_data().convert::<f32>();
    let y_data = state.y.clone().into_data().convert::<f32>();
    let angles_data = state.angles.clone().into_data().convert::<f32>();

    let x_vec = x_data.as_slice::<f32>().unwrap();
    let y_vec = y_data.as_slice::<f32>().unwrap();
    let angles_vec = angles_data.as_slice::<f32>().unwrap();

    let hull_x = x_vec[0];
    let hull_y = y_vec[0];

    let screen_w = screen_width();
    let screen_h = screen_height();
    let scale = 50.0;

    let ground_y = screen_h * 0.8;
    let draw_x = (screen_w / 5.0) + hull_x * scale; // Camera fixed, agent moves
    let draw_y = ground_y - hull_y * scale;

    // Ground
    draw_line(0.0, ground_y, screen_w, ground_y, 2.0, BLACK);

    let mut joint_idx = 0;

    fn draw_recursive(
        segments: &[Segment],
        px: f32,
        py: f32,
        pa: f32,
        angles: &[f32],
        idx: &mut usize,
        scale: f32,
    ) {
        for s in segments {
            let a = pa + angles[*idx];
            let ex = px + a.sin() * s.length * scale;
            let ey = py + a.cos() * s.length * scale;
            draw_line(px, py, ex, ey, 4.0, BLACK);
            *idx += 1;
            draw_recursive(&s.children, ex, ey, a, angles, idx, scale);
        }
    }

    draw_recursive(
        std::slice::from_ref(&physics.morphology.root),
        draw_x,
        draw_y,
        0.0,
        &angles_vec,
        &mut joint_idx,
        scale,
    );

    draw_text(&format!("X: {:.2}", hull_x), 20.0, 20.0, 20.0, BLACK);
}
