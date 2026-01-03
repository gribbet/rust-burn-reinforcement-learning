use burn::backend::libtorch::{LibTorch, LibTorchDevice};
use burn::prelude::*;
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use macroquad::prelude::*;
use shared::model::ActorCritic;
use shared::physics::{PhysicsState, Walker, WalkerConfig};

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
    let config = WalkerConfig::default();
    let walker = Walker::new(config, &device);

    // Initialize state with full random range for visualization
    let mut state = walker.initial_state(1, &device);

    let initial_obs = walker.get_observation(&state);
    let input_dim = initial_obs.dims()[1];
    let action_dim = walker.action_dim();
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

        state.target_velocity = Tensor::from_floats([1.0], &device);

        let action = if model_loaded {
            let obs = walker.get_observation(&state);
            let (mean, _, _) = model.forward(obs);
            mean.tanh()
        } else {
            // Zero action to verify physics stability
            Tensor::zeros([1, action_dim], &device)
        };

        state = walker.step(state, action.clone());

        draw_simulation(&state, &walker);

        next_frame().await;
    }
}

fn draw_simulation<B: Backend>(state: &PhysicsState<B>, walker: &Walker<B>) {
    let positions = state.positions.clone().to_data();
    let time = state.time.clone().to_data().as_slice::<f32>().unwrap()[0];
    // positions is [batch, n_particles, 2]
    // We assume batch_size = 1 for viz
    let pos_slice = positions.as_slice::<f32>().unwrap();
    // Layout: [p0_x, p0_y, p1_x, p1_y, ...]

    let get_pos = |i: usize| (pos_slice[i * 2], pos_slice[i * 2 + 1]);

    let screen_w = screen_width();
    let screen_h = screen_height();
    let scale = 50.0;

    let ground_y = screen_h * 0.8;
    let draw_x_offset = screen_w / 2.0;

    // Ground
    draw_line(0.0, ground_y, screen_w, ground_y, 2.0, BLACK);

    // Draw edges
    for &(p1, p2, _) in &walker.edges {
        let (x1, y1) = get_pos(p1);
        let (x2, y2) = get_pos(p2);

        let s_x1 = draw_x_offset + x1 * scale;
        let s_y1 = ground_y - y1 * scale;
        let s_x2 = draw_x_offset + x2 * scale;
        let s_y2 = ground_y - y2 * scale;

        draw_line(s_x1, s_y1, s_x2, s_y2, 4.0, BLACK);
    }

    // Draw Info
    let (root_x, root_y) = get_pos(0);
    draw_text(&format!("X: {:.2}", root_x), 20.0, 20.0, 20.0, BLACK);
    draw_text(&format!("Time: {:.2}s", time), 20.0, 40.0, 20.0, BLACK);
    if root_y < walker.config.fall_y {
        draw_text("FALLEN", 20.0, 70.0, 30.0, RED);
    }
}
