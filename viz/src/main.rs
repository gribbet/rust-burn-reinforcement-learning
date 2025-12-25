use burn::backend::libtorch::{LibTorch, LibTorchDevice};
use burn::prelude::*;
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use macroquad::prelude::*;
use shared::model::ActorCritic;
use shared::physics::{BipedalWalkerPhysics, PhysicsState};

fn window_conf() -> Conf {
    Conf {
        window_title: "Bipedal Walker".to_owned(),
        sample_count: 4,
        high_dpi: true,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let device = LibTorchDevice::Mps;
    let physics = BipedalWalkerPhysics::default();

    // Manually initialize state to be upright and deterministic for visualization
    let mut state = physics.initial_state(1, &device);

    let initial_obs = physics.get_observation(&state);
    let input_dim = initial_obs.dims()[1];
    let mut model = ActorCritic::<LibTorch>::new(input_dim, 4, &device);
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

        let target_vel = if is_key_down(KeyCode::Right) {
            1.0
        } else if is_key_down(KeyCode::Left) {
            -1.0
        } else {
            0.0
        };

        state.target_velocity = Tensor::from_floats([target_vel], &device);

        let action = if model_loaded {
            let obs = physics.get_observation(&state);
            let (mean, _, _) = model.forward(obs);
            mean.clamp(-1.0, 1.0)
        } else {
            Tensor::zeros([1, 4], &device)
        };

        state = physics.step(state, action.clone());

        draw_simulation(&state, &physics);

        next_frame().await;
    }
}

fn draw_simulation<B: Backend>(state: &PhysicsState<B>, physics: &BipedalWalkerPhysics) {
    let hull_x: f32 = state.hull_x.clone().into_scalar().elem();
    let hull_y: f32 = state.hull_y.clone().into_scalar().elem();
    let hull_angle: f32 = state.hull_angle.clone().into_scalar().elem();

    let screen_w = screen_width();
    let screen_h = screen_height();
    let scale = 50.0;

    let ground_y = screen_h * 0.8;
    let draw_x = (screen_w / 5.0) + hull_x * scale; // Camera fixed, agent moves
    let draw_y = ground_y - hull_y * scale;

    // Ground
    draw_line(0.0, ground_y, screen_w, ground_y, 2.0, BLACK);

    // Hull
    let half_w = 20.0;
    let half_h = 10.0;
    let cos_a = hull_angle.cos();
    let sin_a = hull_angle.sin();

    let transform =
        |x: f32, y: f32| vec2(draw_x + x * cos_a - y * sin_a, draw_y + x * sin_a + y * cos_a);

    let p1 = transform(-half_w, -half_h);
    let p2 = transform(half_w, -half_h);
    let p3 = transform(half_w, half_h);
    let p4 = transform(-half_w, half_h);

    draw_triangle(p1, p2, p3, BLUE);
    draw_triangle(p1, p3, p4, BLUE);

    let thigh_len = physics.leg_length * 0.5 * scale;
    let shank_len = physics.leg_length * 0.5 * scale;

    let draw_leg = |hip_angle: f32, knee_angle: f32, color: Color| {
        let hip_x = draw_x;
        let hip_y = draw_y;

        let abs_hip_angle = hull_angle + hip_angle;
        let thigh_end_x = hip_x + abs_hip_angle.sin() * thigh_len;
        let thigh_end_y = hip_y + abs_hip_angle.cos() * thigh_len;
        draw_line(hip_x, hip_y, thigh_end_x, thigh_end_y, 4.0, color);

        let knee_abs_angle = abs_hip_angle + knee_angle;
        let foot_x = thigh_end_x + knee_abs_angle.sin() * shank_len;
        let foot_y = thigh_end_y + knee_abs_angle.cos() * shank_len;
        draw_line(thigh_end_x, thigh_end_y, foot_x, foot_y, 4.0, color);
    };

    let l_hip: f32 = state.left_hip_angle.clone().into_scalar().elem();
    let l_knee: f32 = state.left_knee_angle.clone().into_scalar().elem();
    let r_hip: f32 = state.right_hip_angle.clone().into_scalar().elem();
    let r_knee: f32 = state.right_knee_angle.clone().into_scalar().elem();

    draw_leg(l_hip, l_knee, RED);
    draw_leg(r_hip, r_knee, GREEN);

    draw_text(&format!("X: {:.2}", hull_x), 20.0, 20.0, 20.0, BLACK);
}
