use burn::backend::libtorch::{LibTorch, LibTorchDevice};
use burn::prelude::*;
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use macroquad::prelude::*;
use shared::model::ActorCritic;
use shared::physics::{CartPolePhysics, PhysicsState};

fn window_conf() -> Conf {
    Conf {
        window_title: "Cartpole".to_owned(),
        sample_count: 4,
        high_dpi: true,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let device = LibTorchDevice::Mps;
    let physics = CartPolePhysics::default();

    let mut state = PhysicsState {
        cart_position: Tensor::zeros([1], &device),
        cart_velocity: Tensor::zeros([1], &device),
        pole_angle: Tensor::from_data([0.1], &device),
        pole_angular_velocity: Tensor::zeros([1], &device),
        time: Tensor::zeros([1], &device),
    };

    let initial_obs = physics.get_observation(&state);
    let input_dim = initial_obs.dims()[1];
    let mut model = ActorCritic::<LibTorch>::new(input_dim, 1, &device);
    let recorder = BinFileRecorder::<FullPrecisionSettings>::default();

    match model.clone().load_file("model", &recorder, &device) {
        Ok(loaded_model) => {
            model = loaded_model;
            println!("Model loaded successfully.");
        }
        Err(e) => println!("Model file not found at model.bin. Using random model. Error: {:?}", e),
    }

    loop {
        clear_background(WHITE);

        let mut manual_force = 0.0;
        if is_key_down(KeyCode::Left) {
            manual_force = -0.1;
        } else if is_key_down(KeyCode::Right) {
            manual_force = 0.1;
        }

        let action = if manual_force != 0.0 {
            Tensor::<LibTorch, 1>::from_data([manual_force], &device)
        } else {
            let obs = physics.get_observation(&state);
            let (mean, _, _) = model.forward(obs);
            mean.squeeze_dim(0)
        };

        state = physics.step(state, action);

        draw_simulation(&state, physics.pole_length);

        next_frame().await;
    }
}

fn draw_simulation<B: Backend>(state: &PhysicsState<B>, pole_length: f32) {
    let x: f32 = state.cart_position.clone().into_scalar().elem();
    let theta: f32 = state.pole_angle.clone().into_scalar().elem();

    let screen_w = screen_width();
    let screen_h = screen_height();
    let scale = 100.0;
    let cart_x = screen_w / 2.0 + x * scale;
    let cart_y = screen_h * 0.7;

    draw_line(0.0, cart_y, screen_w, cart_y, 2.0, BLACK);

    draw_rectangle(cart_x - 30.0, cart_y - 15.0, 60.0, 30.0, BLUE);

    let visual_pole_length = pole_length * scale;
    let pole_x = cart_x + theta.sin() * visual_pole_length;
    let pole_y = cart_y - theta.cos() * visual_pole_length;
    draw_line(cart_x, cart_y, pole_x, pole_y, 6.0, RED);
}
