use crate::distribution::DiagonalGaussian;
use crate::env::{EnvironmentState, TrainingEnv, TrainingStep};
use burn::{
    config::Config,
    grad_clipping::GradientClippingConfig,
    module::AutodiffModule,
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::*,
    record::{BinFileRecorder, FullPrecisionSettings},
    tensor::{backend::AutodiffBackend, Distribution, Int},
};
use shared::model::ActorCritic;
use shared::physics::PhysicsState;
use std::time::Instant;

#[derive(Config, Debug)]
pub struct ProximalPolicyOptimizationConfig {
    #[config(default = 2048)]
    pub environments_count: usize,
    #[config(default = 128)]
    pub rollout_length: usize,
    #[config(default = 0.99)]
    pub gamma: f32,
    #[config(default = 0.95)]
    pub generalized_advantage_estimation_lambda: f32,
    #[config(default = 0.2)]
    pub proximal_policy_optimization_clip: f32,
    #[config(default = 0.02)]
    pub entropy_coefficient: f32,
    #[config(default = 0.5)]
    pub value_coefficient: f32,
    #[config(default = 3e-4)]
    pub learning_rate: f64,
    #[config(default = 4)]
    pub update_epochs: usize,
    #[config(default = 64)]
    pub minibatches: usize,
    #[config(default = 0.5)]
    pub max_grad_norm: f32,
    #[config(default = 10.0)]
    pub max_time: f32,
}

pub fn train<B: AutodiffBackend>(
    device: B::Device,
    config: ProximalPolicyOptimizationConfig,
    iterations: usize,
) {
    let ProximalPolicyOptimizationConfig {
        environments_count,
        rollout_length,
        learning_rate,
        update_epochs,
        minibatches,
        max_grad_norm,
        max_time,
        ..
    } = config;
    let environment = TrainingEnv::<B::InnerBackend>::new(&device, max_time);

    let (observation_inner, state_inner) = environment.reset(environments_count, &device);
    let mut observation = Tensor::<B, 2>::from_inner(observation_inner);
    let mut state = EnvironmentState::from_inner(state_inner);
    let mut current_episode_rewards =
        Tensor::<B::InnerBackend, 1>::zeros([environments_count], &device);

    let input_dimension = observation.dims()[1];
    let action_dimension = environment.action_dim();

    let mut model = ActorCritic::<B>::new(input_dimension, action_dimension, &device);
    let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
    if std::path::Path::new("model.bin").exists() {
        match model.clone().load_file("model", &recorder, &device) {
            Ok(loaded_model) => {
                model = loaded_model;
                println!("Successfully loaded existing model from model.bin");
            }
            Err(e) => println!("Failed to load existing model: {:?}", e),
        }
    }

    let mut optimizer = AdamConfig::new()
        .with_grad_clipping(Some(GradientClippingConfig::Norm(max_grad_norm)))
        .init();

    println!("Starting training for {} iterations...", iterations);
    println!("Total steps per iteration: {}", environments_count * rollout_length);

    let start_time = Instant::now();

    for i in 0..iterations {
        let iteration_start = Instant::now();

        let rollout = collect_rollout(
            &model,
            &environment,
            &mut observation,
            &mut state,
            &mut current_episode_rewards,
            &config,
            &device,
        );

        let model_valid = model.clone().valid();
        let (_, _, last_values) = model_valid.forward(observation.clone().inner());
        let next_value = last_values.squeeze_dim::<1>(1);

        let (all_returns, all_advantages) = compute_generalized_advantage_estimation(
            &rollout.rewards,
            &rollout.dones,
            &rollout.values,
            next_value,
            &config,
        );

        let observation_tensor = Tensor::<B, 2>::from_inner(Tensor::cat(rollout.observations, 0));
        let actions_tensor = Tensor::<B, 2>::from_inner(Tensor::cat(rollout.actions, 0));
        let log_probabilities_tensor =
            Tensor::<B, 1>::from_inner(Tensor::cat(rollout.log_probabilities, 0));
        let returns_tensor = Tensor::<B, 1>::from_inner(Tensor::cat(all_returns, 0));
        let advantages_tensor = Tensor::<B, 1>::from_inner(Tensor::cat(all_advantages, 0));
        let old_values_tensor = Tensor::<B, 1>::from_inner(Tensor::cat(rollout.values, 0));

        let advantages_mean = advantages_tensor.clone().mean();
        let advantages_standard_deviation =
            advantages_tensor.clone().var(0).sqrt().add_scalar(1e-8);
        let advantages_tensor =
            (advantages_tensor - advantages_mean) / advantages_standard_deviation;

        let number_of_samples = environments_count * rollout_length;
        let batch_size = number_of_samples / minibatches;

        for _ in 0..update_epochs {
            let indices_tensor =
                Tensor::<B, 1>::random([number_of_samples], Distribution::Default, &device)
                    .argsort(0);

            let shuffled_observations =
                observation_tensor.clone().select(0, indices_tensor.clone());
            let shuffled_actions = actions_tensor.clone().select(0, indices_tensor.clone());
            let shuffled_log_probabilities =
                log_probabilities_tensor.clone().select(0, indices_tensor.clone());
            let shuffled_returns = returns_tensor.clone().select(0, indices_tensor.clone());
            let shuffled_advantages = advantages_tensor.clone().select(0, indices_tensor.clone());
            let shuffled_old_values = old_values_tensor.clone().select(0, indices_tensor.clone());

            for batch_start in (0..number_of_samples).step_by(batch_size) {
                let batch_end = std::cmp::min(batch_start + batch_size, number_of_samples);

                let batch_observations =
                    shuffled_observations.clone().slice([batch_start..batch_end]);
                let batch_actions = shuffled_actions.clone().slice([batch_start..batch_end]);
                let batch_old_log_probabilities =
                    shuffled_log_probabilities.clone().slice([batch_start..batch_end]);
                let batch_returns = shuffled_returns.clone().slice([batch_start..batch_end]);
                let batch_advantages = shuffled_advantages.clone().slice([batch_start..batch_end]);
                let batch_old_values = shuffled_old_values.clone().slice([batch_start..batch_end]);

                let loss = compute_proximal_policy_optimization_loss(
                    &model,
                    batch_observations,
                    batch_actions,
                    batch_old_log_probabilities,
                    batch_returns,
                    batch_advantages,
                    batch_old_values,
                    &config,
                );

                let grads = loss.backward();
                let grads = GradientsParams::from_grads(grads, &model);
                model = optimizer.step(learning_rate, model, grads);
            }
        }

        let iteration_end = Instant::now();
        let duration = iteration_end.duration_since(iteration_start).as_secs_f64();
        let steps_per_second = number_of_samples as f64 / duration;

        if i % 10 == 0 || i == iterations - 1 {
            let total_episodes = rollout.total_episodes;
            let fallen_pct = if total_episodes > 0 {
                (rollout.total_fallen as f64 / total_episodes as f64) * 100.0
            } else {
                0.0
            };

            let avg_episode_reward =
                if total_episodes > 0 { rollout.total_reward / total_episodes as f32 } else { 0.0 };

            println!(
                "Iter {:4} | Reward: {:7.2} | Fallen: {:6.2}% | SPS: {:8.0}",
                i, avg_episode_reward, fallen_pct, steps_per_second
            );

            if i % 10 == 0 {
                let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
                model
                    .clone()
                    .save_file("model", &recorder)
                    .expect("Should be able to save the model");
            }
        }
    }

    let total_time = start_time.elapsed().as_secs_f64();
    println!("Training finished in {:.2} seconds.", total_time);

    let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
    model.save_file("model", &recorder).expect("Should be able to save the model");
    println!("Model saved to model.bin");
}

fn compute_proximal_policy_optimization_loss<B: AutodiffBackend>(
    model: &ActorCritic<B>,
    observation: Tensor<B, 2>,
    actions: Tensor<B, 2>,
    old_log_probabilities: Tensor<B, 1>,
    returns: Tensor<B, 1>,
    advantages: Tensor<B, 1>,
    old_values: Tensor<B, 1>,
    config: &ProximalPolicyOptimizationConfig,
) -> Tensor<B, 1> {
    let ProximalPolicyOptimizationConfig {
        proximal_policy_optimization_clip,
        entropy_coefficient,
        value_coefficient,
        ..
    } = *config;
    let (mean, log_std, values) = model.forward(observation);
    let values = values.squeeze_dim::<1>(1);

    let distribution = DiagonalGaussian::new(mean, log_std.clamp(-5.0, 2.0));
    let log_probabilities = distribution.log_probability(actions);
    let entropy = distribution.entropy();

    let ratio = (log_probabilities - old_log_probabilities).exp();
    let surrogate_1 = ratio.clone() * advantages.clone();
    let surrogate_2 = ratio
        .clamp(1.0 - proximal_policy_optimization_clip, 1.0 + proximal_policy_optimization_clip)
        * advantages;
    let policy_loss = -surrogate_1.min_pair(surrogate_2).mean();

    let value_loss_unclipped = (values.clone() - returns.clone()).powf_scalar(2.0);
    let value_clipped = old_values.clone()
        + (values - old_values.clone())
            .clamp(-proximal_policy_optimization_clip, proximal_policy_optimization_clip);
    let value_loss_clipped = (value_clipped - returns).powf_scalar(2.0);
    let value_loss = value_loss_unclipped.max_pair(value_loss_clipped).mean().mul_scalar(0.5);

    policy_loss + value_loss.mul_scalar(value_coefficient) - entropy.mul_scalar(entropy_coefficient)
}

fn collect_rollout<B: AutodiffBackend>(
    model: &ActorCritic<B>,
    environment: &TrainingEnv<B::InnerBackend>,
    observation_outer: &mut Tensor<B, 2>,
    state_outer: &mut PhysicsState<B>,
    current_episode_rewards: &mut Tensor<B::InnerBackend, 1>,
    config: &ProximalPolicyOptimizationConfig,
    device: &B::Device,
) -> Rollout<B::InnerBackend> {
    let ProximalPolicyOptimizationConfig { environments_count, rollout_length, .. } = *config;
    let model_valid = model.clone().valid();
    let action_dim = environment.action_dim();

    let action_noise = Tensor::<B::InnerBackend, 3>::random(
        [rollout_length, environments_count, action_dim],
        Distribution::Normal(0.0, 1.0),
        device,
    );

    let (reset_observation, reset_state) = environment.reset(environments_count, device);

    let mut observation = observation_outer.clone().inner();
    let mut state = EnvironmentState::inner(state_outer.clone());

    let mut rollout = Rollout::<B::InnerBackend>::new(rollout_length);

    for step in 0..rollout_length {
        let noise = action_noise.clone().slice([step..step + 1]).squeeze_dim::<2>(0);

        let (mean, log_std, value) = model_valid.forward(observation.clone());
        let distribution = DiagonalGaussian::new(mean, log_std);

        let action = distribution.sample(noise);
        let log_probability = distribution.log_probability(action.clone());
        let value = value.squeeze_dim(1);

        let step = environment.step(state, action.clone());
        let TrainingStep { reward, done, is_fallen, .. } = step;

        // Update cumulative reward
        *current_episode_rewards = current_episode_rewards.clone() + reward.clone();

        let is_done = done.clone().bool();

        rollout.push(
            observation.clone(),
            action,
            log_probability,
            value,
            reward,
            done.clone(),
            is_fallen.clone(),
            current_episode_rewards.clone(),
        );

        *current_episode_rewards = current_episode_rewards
            .clone()
            .mask_where(is_done.clone(), Tensor::zeros_like(current_episode_rewards));

        observation = step.observation;
        state = step.state;

        let dims = observation.dims();
        observation = observation.mask_where(
            is_done.clone().unsqueeze_dim::<2>(1).expand(dims),
            reset_observation.clone(),
        );
        state = state.mask_where(is_done, reset_state.clone());
    }

    // Aggregated metrics on GPU to avoid large data transfer
    let all_dones = Tensor::cat(rollout.dones.clone(), 0);
    let all_rewards = Tensor::cat(rollout.cumulative_rewards.clone(), 0);
    let all_is_fallens = Tensor::cat(rollout.is_fallens.clone(), 0);

    let total_episodes_t = all_dones.clone().sum().float().reshape([1]);
    let total_fallen_t = (all_dones.clone() * all_is_fallens).sum().float().reshape([1]);
    let total_reward_t = (all_dones.float() * all_rewards).sum().reshape([1]);

    let metrics =
        Tensor::cat(vec![total_episodes_t, total_fallen_t, total_reward_t], 0).into_data();
    let metrics_slice = metrics.as_slice::<f32>().unwrap();

    let total_episodes = metrics_slice[0] as usize;

    if total_episodes > 0 {
        rollout.total_episodes = total_episodes;
        rollout.total_fallen = metrics_slice[1] as usize;
        rollout.total_reward = metrics_slice[2];
    }

    *observation_outer = Tensor::from_inner(observation.clone());
    *state_outer = EnvironmentState::from_inner(state);

    rollout
}

struct Rollout<B: Backend> {
    observations: Vec<Tensor<B, 2>>,
    actions: Vec<Tensor<B, 2>>,
    log_probabilities: Vec<Tensor<B, 1>>,
    values: Vec<Tensor<B, 1>>,
    rewards: Vec<Tensor<B, 1>>,
    dones: Vec<Tensor<B, 1, Int>>,
    is_fallens: Vec<Tensor<B, 1, Int>>,
    cumulative_rewards: Vec<Tensor<B, 1>>,
    total_fallen: usize,
    total_reward: f32,
    total_episodes: usize,
}

impl<B: Backend> Rollout<B> {
    fn new(capacity: usize) -> Self {
        Self {
            observations: Vec::with_capacity(capacity),
            actions: Vec::with_capacity(capacity),
            log_probabilities: Vec::with_capacity(capacity),
            values: Vec::with_capacity(capacity),
            rewards: Vec::with_capacity(capacity),
            dones: Vec::with_capacity(capacity),
            is_fallens: Vec::with_capacity(capacity),
            cumulative_rewards: Vec::with_capacity(capacity),
            total_fallen: 0,
            total_reward: 0.0,
            total_episodes: 0,
        }
    }

    fn push(
        &mut self,
        observation: Tensor<B, 2>,
        action: Tensor<B, 2>,
        log_probability: Tensor<B, 1>,
        value: Tensor<B, 1>,
        reward: Tensor<B, 1>,
        done: Tensor<B, 1, Int>,
        is_fallen: Tensor<B, 1, Int>,
        cumulative_reward: Tensor<B, 1>,
    ) {
        self.observations.push(observation);
        self.actions.push(action);
        self.log_probabilities.push(log_probability);
        self.values.push(value);
        self.rewards.push(reward);
        self.dones.push(done);
        self.is_fallens.push(is_fallen);
        self.cumulative_rewards.push(cumulative_reward);
    }
}

fn compute_generalized_advantage_estimation<B: Backend>(
    rewards: &[Tensor<B, 1>],
    dones: &[Tensor<B, 1, Int>],
    values: &[Tensor<B, 1>],
    next_value: Tensor<B, 1>,
    config: &ProximalPolicyOptimizationConfig,
) -> (Vec<Tensor<B, 1>>, Vec<Tensor<B, 1>>) {
    let ProximalPolicyOptimizationConfig { gamma, generalized_advantage_estimation_lambda, .. } =
        *config;
    let rollout_length = rewards.len();
    let environments_count = next_value.dims()[0];
    let device = next_value.device();

    // Stack everything into [T, B]
    let rewards_t =
        Tensor::cat(rewards.iter().map(|t| t.clone().unsqueeze_dim::<2>(0)).collect(), 0);
    let values_t = Tensor::cat(values.iter().map(|t| t.clone().unsqueeze_dim::<2>(0)).collect(), 0);
    let dones_t =
        Tensor::cat(dones.iter().map(|t| t.clone().float().unsqueeze_dim::<2>(0)).collect(), 0);
    let done_masks_t = dones_t.clone().equal_elem(0.0).float();

    // Calculate deltas: delta_t = r_t + gamma * (1-d_t) * V_{t+1} - V_t
    let next_values = Tensor::cat(
        vec![values_t.clone().slice([1..rollout_length]), next_value.unsqueeze_dim::<2>(0)],
        0,
    );
    let deltas = rewards_t + next_values * done_masks_t * gamma - values_t.clone();

    // Calculate advantages using matrix multiplication: A = M * deltas
    // M[i, k] = (gamma * lambda)^(k-i) * prod_{j=i}^{k-1} (1-d_j)

    let indices = Tensor::<B, 1, Int>::arange(0..rollout_length as i64, &device).float();
    let i = indices.clone().reshape([rollout_length, 1]);
    let k = indices.reshape([1, rollout_length]);
    let dist = k - i.clone(); // [T, T], dist[i, k] = k - i
    let triu_mask = dist.clone().greater_equal_elem(0.0).float();

    let decay = (gamma * generalized_advantage_estimation_lambda) as f64;
    let dist_safe = dist.clone().mask_fill(dist.clone().lower_elem(0.0), 0.0);
    let powers = (dist_safe * decay.max(1e-10).ln()).exp() * triu_mask;

    let c =
        Tensor::cat(vec![Tensor::zeros([1, environments_count], &device), dones_t.cumsum(0)], 0)
            .swap_dims(0, 1); // [B, T+1]

    let c_for_k = c.clone().slice([0..environments_count, 0..rollout_length]).reshape([
        environments_count,
        1,
        rollout_length,
    ]); // [B, 1, T]
    let c_for_i = c.slice([0..environments_count, 0..rollout_length]).reshape([
        environments_count,
        rollout_length,
        1,
    ]); // [B, T, 1]
    let segment_mask = (c_for_k - c_for_i).equal_elem(0.0).float(); // [B, T, T]

    let m = powers.reshape([1, rollout_length, rollout_length]) * segment_mask; // [B, T, T]

    let deltas_b = deltas.swap_dims(0, 1).reshape([environments_count, rollout_length, 1]); // [B, T, 1]
    let advantages_b = m.matmul(deltas_b).squeeze_dim::<2>(2); // [B, T]
    let advantages_t = advantages_b.swap_dims(0, 1); // [T, B]

    let returns_t = advantages_t.clone() + values_t;

    let returns = (0..rollout_length)
        .map(|i| returns_t.clone().slice([i..i + 1]).squeeze_dim::<1>(0))
        .collect();
    let advantages = (0..rollout_length)
        .map(|i| advantages_t.clone().slice([i..i + 1]).squeeze_dim::<1>(0))
        .collect();

    (returns, advantages)
}
