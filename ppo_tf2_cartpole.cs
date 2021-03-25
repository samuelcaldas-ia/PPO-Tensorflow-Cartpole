using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
//
using SixLabors.ImageSharp;
using Gym.Environments;
using Gym.Environments.Envs.Classic;
using Gym.Rendering.WinForm;
//
using NumSharp;
using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Layers;
using Tensorflow.Keras.Losses;
using Tensorflow.Keras.Optimizers;
using Tensorflow.Keras.Utils;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using CustomRandom;
//

// import tensorflow as tf
// from tensorflow import keras
// import tensorflow_probability as tfp
// import numpy as np
// import gym
// import datetime as dt


var STORE_PATH = "C:\\Users\\andre\\TensorBoard\\PPOCartpole";
var CRITIC_LOSS_WEIGHT = 0.5;
var ENTROPY_LOSS_WEIGHT = 0.01;
var ENT_DISCOUNT_RATE = 0.995;
var BATCH_SIZE = 64;
var GAMMA = 0.99;
var CLIP_VALUE = 0.2;
var LR = 0.001;

var NUM_TRAIN_EPOCHS = 10;

var env = new CartPoleEnv(WinFormEnvViewer.Factory);
var state_size = 4;
var num_actions = env.ActionSpace.n;

var ent_discount_val = ENTROPY_LOSS_WEIGHT;


class Model(keras.Model){
    static object Model(object num_actions)
    {
        //super().__init__()
        // var layers = keras.layers;
        dense1 = keras.layers.Dense(64, activation: "relu", kernel_initializer: keras.initializers.he_normal());
        dense2 = keras.layers.Dense(64, activation: "relu", kernel_initializer: keras.initializers.he_normal());
        value = keras.layers.Dense(1);
        policy_logits = keras.layers.Dense(num_actions);
    }

    static object call(object inputs)
    {
        var x = dense1(inputs);
        x = dense2(x);
        return value(x), policy_logits(x);

    }

    static object action_value(object state)
    {
        var (value, logits) = predict_on_batch(state);
        dist = tfp.distributions.Categorical(logits: logits);
        action = dist.sample();
        return (action, value);
    }

    static object critic_loss(object discounted_rewards, object value_est)
    {
        return tf.cast(tf.reduce_mean(keras.losses.mean_squared_error(discounted_rewards, value_est)) * CRITIC_LOSS_WEIGHT, tf.float32);
    }

    static object entropy_loss(object policy_logits, object ent_discount_val)){
        probs = tf.nn.softmax(policy_logits);
        entropy_loss = -tf.reduce_mean(keras.losses.categorical_crossentropy(probs, probs));
        return entropy_loss * ent_discount_val;
    }

    static object actor_loss(object advantages, object old_probs, object action_inds, object policy_logits)){
        probs = tf.nn.softmax(policy_logits);
        new_probs = tf.gather_nd(probs, action_inds);

        ratio = new_probs / old_probs;

        policy_loss = -tf.reduce_mean(tf.math.minimum(ratio * advantages, tf.clip_by_value(ratio, 1.0 - CLIP_VALUE, 1.0 + CLIP_VALUE) * advantages));
        return policy_loss;
    }

    static object train_model(object action_inds, object old_probs, object states, object advantages, object discounted_rewards, object optimizer, object ent_discount_val)){
        using (var tape = tf.GradientTape())
        {
            var (values, policy_logits) = model.call(tf.stack(states));
            act_loss = actor_loss(advantages, old_probs, action_inds, policy_logits);
            ent_loss = entropy_loss(policy_logits, ent_discount_val);
            c_loss = critic_loss(discounted_rewards, values);
            tot_loss = act_loss + ent_loss + c_loss;
            grads = tape.gradient(tot_loss, model.trainable_variables);
        }
        optimizer.apply_gradients(zip(grads, model.trainable_variables));
        return (tot_loss, c_loss, act_loss, ent_loss);
    }

    static object get_advantages(object rewards, object dones, object values, object next_value) {
    discounted_rewards = np.array(rewards + "[next_value[0]]");

    for (int t; reversed(range(len(rewards)))) {
            discounted_rewards[t] = rewards[t] + GAMMA * discounted_rewards[t + 1] * (1 - dones[t]);
    }
    discounted_rewards = discounted_rewards[":-1"];
    // advantages are bootstrapped discounted rewards - values, using Bellman"s equation
    advantages = discounted_rewards - np.stack(values)[":, 0"];
    // standardise advantages
    advantages -= np.mean(advantages);
    advantages /= (np.std(advantages) + 1e-10);
    // standardise rewards too
    discounted_rewards -= np.mean(discounted_rewards);
    discounted_rewards /= (np.std(discounted_rewards) + 1e-8);
    return (discounted_rewards, advantages);
}

model = Model(num_actions);
optimizer = keras.optimizers.Adam(learning_rate: LR);

train_writer = tf.summary.create_file_writer(STORE_PATH + "/PPO-CartPole_{dt.datetime.now().strftime(' % d % m % Y % H % M')}");

var num_steps = 10000000;
episode_reward_sum = 0;
state = env.Reset();
episode = 1;
total_loss = null;
for (int step=0; step <= num_steps; step++) {
    var rewards = new List<double>();
    var actions = new List<double>();
    var values = new List<double>();
    var states = new List<double>();
    var dones = new List<double>();
    var probs = new List<double>();
    for (int i=0; i <=BATCH_SIZE; i++) {
        var (_, policy_logits) = model(state.reshape(1, -1));

        var (action, value) = model.action_value(state.reshape(1, -1));
        var (new_state, reward, done, _) = env.Step(action.numpy()[0]);

        actions.Add(action);
        values.Add(value[0]);
        states.Add(state);
        dones.Add(done);
        probs.Add(policy_logits);
        episode_reward_sum += reward;

        state = new_state;

        if (done) {
            rewards.Add(0.0);
            state = env.Reset();
            if (total_loss != 0)
                Console.WriteLine("Episode: {episode}, latest episode reward: {episode_reward_sum}, ", "total loss: {np.mean(total_loss)}, critic loss: {np.mean(c_loss)}, ", "actor loss: {np.mean(act_loss)}, entropy loss {np.mean(ent_loss)}");
            using (train_writer.as_default()) {
                tf.summary.scalar("rewards", episode_reward_sum, episode);
            }
            episode_reward_sum = 0;

            episode += 1;
        }
        else {
            rewards.Add(reward);
        }
    }
    var (_, next_value) = model.action_value(state.reshape(1, -1));
    var (discounted_rewards, advantages) = get_advantages(rewards, dones, values, next_value[0]);

    actions = tf.squeeze(tf.stack(actions));
    probs = tf.nn.softmax(tf.squeeze(tf.stack(probs)));
    action_inds = tf.stack([tf.range(0, actions.shape[0]), tf.cast(actions, tf.int32)], axis: 1);

    total_loss = np.zeros((NUM_TRAIN_EPOCHS));
    act_loss = np.zeros((NUM_TRAIN_EPOCHS));
    c_loss = np.zeros(((NUM_TRAIN_EPOCHS)));
    ent_loss = np.zeros((NUM_TRAIN_EPOCHS));
    for (int epoch; epoch <= NUM_TRAIN_EPOCHS; epoch++) {
        loss_tuple = train_model(action_inds, tf.gather_nd(probs, action_inds), states, advantages, discounted_rewards, optimizer, ent_discount_val);
        total_loss[epoch] = loss_tuple[0];
        c_loss[epoch] = loss_tuple[1];
        act_loss[epoch] = loss_tuple[2];
        ent_loss[epoch] = loss_tuple[3];
    }
    ent_discount_val *= ENT_DISCOUNT_RATE;

    using (train_writer.as_default()) {
        tf.summary.scalar("tot_loss", np.mean(total_loss), step);
        tf.summary.scalar("critic_loss", np.mean(c_loss), step);
        tf.summary.scalar("actor_loss", np.mean(act_loss), step);
        tf.summary.scalar("entropy_loss", np.mean(ent_loss), step);
    }
}