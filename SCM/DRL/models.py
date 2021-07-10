import os
from datetime import datetime
from ray.rllib.utils import try_import_tf, try_import_torch


import ray.rllib.agents.ddpg as ddpg
import ray.rllib.agents.a3c as a3c
from ray.tune.logger import pretty_print

import ray.cloudpickle as cloudpickle


from environment import SimpleSupplyChain

tf = try_import_tf()
torch = try_import_torch()


def train_ddpg(num_epochs=500, normalize_actions=True):
    config = ddpg.DEFAULT_CONFIG.copy()
    config["log_level"] = "INFO"
    config["actor_hiddens"] = [512, 512]
    config["critic_hiddens"] = [512, 512]
    config["gamma"] = 0.95

    config["normalize_actions"] = normalize_actions  # trial

    config["timesteps_per_iteration"] = 1000
    config["target_network_update_freq"] = 5
    config["buffer_size"] = 10000
    config["num_gpus"] = 1
    # config["num_workers"] = 2
    config["framework"] = "tfe"

    trainer = ddpg.DDPGTrainer(config=config, env=SimpleSupplyChain)
    for epoch in range(num_epochs):
        result = trainer.train()
        print(pretty_print(result))
        checkpoint = trainer.save()
        print("Checkpoint saved at", checkpoint)

    return checkpoint


def train_td3(num_epochs=500, normalize_actions=True):
    config = ddpg.DEFAULT_CONFIG.copy()
    config["twin_q"] = True  # TD3!
    config["smooth_target_policy"] = True
    config["evaluation_interval"] = 250
    config["actor_hiddens"] = [128, 128]
    config["critic_hiddens"] = [128, 128]

    config["normalize_actions"] = normalize_actions  # trial

    config["lr"] = 3e-4
    config["gamma"] = 0.95
    config["timesteps_per_iteration"] = 1000
    config["target_network_update_freq"] = 5
    config["buffer_size"] = 5000
    config["num_gpus"] = 1
    config["framework"] = "tf2"  # "tfe"
    # config["simple_optimizer"] = True

    env = SimpleSupplyChain

    trainer = ddpg.DDPGTrainer(config=config, env=env)
    for epoch in range(num_epochs):
        result = trainer.train()
        print(pretty_print(result))
        checkpoint = trainer.save()
        print(f"Checkpoint saved at: {checkpoint}")

    return checkpoint


def train_a3c(normalize_actions=True):
    config = a3c.DEFAULT_CONFIG.copy()
    config["lr"] = 3e-4

    # From commom config
    config["gamma"] = 0.95
    config["normalize_actions"] = normalize_actions  # trial

    config["evaluation_interval"] = 50
    config["framework"] = "torch"

    trainer = a3c.A3CTrainer(config=config, env=SimpleSupplyChain)
    for i in range(500):
        result = trainer.train()
        print(pretty_print(result))
        checkpoint = trainer.save()
        print(f"Checkpoint saved at: {checkpoint}")

    return checkpoint


def load_policy(log_dir, checkpoint_id):
    config_path = os.path.join(log_dir, "params.pkl")
    with open(config_path, "rb") as read_file:
        config = cloudpickle.load(read_file)

        print(config)

    params_path = os.path.join(
        log_dir, f"checkpoint_{checkpoint_id.zfill(6)}", f"checkpoint-{checkpoint_id}")
    print(params_path)
    trainer = ddpg.DDPGTrainer(config=config, env=SimpleSupplyChain)
    trainer.restore(params_path)
    return trainer.get_policy()
