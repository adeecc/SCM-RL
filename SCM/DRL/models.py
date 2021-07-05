import os
from datetime import datetime
from ray.rllib.utils import try_import_tf, try_import_torch

import ray.rllib.agents.ddpg as ddpg
import ray.rllib.agents.a3c as a3c
from ray.tune.logger import pretty_print

from environment import SimpleSupplyChain

tf = try_import_tf()
torch = try_import_torch()


def train_ddpg():
    config = ddpg.DEFAULT_CONFIG.copy()
    config["log_level"] = "WARN"
    config["actor_hiddens"] = [32, 32]
    config["critic_hiddens"] = [32, 32]
    config["gamma"] = 0.95
    config["timesteps_per_iteration"] = 2000
    config["target_network_update_freq"] = 5
    config["buffer_size"] = 1024
    config["num_gpus"] = 1
    # config["num_workers"] = 2
    config["framework"] = "tfe"
    config["simple_optimizer"] = True

    trainer = ddpg.DDPGTrainer(config=config, env=SimpleSupplyChain)
    for i in range(500):
        result = trainer.train()
        print(pretty_print(result))
        checkpoint = trainer.save()
        print("Checkpoint saved at", checkpoint)

        # Checkpoint saved at /home/adeecc/ray_results/DDPG_SimpleSupplyChain_2021-06-27_22-59-01ktnytcmt/checkpoint_000500/checkpoint-500


def train_td3(normalize_actions=True):
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
    config["framework"] = "tfe"
    config["simple_optimizer"] = True

    env = SimpleSupplyChain
    checkpoint_dir_prefix = f"/home/adeecc/ray_results/TD3_SimpleSupplyChain_{datetime.now():%d-%m-%y_%H-%M}"

    trainer = ddpg.DDPGTrainer(config=config, env=env)
    for i in range(500):
        result = trainer.train()
        print(pretty_print(result))
        checkpoint = trainer.save(checkpoint_dir_prefix)
        print(f"Checkpoint saved at: {checkpoint}")


def train_a3c(normalize_actions=True):
    config = a3c.DEFAULT_CONFIG.copy()
    config["lr"] = 3e-4

    # From commom config
    config["gamma"] = 0.95
    config["normalize_actions"] = normalize_actions  # trial

    config["evaluation_interval"] = 50
    config["framework"] = "torch"

    checkpoint_dir_prefix = f"/home/adeecc/ray_results/A3C_SimpleSupplyChain_{datetime.now():%d-%m-%y_%H-%M}"

    trainer = a3c.A3CTrainer(config=config, env=SimpleSupplyChain)
    for i in range(500):
        result = trainer.train()
        print(pretty_print(result))
        checkpoint = trainer.save(checkpoint_dir_prefix)
        print(f"Checkpoint saved at: {checkpoint}")
