"""测试 YAML 配置加载与 epsilon 配置化。"""

from config import load_config
from train import epsilon_schedule


def test_load_base_config():
    config = load_config()

    assert config["environment"]["size"] == 8
    assert config["environment"]["max_steps"] == 200
    assert config["agent"]["gamma"] == 0.95
    assert config["agent"]["learning_rate"] == 0.001
    assert config["agent"]["batch_size"] == 64
    assert config["agent"]["warmup"] == 1000
    assert config["agent"]["replay_capacity"] == 50000
    assert config["training"]["target_sync_interval"] == 500


def test_epsilon_schedule_uses_custom_config():
    assert abs(
        epsilon_schedule(0, start=1.0, end=0.1, decay_steps=1000) - 1.0
    ) < 1e-9
    assert abs(
        epsilon_schedule(500, start=1.0, end=0.1, decay_steps=1000) - 0.55
    ) < 1e-9
    assert abs(
        epsilon_schedule(2000, start=1.0, end=0.1, decay_steps=1000) - 0.1
    ) < 1e-9


def test_train_receives_values_from_config(tmp_path, monkeypatch):
    from train import train

    config = load_config()
    config["environment"]["max_steps"] = 1
    config["agent"].update(
        {
            "gamma": 0.9,
            "learning_rate": 0.002,
            "batch_size": 32,
            "warmup": 64,
            "replay_capacity": 128,
        }
    )

    monkeypatch.chdir(tmp_path)
    agent = train(
        n_episodes=1,
        seed=321,
        algorithm="ddqn",
        run_tag="config_test",
        config=config,
    )

    assert agent.gamma == 0.9
    assert agent.batch_size == 32
    assert agent.warmup == 64
    assert agent.buf.buffer.maxlen == 128
    assert agent.optimizer.param_groups[0]["lr"] == 0.002
