"""
文件: tests/test_buffer_ablation.py
用途: 验证 Replay Buffer 缩容实验配置

运行:
python -m unittest tests.test_buffer_ablation -v
"""

import unittest

from agents.ddqn import ReplayBuffer, DDQNAgent


class TestBufferAblation(unittest.TestCase):

    def test_buffer_capacity(self):
        """验证 Buffer 最大容量为 1000。"""

        buffer = ReplayBuffer(capacity=1000)

        self.assertEqual(buffer.buffer.maxlen, 1000)

    def test_fifo_eviction(self):
        """验证容量满后自动淘汰最旧经验。"""

        buffer = ReplayBuffer(capacity=3)

        for i in range(4):
            buffer.push(
                s=[i],
                a=0,
                r=float(i),
                s2=[i + 1],
                done=False,
            )

        self.assertEqual(len(buffer), 3)

        # 第 0 条经验应该已经被淘汰
        states = [item[0][0] for item in buffer.buffer]

        self.assertEqual(states, [1, 2, 3])

    def test_agent_small_buffer(self):
        """验证 DDQNAgent 能正确创建 1000 容量的 Buffer。"""

        agent = DDQNAgent(replay_capacity=1000)

        self.assertEqual(len(agent.buf), 0)
        self.assertEqual(agent.buf.buffer.maxlen, 1000)

    def test_agent_default_capacity(self):
        """验证默认容量仍然是 50000。"""

        agent = DDQNAgent()

        self.assertEqual(
            agent.buf.buffer.maxlen,
            50000,
        )

    def test_invalid_capacity(self):
        """验证容量不能小于 warmup。"""

        with self.assertRaises(ValueError):
            DDQNAgent(
                replay_capacity=999,
                warmup=1000,
            )


if __name__ == "__main__":
    unittest.main()
