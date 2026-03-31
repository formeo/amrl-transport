"""
Example: Run an RL training loop with the SimulatorTransport.

This demonstrates the full pipeline:
1. Create a simulated STM with atoms on a surface
2. Set up the TransportEnv (drop-in for SINGROUP's RealExpEnv)
3. Run a simple random-action loop (replace with SAC agent for real training)

No hardware needed — runs anywhere.

Usage:
    python examples/train_with_simulator.py
"""
import numpy as np

from amrl_transport.integration import TransportEnv
from amrl_transport.transport import SimulatorTransport


def main():
    # ── 1. Set up simulated STM ──────────────────────────────
    stm = SimulatorTransport(
        lattice_constant=0.288,  # Ag(111)
        atom_positions=np.array([[0.0, 0.0]]),  # one atom at origin
        noise_level=0.02,
        manipulation_success_rate=0.8,
        seed=42,
    )
    stm.connect()
    print(f"Connected to: {stm.name}")
    print(f"Initial atoms: {stm.atom_positions}")

    # ── 2. Create the RL environment ─────────────────────────
    template = np.zeros((16, 16))  # dummy template for simulator

    env = TransportEnv(
        transport=stm,
        step_nm=0.2,
        max_mvolt=20,
        max_pcurrent_to_mvolt_ratio=2850,
        goal_nm=1.5,
        template=template,
        current_jump=4,
        im_size_nm=5.0,
        offset_nm=np.array([0.0, 0.0]),
        manip_limit_nm=np.array([-4.0, 4.0, -4.0, 4.0]),
        pixel=64,
        template_max_y=30,
        scan_mV=100,
        max_len=10,
        load_weight="dummy.pth",
    )

    # ── 3. Run episodes ──────────────────────────────────────
    num_episodes = 5

    for ep in range(num_episodes):
        state, info = env.reset()
        print(f"\n{'='*50}")
        print(f"Episode {ep + 1}")
        print(f"  Start: {info['start_absolute_nm']}")
        print(f"  Goal:  {info['goal_absolute_nm']}")

        total_reward = 0
        for step in range(env.max_len):
            # Random action (replace with trained SAC agent)
            action = np.random.uniform(-1, 1, size=6)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            print(f"  Step {step + 1}: reward={reward:.3f}, done={done}")

            if done:
                break

        print(f"  Total reward: {total_reward:.3f}")
        dist = info.get("dist_destination", float("nan"))
        print(f"  Final distance to target: {dist:.3f} nm")

    # ── 4. With real SINGROUP SAC agent (if AMRL is installed) ─
    try:
        from AMRL import sac_agent
        print("\n\nSINGROUP AMRL package detected!")
        print("You can now use sac_agent with TransportEnv.")
        print("Just replace RealExpEnv with TransportEnv in training notebooks.")
    except ImportError:
        print("\n\nTip: Install AMRL for the real SAC agent:")
        print("  pip install git+https://github.com/SINGROUP/Atom_manipulation_with_RL.git")

    stm.disconnect()
    print("\nDone!")


if __name__ == "__main__":
    main()
