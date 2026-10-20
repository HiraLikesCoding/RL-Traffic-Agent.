
import random
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Smart Traffic Signal Agent",
    page_icon="🚦",
    layout="wide"
)

ACTIONS = [0, 1]
ACTION_NAMES = ["Green for Road A", "Green for Road B"]


def traffic_state(a, b):
    def bucket(n):
        if n <= 4:
            return 0
        if n <= 9:
            return 1
        return 2

    return bucket(a) * 3 + bucket(b)


def choose_action(q_table, state, epsilon):
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    return int(np.argmax(q_table[state]))


def step_traffic(a, b, action, arrival_a, arrival_b, capacity):
    served_a = min(a, capacity) if action == 0 else 0
    served_b = min(b, capacity) if action == 1 else 0

    next_a = max(0, a - served_a) + random.randint(0, arrival_a)
    next_b = max(0, b - served_b) + random.randint(0, arrival_b)

    reward = -(next_a + next_b) + 0.5 * (served_a + served_b)

    return min(next_a, 20), min(next_b, 20), reward, served_a + served_b


def train_agent(episodes, steps, arrival_a, arrival_b, capacity, alpha, gamma):
    q_table = np.zeros((9, 2))
    rewards_history = []

    random.seed(42)
    np.random.seed(42)

    for episode in range(episodes):
        a = random.randint(0, 10)
        b = random.randint(0, 10)
        total_reward = 0

        for _ in range(steps):
            state = traffic_state(a, b)
            epsilon = max(0.05, 1.0 - episode / max(1, episodes - 1))
            action = choose_action(q_table, state, epsilon)

            next_a, next_b, reward, _ = step_traffic(
                a, b, action, arrival_a, arrival_b, capacity
            )

            next_state = traffic_state(next_a, next_b)

            q_table[state, action] += alpha * (
                reward
                + gamma * np.max(q_table[next_state])
                - q_table[state, action]
            )

            a, b = next_a, next_b
            total_reward += reward

        rewards_history.append(total_reward)

    return q_table, rewards_history


def evaluate_policy(q_table, episodes, steps, arrival_a, arrival_b, capacity, learned=True):
    rows = []

    for episode in range(episodes):
        a, b = 5, 5
        total_wait = 0
        served_total = 0

        for t in range(steps):
            state = traffic_state(a, b)

            if learned:
                action = int(np.argmax(q_table[state]))
            else:
                action = t % 2

            next_a, next_b, _, served = step_traffic(
                a, b, action, arrival_a, arrival_b, capacity
            )

            total_wait += next_a + next_b
            served_total += served
            a, b = next_a, next_b

        rows.append({
            "Episode": episode + 1,
            "Total waiting vehicles": total_wait,
            "Vehicles served": served_total
        })

    return pd.DataFrame(rows)


st.title("🚦 Smart Traffic Signal Agent")

st.write(
    "A Reinforcement Learning MVP that uses Q-Learning "
    "to select which road receives a green signal."
)

st.info(
    "Educational simulation only. This application does not "
    "control real traffic lights."
)

with st.sidebar:
    st.header("Simulation settings")

    episodes = st.slider("Training episodes", 100, 3000, 1000, 100)
    steps = st.slider("Steps per episode", 20, 100, 40, 10)

    arrival_a = st.slider("Maximum arrivals on Road A", 0, 5, 3)
    arrival_b = st.slider("Maximum arrivals on Road B", 0, 5, 3)
    capacity = st.slider("Vehicles cleared per green step", 1, 10, 5)

    alpha = st.slider("Learning rate (alpha)", 0.05, 1.0, 0.2, 0.05)
    gamma = st.slider("Discount factor (gamma)", 0.0, 0.99, 0.9, 0.05)

    train_button = st.button(
        "Train Q-Learning Agent",
        type="primary",
        use_container_width=True
    )

if "q_table" not in st.session_state:
    st.session_state.q_table = None
    st.session_state.rewards = None
    st.session_state.settings = None

if train_button:
    with st.spinner("Training the agent..."):
        q_table, rewards = train_agent(
            episodes, steps, arrival_a, arrival_b,
            capacity, alpha, gamma
        )

    st.session_state.q_table = q_table
    st.session_state.rewards = rewards
    st.session_state.settings = (
        arrival_a, arrival_b, capacity, steps
    )

    st.success("Training completed!")

if st.session_state.q_table is None:
    st.subheader("How to start")
    st.write("1. Select your settings in the sidebar.")
    st.write("2. Click Train Q-Learning Agent.")
    st.write("3. Explore the learned decisions and test performance.")

else:
    q_table = st.session_state.q_table
    rewards = st.session_state.rewards
    saved_a, saved_b, saved_capacity, saved_steps = (
        st.session_state.settings
    )

    st.subheader("Learned agent")

    col1, col2, col3 = st.columns(3)

    col1.metric("Training episodes", f"{len(rewards):,}")
    col2.metric("Average reward (last 50)", f"{np.mean(rewards[-50:]):.1f}")
    col3.metric("Q-table states", q_table.shape[0])

    st.subheader("Training progress")

    reward_df = pd.DataFrame({
        "Episode": np.arange(1, len(rewards) + 1),
        "Total reward": rewards
    }).set_index("Episode")

    st.line_chart(reward_df)

    st.subheader("Try a traffic situation")

    col1, col2 = st.columns(2)

    with col1:
        queue_a = st.slider(
            "Vehicles waiting on Road A", 0, 20, 12
        )

    with col2:
        queue_b = st.slider(
            "Vehicles waiting on Road B", 0, 20, 4
        )

    state = traffic_state(queue_a, queue_b)
    action = int(np.argmax(q_table[state]))

    st.success(f"Agent decision: {ACTION_NAMES[action]}")

    st.write(
        f"Q-values — Road A: {q_table[state, 0]:.2f}, "
        f"Road B: {q_table[state, 1]:.2f}"
    )

    st.caption(
        "Queue counts are grouped into small, medium and large categories."
    )

    st.subheader("Compare traffic policies")

    if st.button("Run comparison", use_container_width=True):
        learned = evaluate_policy(
            q_table, 30, saved_steps, saved_a,
            saved_b, saved_capacity, True
        )

        fixed = evaluate_policy(
            q_table, 30, saved_steps, saved_a,
            saved_b, saved_capacity, False
        )

        comparison = pd.DataFrame({
            "Policy": ["Q-Learning agent", "Fixed-time baseline"],
            "Average waiting vehicles": [
                learned["Total waiting vehicles"].mean(),
                fixed["Total waiting vehicles"].mean()
            ],
            "Average vehicles served": [
                learned["Vehicles served"].mean(),
                fixed["Vehicles served"].mean()
            ]
        })

        st.dataframe(
            comparison, use_container_width=True, hide_index=True
        )

        st.bar_chart(comparison.set_index("Policy"))

        st.caption(
            "Results are from a simplified randomized simulation. "
            "They do not establish real-world traffic benefits."
        )

    st.subheader("Learned Q-table")

    labels = [
        "A small / B small",
        "A small / B medium",
        "A small / B large",
        "A medium / B small",
        "A medium / B medium",
        "A medium / B large",
        "A large / B small",
        "A large / B medium",
        "A large / B large"
    ]

    q_df = pd.DataFrame(
        q_table,
        columns=ACTION_NAMES,
        index=labels
    )

    st.dataframe(q_df.round(2), use_container_width=True)

st.divider()

st.caption(
    "Academic MVP | Python | Streamlit | NumPy | pandas | Q-Learning"
)
