import numpy as np
np.random.seed(42)


"""
I won't include 'done' in the GAE, which is usuall done,
because this has infinite timesteps, and
it's just redundant.

"""


class PendulumENV:
    def __init__(self):
        self.length = 1
        self.mass = 1
        self.g = 10
        self.theta = 0
        self.max_speed = 8.0
        self.max_torque = 2.0
        self.dt = 0.05
        self.theta_dot = 0.0
    def reset(self):
        self.theta = np.random.uniform(-np.pi, np.pi)
        self.theta_dot = np.random.uniform(-1.0, 1.0)
        return self._get_state()
    
    def theta_from_state(self, state):
        return np.arctan2(state[1], state[0])

    def _get_state(self):
        return np.array ([
            np.cos(self.theta),
            np.sin(self.theta),
            self.theta_dot
        ])

    def step(self, action):
        action = np.clip(action, -self.max_torque, self.max_torque)

        theta_acc = (3 * action) / (self.mass * self.length ** 2) - (3/2) * self.g/self.length * np.sin(self.theta)

        self.theta_dot += theta_acc * self.dt
        self.theta_dot = np.clip(self.theta_dot, -self.max_speed, self.max_speed)
        self.theta += self.theta_dot * self.dt

        reward = np.cos(self.theta) - 0.1 * self.theta_dot ** 2 - 0.001 * action ** 2



        return self._get_state(), reward, False
    
#Helper Functions

def compute_log_prob(action, mean, std):
    return -0.5 * ((action - mean) / std) ** 2 - np.log(std) - 0.5 * np.log(2 * np.pi)





from ActorAndCritic import Actor, Critic

episodes = 500
time_steps_per_batch = 2048
update_epochs = 10
gamma = 0.9
lambda_gae = 0.95
clip_epsilon = 0.2
lr = 0.003





def collect_trajectories(env, actor, critic, time_steps_per_batch):
    states, actions, rewards, values, old_log_probs = [], [], [], [], []
    state = env.reset()

    for _ in range(time_steps_per_batch):
        mean, std = actor.forward(state)  # Call forward ONCE
        action = np.random.normal(mean, std)  # Sample manually
        action = np.clip(action, -2.0, 2.0)
        log_prob = compute_log_prob(action, mean, std)  # Use helper
        
        value = critic.forward(state).squeeze()
        next_state, reward, _ = env.step(action.item())

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        values.append(value)
        old_log_probs.append(log_prob)
        state = next_state

    return (np.array(states), np.array(actions).reshape(-1, 1),
            np.array(rewards), np.array(values), 
            np.array(old_log_probs).reshape(-1, 1))


def compute_GAE(rewards, values, gamma, lambda_gae):
    advantages = np.zeros_like(rewards)
    gae = 0.0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0.0
        else:
            next_value = values[t+1]

        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lambda_gae * gae
        advantages[t] = gae

    returns = advantages + values
    return returns, advantages




def ppo_update(actor, critic, returns, states, old_log_probs, actions, advantages,
               epochs, batch_size, clip_epsilon, lr):
    n = len(states)

    for _ in range(epochs):
        idx = np.random.permutation(n)
        for start in range(0, n, batch_size):
            batch_idx = idx[start:start+batch_size]

            s = states[batch_idx]
            a = actions[batch_idx]
            old_lp = old_log_probs[batch_idx]
            ret = returns[batch_idx]
            adv = advantages[batch_idx].reshape(-1, 1)

            #normalize
            adv = (adv - adv.mean()) / (adv.std() + 1e-5)
            
            

            #actor loss
            mean, std = actor.forward(s)
            new_log_probs = compute_log_prob(a, mean, std)
            ratio = np.exp(new_log_probs - old_lp)
            clipped = np.clip(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
            d_log_prob = adv
            mask = (ratio * adv < clipped * adv)
            d_log_prob = np.where(mask, adv, 0.0)
            d_mean = d_log_prob * (a - mean) / (std ** 2)
            d_std = d_log_prob * ((a - mean) ** 2 - std ** 2) / (std ** 3)


            #backprop
            critic.backward(s, ret, lr)
            actor.backward(d_mean, d_std, lr)

        

actor = Actor()
critic = Critic()
env = PendulumENV()





final_states = None

for ep in range(episodes):
    states, actions, rewards, values, old_lp = collect_trajectories(env, actor, critic, time_steps_per_batch)
    returns, advantages = compute_GAE(rewards, values, gamma, lambda_gae)
    ppo_update(actor, critic, returns, states, old_lp, actions, advantages,
               update_epochs, 64, clip_epsilon, lr)
    

    if ep == episodes - 1:
        final_states = states
    
    if ep % 50 == 0:
        print(f"Ep {ep}, Avg reward: {np.mean(rewards):.2f}")


import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


np.random.seed(12345)  # Different seed
untrained_actor = Actor()
untrained_env = PendulumENV()



# Reset to seed 42 for the trained one (already done above)
np.random.seed(42)

state_trained = env.reset()
state_untrained = untrained_env.reset()

trained_states = []
trained_rewards = []
untrained_states = []
untrained_rewards = []

for _ in range(500):
    # Trained
    mean, _ = actor.forward(state_trained)
    action = np.clip(mean.item(), -2.0, 2.0)
    state_trained, reward_trained, _ = env.step(action)
    trained_states.append(state_trained)
    trained_rewards.append(reward_trained)
    
    # Untrained
    mean_u, std_u = untrained_actor.forward(state_untrained)
    action_u = np.random.normal(mean_u, std_u)
    action_u = np.clip(action_u, -2.0, 2.0)
    state_untrained, reward_untrained, _ = untrained_env.step(action_u.item())
    untrained_states.append(state_untrained)
    untrained_rewards.append(reward_untrained)




fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
fig.patch.set_facecolor('#0d0d1a')

for ax in [ax1, ax2]:
    ax.set_facecolor('#0d0d1a')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')

# Trained (right)
ax2.plot(0, 0, 'o', color='white', markersize=10)
rod_trained, = ax2.plot([], [], 'r-', linewidth=5)
mass_trained, = ax2.plot([], [], 'o', color='#00ff88', markersize=18)
reward_trained_text = ax2.text(-1.4, 1.3, '', fontsize=14, color='white', fontfamily='monospace')
ax2.set_title('TRAINED PPO', fontsize=16, color='#00ff88', fontweight='bold', pad=15)

# Untrained (left)
ax1.plot(0, 0, 'o', color='white', markersize=10)
rod_untrained, = ax1.plot([], [], 'r-', linewidth=5)
mass_untrained, = ax1.plot([], [], 'o', color='#ff6b6b', markersize=18)
reward_untrained_text = ax1.text(-1.4, 1.3, '', fontsize=14, color='white', fontfamily='monospace')
ax1.set_title('UNTRAINED', fontsize=16, color='#ff6b6b', fontweight='bold', pad=15)

def animate(frame):
    # Trained
    theta_t = env.theta_from_state(trained_states[frame])
    rod_trained.set_data([0, np.sin(theta_t)], [0, np.cos(theta_t)])
    mass_trained.set_data([np.sin(theta_t)], [np.cos(theta_t)])
    reward_trained_text.set_text(f'Reward: {trained_rewards[frame]:.2f}')
    
    # Untrained
    theta_u = untrained_env.theta_from_state(untrained_states[frame])
    rod_untrained.set_data([0, np.sin(theta_u)], [0, np.cos(theta_u)])
    mass_untrained.set_data([np.sin(theta_u)], [np.cos(theta_u)])
    reward_untrained_text.set_text(f'Reward: {untrained_rewards[frame]:.2f}')
    
    return rod_trained, mass_trained, reward_trained_text, rod_untrained, mass_untrained, reward_untrained_text

ani = animation.FuncAnimation(fig, animate, frames=500, interval=20, blit=True)
plt.suptitle('PPO Pendulum — Trained vs Untrained', fontsize=20, color='white', fontweight='bold', y=0.98)
plt.show()