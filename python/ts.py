import pygame
import random
import sys
import math
import os
import json
from dataclasses import dataclass

pygame.init()

# ------------------ Config ------------------


@dataclass(frozen=True)
class DisplayConfig:
    width: int = 1200
    height: int = 740
    fps: int = 60


@dataclass(frozen=True)
class AssetConfig:
    background_image: str = "python/images/bg_forest_dark.png"
    player_image: str = "python/images/f22.png"
    enemy_image: str = "python/images/su57.png"
    cloud_image: str = "python/images/cloud5.png"
    enemy_target_height: int = 60
    cloud_scale: float = 0.9
    player_target_height: int = 70


@dataclass(frozen=True)
class GameplayConfig:
    enemy_speed_min: float = 170.0
    enemy_speed_max: float = 210.0
    enemy_spawn_interval: float = 0.15
    cloud_scroll_speed: float = 140.0
    player_speed: float = 700.0
    bot_speed: float = 700.0
    bullet_speed: float = 700.0
    grid_spacing: int = 40
    grid_scroll_speed: float = 100.0
    p2_accuracy_focus: float = 10.0
    p2_aim_accuracy: float = 0.9
    p2_miss_offset: float = 60.0
    p1_bullet_cooldown: float = 0.10
    bot_bullet_cooldown: float = 0.10


@dataclass(frozen=True)
class UiConfig:
    hud_font_size: int = 28
    popup_font_size: int = 24
    heatmap_height: int = 70
    heatmap_margin_left: int = 12
    heatmap_margin_right: int = 0
    heatmap_margin_bottom: int = 0
    corr_window_size: int = 600
    q_log_interval: float = 10.0

    @property
    def play_area_bottom(self):
        return DISPLAY.height - self.heatmap_height


@dataclass(frozen=True)
class PersistenceConfig:
    start_from_saved_q: bool = False
    q_save_file_p2: str = "q_table_p2.json"
    q_save_file_p3: str = "q_table_p3.json"
    q_save_file_p4_mean: str = "q_table_p4_mean.json"
    q_save_file_p4_var: str = "q_table_p4_var.json"


DISPLAY = DisplayConfig()
ASSETS = AssetConfig()
GAME = GameplayConfig()
UI = UiConfig()
PERSISTENCE = PersistenceConfig()

BG_COLOR      = (10, 10, 20)
GRID_COLOR    = (40, 80, 40)
PLAYER1_COLOR = (0, 255, 0)   # P1: yellow-ish
PLAYER2_COLOR = (80, 200, 255)   # P2: blue-ish
PLAYER3_COLOR = (255, 120, 255)  # P3: magenta-ish
PLAYER4_COLOR = (0, 255, 180)    # P4: teal-ish
ENEMY_COLOR   = (255, 80, 80) # red
# ENEMY_COLOR   = (0, 0, 0) # black
BULLET_COLOR  = (0, 255, 0)

OWNER_COLORS = {
    "P1": PLAYER1_COLOR,
    "P2": PLAYER2_COLOR,
    "P3": PLAYER3_COLOR,
    "P4": PLAYER4_COLOR,
}

OWNER_IDS = ("P1", "P2", "P3", "P4")
BOT_OWNER_IDS = ("P2", "P3", "P4")


P3_ACCURACY_FOCUS = GAME.p2_accuracy_focus

# Scoring modes
MODE_TEAM   = "TEAM"
MODE_VERSUS = "VERSUS"

# Players should not be able to move into the Q-heatmap area at the bottom.
# Total heatmap height = UI.heatmap_height.


# ------------------ RL algorithms ------------------
ALGO_Q_LEARNING     = "Q_LEARNING"
ALGO_SARSA          = "SARSA"
ALGO_EXPECTED_SARSA = "EXPECTED_SARSA"
ALGO_N_STEP_SARSA   = "N_STEP_SARSA"
ALGO_BAYES_Q        = "BAYES_Q"   

# N-step horizon for N-STEP SARSA
N_STEP = 2

# --------------- RL config (RL players) ---------------
N_STATE_BINS = 49   # finer spatial precision

A_MOVE_LEFT, A_MOVE_RIGHT, A_STAY, A_FIRE = range(4)
N_ACTIONS = 4


# ---------- P2 RL params ----------
RL_ALPHA_P2 = 0.20   # learning rate 
RL_GAMMA_P2 = 0.99   # discount rate

RL_EPSILON_START_P2 = 0.99   # initial exploration 
RL_EPSILON_MIN_P2   = 0.01   # floor 
RL_EPSILON_DECAY_P2 = 0.95  # multiplicative decay per second 


# ---------- P3 RL params ----------
RL_ALPHA_P3 = 0.20   # learning rate 
RL_GAMMA_P3 = 0.95   # discount rate

RL_EPSILON_START_P3 = 0.99   # initial exploration
RL_EPSILON_MIN_P3   = 0.01   # floor
RL_EPSILON_DECAY_P3 = 0.98  # multiplicative decay per second


# Main RL rewards (shared)
HIT_REWARD       = 10.0    # strong for kills
ESCAPE_PENALTY   = -1.0    # strong when an enemy escapes

# Accuracy shaping (shared by P2/P3)
ACCURACY_FIRE_COST  = -1.5  # small cost whenever a bot fires
MISS_BULLET_PENALTY = -1.0  # penalty when a bot's bullet leaves screen

# Shaping weights (small; just guidance)
PROXIMITY_WEIGHT        = 0.1
ALIGN_FIRE_BONUS        = 1.0
ALIGN_NO_FIRE_PENALTY   = -1.0
IDLE_ALIGNED_PENALTY    = -0.1
MISALIGNED_FIRE_PENALTY = -0.1
TOWARD_MOVE_BONUS       = 0.1
STEP_REWARD             = 0.05
CENTERING_WEIGHT        = 0.15


# Separate Q-tables for P2 and P3
Q_P2 = [[0.0 for _ in range(N_ACTIONS)] for _ in range(N_STATE_BINS)]
Q_P3 = [[0.0 for _ in range(N_ACTIONS)] for _ in range(N_STATE_BINS)]

# Bayesian Q "ideal observer" tables (posterior mean & variance) for P4
Q_BAYES_MEAN = [[0.0 for _ in range(N_ACTIONS)] for _ in range(N_STATE_BINS)]
Q_BAYES_VAR  = [[10.0 for _ in range(N_ACTIONS)] for _ in range(N_STATE_BINS)]  # large prior variance
BAYES_OBS_VAR = 1.0  # assumed noise variance in TD targets
BAYES_PROCESS_VAR  = 0.4  # how much uncertainty we add back each update
BAYES_MIN_VAR      = 0.4   # floor on variance so learning rate never dies

# Bayesian Q exploration (P4)
BAYES_EPSILON = 0.3  # 30% of the time, choose a random action
BAYES_EPSILON_START = 0.3
BAYES_EPSILON_MIN   = 0.05
BAYES_EPSILON_DECAY = 0.98  # per second, similar to other agents


def save_q_tables():
    """Save current Q_P2/Q_P3 and Bayesian Q_BAYES_MEAN/Q_BAYES_VAR to disk as JSON."""
    try:
        with open(PERSISTENCE.q_save_file_p2, "w") as f:
            json.dump(Q_P2, f)
        with open(PERSISTENCE.q_save_file_p3, "w") as f:
            json.dump(Q_P3, f)
        with open(PERSISTENCE.q_save_file_p4_mean, "w") as f:
            json.dump(Q_BAYES_MEAN, f)
        with open(PERSISTENCE.q_save_file_p4_var, "w") as f:
            json.dump(Q_BAYES_VAR, f)
        print("Saved Q-tables (P2, P3, P4-Bayes) to disk.")
    except Exception as e:
        print(f"Error saving Q-tables: {e}")


def _shape_ok(table):
    return (
        isinstance(table, list)
        and len(table) == N_STATE_BINS
        and all(isinstance(row, list) and len(row) == N_ACTIONS for row in table)
    )


def load_q_tables_if_enabled():
    """
    If saved-Q loading is enabled and files exist, load Q_P2/Q_P3 and P4 Bayes tables from disk.
    Otherwise leave them as freshly initialised.
    """
    global Q_P2, Q_P3, Q_BAYES_MEAN, Q_BAYES_VAR

    if not PERSISTENCE.start_from_saved_q:
        print("Saved Q-table loading disabled; starting from fresh Q-tables.")
        return

    try:
        # ---- P2 ----
        if os.path.exists(PERSISTENCE.q_save_file_p2):
            with open(PERSISTENCE.q_save_file_p2, "r") as f:
                loaded_p2 = json.load(f)
            if _shape_ok(loaded_p2):
                Q_P2 = loaded_p2
                print("Loaded Q_P2 from disk.")
            else:
                print("Saved Q_P2 has incompatible shape; using fresh table.")
        else:
            print("No saved Q_P2 found; using fresh table.")

        # ---- P3 ----
        if os.path.exists(PERSISTENCE.q_save_file_p3):
            with open(PERSISTENCE.q_save_file_p3, "r") as f:
                loaded_p3 = json.load(f)
            if _shape_ok(loaded_p3):
                Q_P3 = loaded_p3
                print("Loaded Q_P3 from disk.")
            else:
                print("Saved Q_P3 has incompatible shape; using fresh table.")
        else:
            print("No saved Q_P3 found; using fresh table.")

        # ---- P4 Bayes mean ----
        if os.path.exists(PERSISTENCE.q_save_file_p4_mean):
            with open(PERSISTENCE.q_save_file_p4_mean, "r") as f:
                loaded_mean = json.load(f)
            if _shape_ok(loaded_mean):
                Q_BAYES_MEAN = loaded_mean
                print("Loaded Q_BAYES_MEAN from disk.")
            else:
                print("Saved Q_BAYES_MEAN has incompatible shape; using fresh prior.")
        else:
            print("No saved Q_BAYES_MEAN found; using fresh prior.")

        # ---- P4 Bayes var ----
        if os.path.exists(PERSISTENCE.q_save_file_p4_var):
            with open(PERSISTENCE.q_save_file_p4_var, "r") as f:
                loaded_var = json.load(f)
            if _shape_ok(loaded_var):
                Q_BAYES_VAR = loaded_var
                print("Loaded Q_BAYES_VAR from disk.")
            else:
                print("Saved Q_BAYES_VAR has incompatible shape; using fresh prior variances.")
        else:
            print("No saved Q_BAYES_VAR found; using fresh prior variances.")

    except Exception as e:
        print(f"Error loading Q-tables, starting from fresh: {e}")

# Q heatmap visual scaling (just for colors, not the learning itself)
Q_VIS_MIN = -5.0
Q_VIS_MAX =  5.0


# ------------- Correlation heatmap config -------------
# Number of recent frames to use for correlation (moving window)


def compute_corr_matrix(history):
    """
    Compute an NxN Pearson correlation matrix for agents' x-positions
    from a history of tuples (x1, x2, ..., xN).

    Returns: NxN list of floats in [-1, 1].
    If insufficient or degenerate data, returns identity matrix.
    """
    n = len(history)
    if n < 2:
        # If we have no or just one sample, return an identity of size = dim
        dim = len(history[0]) if n > 0 else 4  # default to 4
        return [
            [1.0 if i == j else 0.0 for j in range(dim)]
            for i in range(dim)
        ]

    dim = len(history[0])  # number of agents (4)

    # Means
    sums = [0.0] * dim
    for tup in history:
        for k in range(dim):
            sums[k] += tup[k]
    means = [s / n for s in sums]

    # Variances & covariances
    centered = []
    for tup in history:
        centered.append([tup[k] - means[k] for k in range(dim)])

    var = [0.0] * dim
    cov = [[0.0 for _ in range(dim)] for _ in range(dim)]

    for c in centered:
        for i in range(dim):
            var[i] += c[i] * c[i]
            for j in range(dim):
                cov[i][j] += c[i] * c[j]

    denom = float(n - 1)
    for i in range(dim):
        var[i] /= denom
    for i in range(dim):
        for j in range(dim):
            cov[i][j] /= denom

    std = [math.sqrt(max(v, 1e-8)) for v in var]

    corr = [[0.0 for _ in range(dim)] for _ in range(dim)]
    for i in range(dim):
        for j in range(dim):
            if i == j:
                corr[i][j] = 1.0
            else:
                denom_ij = std[i] * std[j]
                if denom_ij > 0:
                    corr[i][j] = max(-1.0, min(1.0, cov[i][j] / denom_ij))
                else:
                    corr[i][j] = 0.0
    return corr



def draw_corr_heatmap(surf, corr, top_left_x, top_left_y, size):
    """
    Draw an NxN correlation heatmap in a square region.

    corr: NxN list of floats in [-1, 1]
    Color map: -1 -> blue, 0 -> blackish, +1 -> red
    Rows/cols correspond to agents P1, P2, P3, P4 (truncated if fewer).
    """
    dim = len(corr)  # should be 4 for P1–P4
    if dim == 0:
        return

    cell_size = size / float(dim)

    # Draw cells
    for i in range(dim):       # rows
        for j in range(dim):   # cols
            x = top_left_x + j * cell_size
            y = top_left_y + i * cell_size

            val = corr[i][j]          # in [-1, 1]
            color = palette_blue_white_red(val)

            pygame.draw.rect(
                surf,
                color,
                pygame.Rect(
                    int(x),
                    int(y),
                    int(math.ceil(cell_size)),
                    int(math.ceil(cell_size)),
                ),
            )

    # Thin white grid lines
    for k in range(dim + 1):
        x = int(top_left_x + k * cell_size)
        y = int(top_left_y + k * cell_size)
        pygame.draw.line(
            surf, (220, 220, 220),
            (x, top_left_y),
            (x, top_left_y + size),
            1,
        )
        pygame.draw.line(
            surf, (220, 220, 220),
            (top_left_x, y),
            (top_left_x + size, y),
            1,
        )

    # Small labels
    labels_all = ["P1", "P2", "P3", "P4"]
    labels = labels_all[:dim]
    font = pygame.font.SysFont(None, 16)

    # Column labels (top)
    for j, lab in enumerate(labels):
        text = font.render(lab, True, (230, 230, 230))
        tx = top_left_x + j * cell_size + cell_size / 2 - text.get_width() / 2
        ty = top_left_y - text.get_height() - 2
        surf.blit(text, (int(tx), int(ty)))

    # Row labels (left)
    for i, lab in enumerate(labels):
        text = font.render(lab, True, (230, 230, 230))
        tx = top_left_x - text.get_width() - 4
        ty = top_left_y + i * cell_size + cell_size / 2 - text.get_height() / 2
        surf.blit(text, (int(tx), int(ty)))



# ------------------ RL / AI helpers ------------------

def rl_update(Q_table, alpha, gamma, prev_state, prev_action, reward, next_state):
    """
    TD(0) Q-learning update:
        Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
    """
    if prev_state is None or prev_action is None or next_state is None:
        return

    q_old = Q_table[prev_state][prev_action]
    max_next = max(Q_table[next_state])
    target = reward + gamma * max_next
    Q_table[prev_state][prev_action] = q_old + alpha * (target - q_old)


def rl_update_sarsa(Q_table, alpha, gamma,
                    prev_state, prev_action, reward,
                    next_state, next_action):
    """
    On-policy SARSA:
        Q(s,a) <- Q(s,a) + alpha * (r + gamma * Q(s',a') - Q(s,a))
    """
    if (prev_state is None or prev_action is None or
        next_state is None or next_action is None):
        return

    q_old  = Q_table[prev_state][prev_action]
    q_next = Q_table[next_state][next_action]
    target = reward + gamma * q_next
    Q_table[prev_state][prev_action] = q_old + alpha * (target - q_old)


def rl_update_expected_sarsa(Q_table, alpha, gamma,
                             prev_state, prev_action, reward,
                             next_state, epsilon, n_actions):
    """
    Expected SARSA:
        Q(s,a) <- Q(s,a) + alpha * (r + gamma * E_pi[Q(s',·)] - Q(s,a))
    where pi is ε-greedy.
    """
    if prev_state is None or prev_action is None or next_state is None:
        return

    q_next = Q_table[next_state]
    max_q  = max(q_next)
    greedy_actions = [a for a, q in enumerate(q_next) if q == max_q]

    # ε-greedy distribution over actions
    pi = [0.0] * n_actions
    for a in range(n_actions):
        if a in greedy_actions:
            pi[a] = (1.0 - epsilon) / len(greedy_actions) + epsilon / n_actions
        else:
            pi[a] = epsilon / n_actions

    expected_value = sum(pi[a] * q_next[a] for a in range(n_actions))

    q_old = Q_table[prev_state][prev_action]
    target = reward + gamma * expected_value
    Q_table[prev_state][prev_action] = q_old + alpha * (target - q_old)


def rl_update_n_step_sarsa(Q_table, alpha, gamma,
                           trajectory, next_state, next_action):
    """
    N-step SARSA using a trajectory of (state, action, reward) tuples.

    trajectory: list[(s_t, a_t, r_t), ...]  length = 1..N_STEP
    next_state, next_action: bootstrap state/action after the window.
    """
    if not trajectory:
        return

    states  = [s for (s, a, r) in trajectory]
    actions = [a for (s, a, r) in trajectory]
    rewards = [r for (s, a, r) in trajectory]

    G = 0.0
    for i, r in enumerate(rewards):
        G += (gamma ** i) * r

    if next_state is not None and next_action is not None:
        G += (gamma ** len(rewards)) * Q_table[next_state][next_action]

    s0 = states[0]
    a0 = actions[0]

    q_old = Q_table[s0][a0]
    Q_table[s0][a0] = q_old + alpha * (G - q_old)


def bayes_update_q(q_mean, q_var, s, a, target):
    """
    Bayesian Q update with process noise and a variance floor, so the
    posterior over Q(s,a) doesn't collapse to a delta and stop learning.
    """
    mu_old  = q_mean[s][a]
    var_old = q_var[s][a]

    # 1) Add process noise to the prior: model non-stationarity
    var_prior = var_old + BAYES_PROCESS_VAR

    # 2) Observation noise (uncertainty in TD targets)
    var_obs = BAYES_OBS_VAR

    # 3) Posterior variance
    var_post = 1.0 / (1.0 / var_prior + 1.0 / var_obs)

    # 4) Enforce a minimum variance so we never become completely rigid
    var_post = max(var_post, BAYES_MIN_VAR)

    # 5) Posterior mean
    mu_post = var_post * (mu_old / var_prior + target / var_obs)

    q_mean[s][a] = mu_post
    q_var[s][a]  = var_post

    # Optional debug
    if s == N_STATE_BINS // 2 and a == A_FIRE and random.random() < 0.001:
        print(f"[Bayes-Q] s={s}, a={a}, target={target:.3f}, mu={mu_post:.3f}, var={var_post:.3f}")


def choose_target_enemy(p2, enemies):
    """
    Choose an enemy for a bot (P2/P3) to track/shoot:

    - Only consider enemies that are *ahead* of the bot:
        enemy.rect.bottom < p2.y   (i.e., enemy is above the bot on screen)
    - Among those, pick the one with the smallest Euclidean distance:
        sqrt( (dx)^2 + (dy)^2 )
    - If there are no such enemies, return None (no target).
    """
    if not enemies:
        return None

    p2_center_x = p2.x + p2.width / 2

    # Only enemies ABOVE the bot
    candidates = [e for e in enemies if e.rect.bottom < p2.y]
    if not candidates:
        return None

    def dist2(e):
        dx = e.rect.centerx - p2_center_x
        dy = p2.y - e.rect.centery  # positive when enemy is above the bot
        return dx * dx + dy * dy    # squared distance (no need for sqrt)

    return min(candidates, key=dist2)


def sample_spawn_x_gaussian_mixture():
    """
    Sample an x position from a mixture of two Gaussians:
    - Stronger peak at ~1/3 of screen width
    - Slightly weaker peak at ~2/3 of screen width
    """
    while True:
        if random.random() < 0.6:
            mu = DISPLAY.width / 3.0
        else:
            mu = 2.0 * DISPLAY.width / 3.0

        sigma = DISPLAY.width / 10.0
        x = random.gauss(mu, sigma)

        if 0 <= x <= DISPLAY.width - 30:
            return int(x)


def get_state_bin(p2, enemies):
    """
    Discretise horizontal relation between P2 and its chosen target into bins.
    state = 0 .. N_STATE_BINS-1
    """
    if not enemies:
        return N_STATE_BINS // 2

    target = choose_target_enemy(p2, enemies)
    if target is None:
        return N_STATE_BINS // 2

    p2_center_x = p2.x + p2.width / 2
    rel = target.rect.centerx - p2_center_x  # negative: enemy left, positive: enemy right

    rel_norm = (rel + DISPLAY.width / 2) / DISPLAY.width
    rel_norm = max(0.0, min(1.0, rel_norm))

    idx = int(rel_norm * N_STATE_BINS)
    if idx >= N_STATE_BINS:
        idx = N_STATE_BINS - 1
    return idx


def get_rel_x(p2, enemies):
    """
    Return horizontal offset (enemy_x - p2_center_x) for P2's chosen target.
    If no suitable target, returns None.
    """
    if not enemies:
        return None
    target = choose_target_enemy(p2, enemies)
    if target is None:
        return None
    p2_center_x = p2.x + p2.width / 2
    return target.rect.centerx - p2_center_x
  

def algo_display_name(algo):
    if algo == ALGO_Q_LEARNING:
        return "Q-LEARNING"
    elif algo == ALGO_SARSA:
        return "SARSA"
    elif algo == ALGO_EXPECTED_SARSA:
        return "EXPECTED SARSA"
    elif algo == ALGO_N_STEP_SARSA:
        return "N-STEP SARSA"
    else:
        return str(algo)


@dataclass
class ScoreState:
    score: int = 0
    shots: int = 0
    hits: int = 0
    misses: int = 0

    @property
    def accuracy(self):
        return (self.hits / self.shots * 100.0) if self.shots > 0 else 0.0


@dataclass
class BotRuntime:
    owner: str
    player: "Player"
    controller: object
    q_table: list
    current_state: int | None = None
    current_action: int | None = None
    reward_step: float = 0.0

    @property
    def color(self):
        return OWNER_COLORS[self.owner]

    @property
    def heatmap_label(self):
        if self.owner == "P4":
            return "P4 BAYES-Q"
        if self.controller.algo == ALGO_N_STEP_SARSA:
            return f"{self.owner} {self.controller.n_step}-STEP SARSA"
        return f"{self.owner} {algo_display_name(self.controller.algo)}"


def clamp_player_x(player):
    player.x = max(0, min(DISPLAY.width - player.width, player.x))
    player.rect.topleft = (player.x, player.y)


def compute_alignment_reward(action, rel, accuracy_focus):
    reward_step = 0.0
    bin_width = DISPLAY.width / N_STATE_BINS
    aligned_thresh = 0.4 * bin_width if bin_width > 0 else 0

    if rel is not None and aligned_thresh > 0:
        abs_rel = abs(rel)
        max_rel = DISPLAY.width / 2.0
        closeness = 1.0 - min(abs_rel / max_rel, 1.0)
        closeness_sq = closeness * closeness

        reward_step += PROXIMITY_WEIGHT * closeness_sq

        if abs_rel < aligned_thresh:
            if action == A_FIRE:
                reward_step += ALIGN_FIRE_BONUS * accuracy_focus
            else:
                reward_step += ALIGN_NO_FIRE_PENALTY * accuracy_focus

            if action == A_STAY:
                reward_step += IDLE_ALIGNED_PENALTY * accuracy_focus

        if action == A_FIRE and abs_rel > 2 * aligned_thresh:
            reward_step += MISALIGNED_FIRE_PENALTY * accuracy_focus

        if rel < 0 and action == A_MOVE_RIGHT:
            reward_step += TOWARD_MOVE_BONUS
        if rel > 0 and action == A_MOVE_LEFT:
            reward_step += TOWARD_MOVE_BONUS

    return reward_step


def compute_centering_reward(player, enemies):
    if enemies:
        return 0.0

    center_x = DISPLAY.width / 2.0
    agent_center_x = player.x + player.width / 2.0
    dist_from_center = abs(agent_center_x - center_x)
    norm = min(dist_from_center / (DISPLAY.width / 2.0), 1.0)
    return CENTERING_WEIGHT * (1.0 - norm)


def maybe_fire_bullet(player, action, enemies, bullets, cooldown_timer, bullet_cooldown,
                      aim_acc_prob, miss_offset, bullet_owner):
    fired_this_frame = False

    if action != A_FIRE or cooldown_timer < bullet_cooldown or not enemies:
        return cooldown_timer, fired_this_frame

    target = choose_target_enemy(player, enemies)
    if target is None or target.rect.bottom >= player.y:
        return cooldown_timer, fired_this_frame

    center_x = player.x + player.width / 2
    if random.random() < aim_acc_prob:
        bullet_x = center_x
    else:
        bullet_x = center_x + miss_offset * random.choice([-1, 1])

    bullets.append(Bullet(bullet_x, player.y, owner=bullet_owner))
    return 0.0, True


def apply_agent_action(player, action, enemies, bullets, dt, speed, time_since_last_shot,
                       bullet_cooldown, accuracy_focus, aim_acc_prob, miss_offset, bullet_owner):
    reward_step = STEP_REWARD

    if action == A_MOVE_LEFT:
        player.x -= speed * dt
    elif action == A_MOVE_RIGHT:
        player.x += speed * dt

    clamp_player_x(player)

    rel = get_rel_x(player, enemies)
    reward_step += compute_alignment_reward(action, rel, accuracy_focus)
    reward_step += compute_centering_reward(player, enemies)

    time_since_last_shot += dt
    time_since_last_shot, fired_this_frame = maybe_fire_bullet(
        player,
        action,
        enemies,
        bullets,
        time_since_last_shot,
        bullet_cooldown,
        aim_acc_prob,
        miss_offset,
        bullet_owner,
    )
    if fired_this_frame:
        reward_step += ACCURACY_FIRE_COST * accuracy_focus

    return reward_step, fired_this_frame, time_since_last_shot


def update_corr_history(history, players):
    centers = tuple(player.x + player.width / 2.0 for player in players)
    history.append(centers)
    if len(history) > UI.corr_window_size:
        history.pop(0)
    return compute_corr_matrix(history)


def update_projectiles(bullets, focus_by_owner, dt):
    reward_updates = {owner: 0.0 for owner in BOT_OWNER_IDS}
    new_bullets = []

    for bullet in bullets:
        bullet.update(dt)
        if bullet.off_screen():
            if bullet.owner in reward_updates:
                reward_updates[bullet.owner] += MISS_BULLET_PENALTY * focus_by_owner[bullet.owner]
        else:
            new_bullets.append(bullet)

    return new_bullets, reward_updates


def update_enemies(enemies, dt):
    escaped = []
    active = []
    for enemy in enemies:
        enemy.update(dt)
        if enemy.off_screen():
            escaped.append(enemy)
        else:
            active.append(enemy)
    return active, escaped


def owner_closest_to_enemy(enemy_center_x, players_by_owner):
    return min(
        OWNER_IDS,
        key=lambda owner: abs(enemy_center_x - (players_by_owner[owner].x + players_by_owner[owner].width / 2)),
    )


def apply_escape_consequences(escaped_enemies, stats_by_owner, bot_states, players_by_owner,
                              training_mode, mode, score_popups):
    reward_updates = {owner: 0.0 for owner in BOT_OWNER_IDS}

    for enemy in escaped_enemies:
        enemy_center_x = enemy.rect.centerx

        if training_mode:
            impacted_owners = BOT_OWNER_IDS
        elif mode == MODE_TEAM:
            impacted_owners = OWNER_IDS
        else:
            impacted_owners = (owner_closest_to_enemy(enemy_center_x, players_by_owner),)

        for owner in impacted_owners:
            stats_by_owner[owner].score -= 10
            stats_by_owner[owner].misses += 1
            if owner in reward_updates:
                reward_updates[owner] += ESCAPE_PENALTY * bot_states[owner].controller.accuracy_focus

        score_popups.append(
            ScorePopup(
                enemy_center_x,
                DISPLAY.height - UI.heatmap_height - 10,
                "-10",
                (255, 0, 0),
            )
        )

    return reward_updates


def resolve_bullet_collisions(bullets, enemies, stats_by_owner, bot_states, training_mode, score_popups):
    reward_updates = {owner: 0.0 for owner in BOT_OWNER_IDS}

    for bullet in bullets[:]:
        for enemy in enemies[:]:
            if not bullet.rect.colliderect(enemy.rect):
                continue

            if bullet.owner != "P1" or not training_mode:
                stats_by_owner[bullet.owner].score += 10
                stats_by_owner[bullet.owner].hits += 1

            popup_color = OWNER_COLORS.get(bullet.owner, BULLET_COLOR)
            if bullet.owner in reward_updates:
                reward_updates[bullet.owner] += HIT_REWARD * bot_states[bullet.owner].controller.accuracy_focus

            score_popups.append(
                ScorePopup(enemy.rect.centerx, enemy.rect.centery, "+10", popup_color)
            )
            bullets.remove(bullet)
            enemies.remove(enemy)
            break

    return reward_updates


def update_player1(player1, keys, bullets, dt, time_since_last_shot, training_mode):
    if training_mode:
        return time_since_last_shot, False

    time_since_last_shot += dt
    fired_this_frame = False
    if keys[pygame.K_SPACE] and time_since_last_shot >= GAME.p1_bullet_cooldown:
        bullet_x = player1.x + player1.width / 2
        bullets.append(Bullet(bullet_x, player1.y, owner="P1"))
        time_since_last_shot = 0.0
        fired_this_frame = True

    player1.update(dt, keys)
    return time_since_last_shot, fired_this_frame


# ------------------ Helper classes ------------------

class Player:
    def __init__(self, x, y, color, sprite=None):
        self.sprite = sprite
        self.color  = color

        if sprite is not None:
            self.width, self.height = sprite.get_size()
        else:
            self.width  = 30
            self.height = 40

        self.x = x
        self.y = y
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)

    def update(self, dt, keys):
        dx = 0
        dy = 0
        if keys[pygame.K_LEFT]:
            dx -= GAME.player_speed * dt
        if keys[pygame.K_RIGHT]:
            dx += GAME.player_speed * dt
        if keys[pygame.K_UP]:
            dy -= GAME.player_speed * dt
        if keys[pygame.K_DOWN]:
            dy += GAME.player_speed * dt

        self.x += dx
        self.y += dy

        # Horizontal clamp (full width)
        self.x = max(0, min(DISPLAY.width  - self.width,  self.x))
        # Vertical clamp: stop at the play area, not the very bottom of the screen
        self.y = max(0, min(UI.play_area_bottom - self.height, self.y))

        self.rect.topleft = (self.x, self.y)

    def draw(self, surf):
        if self.sprite is not None:
            # Sprite is drawn with its top-left at (x, y)
            surf.blit(self.sprite, (self.x, self.y))
        else:
            # Fallback: original triangle jet
            nose   = (self.x + self.width / 2, self.y)
            left   = (self.x, self.y + self.height)
            right  = (self.x + self.width, self.y + self.height)
            pygame.draw.polygon(surf, self.color, [left, right, nose])


class Bullet:
    def __init__(self, x, y, owner="P1"):
        self.width  = 4
        self.height = 12
        self.x = x - self.width / 2
        self.y = y
        self.owner = owner  # "P1", "P2", "P3", "P4"
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)

        self.color = OWNER_COLORS.get(owner, BULLET_COLOR)

    def update(self, dt):
        self.y -= GAME.bullet_speed * dt
        self.rect.topleft = (self.x, self.y)

    def off_screen(self):
        return self.y + self.height < 0

    def draw(self, surf):
        pygame.draw.rect(surf, self.color, self.rect)


class Enemy:
    def __init__(self, x, y, base_sprite):
        # 1) Sample a speed for this enemy
        self.speed = random.uniform(GAME.enemy_speed_min, GAME.enemy_speed_max)

        # 2) Map speed -> scale factor
        #    slow (ENEMY_SPEED_MIN)  → SCALE_MIN
        #    fast (ENEMY_SPEED_MAX)  → SCALE_MAX
        SCALE_MIN = 0.88   # 12% smaller for the slowest
        SCALE_MAX = 1.12   # 12% larger for the fastest

        speed_range = GAME.enemy_speed_max - GAME.enemy_speed_min
        if speed_range > 0:
            t = (self.speed - GAME.enemy_speed_min) / speed_range  # 0 = slow, 1 = fast
        else:
            t = 0.5  # fallback if min==max

        scale_factor = SCALE_MIN + (SCALE_MAX - SCALE_MIN) * t

        # 3) Create a per-enemy scaled sprite from the shared base_sprite
        w = base_sprite.get_width()
        h = base_sprite.get_height()
        new_size = (int(w * scale_factor), int(h * scale_factor))
        self.sprite = pygame.transform.smoothscale(base_sprite, new_size)

        # 4) Set up rect / position
        self.width, self.height = self.sprite.get_size()

        # position is top-left of the sprite
        self.x = x
        self.y = y
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)

    def update(self, dt):
        self.y += self.speed * dt
        self.rect.topleft = (self.x, self.y)

    def off_screen(self):
        return self.y > DISPLAY.height

    def draw(self, surf):
        # Draw the sprite instead of a red triangle
        surf.blit(self.sprite, (self.x, self.y))


class RLAgent:
    """
    Generic tabular RL agent that can use:
      - Q-learning (TD(0))
      - SARSA
      - Expected SARSA
      - N-step SARSA
    It controls a Player but does not own it.
    """

    def __init__(
        self,
        name,
        algo,
        Q_table,
        alpha,
        gamma,
        epsilon_start,
        epsilon_min,
        epsilon_decay,
        bullet_cooldown,
        speed,
        accuracy_focus=1.0,
        aim_acc_prob=1.0,
        miss_offset=0.0,
        bullet_owner="P2",
        n_step=N_STEP,
    ):
        self.name           = name
        self.algo           = algo
        self.Q              = Q_table
        self.alpha          = alpha
        self.gamma          = gamma
        self.epsilon        = epsilon_start
        self.epsilon_min    = epsilon_min
        self.epsilon_decay  = epsilon_decay
        self.bullet_cooldown = bullet_cooldown
        self.speed          = speed
        self.accuracy_focus = accuracy_focus
        self.aim_acc_prob   = aim_acc_prob
        self.miss_offset    = miss_offset
        self.bullet_owner   = bullet_owner
        self.n_step         = n_step

        # Internal RL state
        self.prev_state  = None
        self.prev_action = None
        self.prev_reward = 0.0

        self.time_since_last_shot = 0.0
        self.q_log_timer          = 0.0

        # For N-step SARSA
        self.trajectory = []

    def select_action(self, state):
        """ε-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randint(0, N_ACTIONS - 1)

        q_row = self.Q[state]
        max_q = max(q_row)
        best_actions = [i for i, q in enumerate(q_row) if q == max_q]
        return random.choice(best_actions)

    def _do_update(self, current_state, current_action):
        """Apply the chosen TD rule based on stored prev_state/action/reward."""
        ps = self.prev_state
        pa = self.prev_action
        r  = self.prev_reward
        ns = current_state
        na = current_action

        if ps is None or pa is None:
            return

        if self.algo == ALGO_Q_LEARNING:
            # standard TD(0) Q-learning
            rl_update(self.Q, self.alpha, self.gamma, ps, pa, r, ns)

        elif self.algo == ALGO_SARSA:
            rl_update_sarsa(self.Q, self.alpha, self.gamma, ps, pa, r, ns, na)

        elif self.algo == ALGO_EXPECTED_SARSA:
            rl_update_expected_sarsa(
                self.Q, self.alpha, self.gamma,
                ps, pa, r,
                ns, self.epsilon, N_ACTIONS
            )

        elif self.algo == ALGO_N_STEP_SARSA:
            # accumulate trajectory then apply N-step updates
            self.trajectory.append((ps, pa, r))
            if len(self.trajectory) >= self.n_step:
                window = self.trajectory[:self.n_step]
                rl_update_n_step_sarsa(
                    self.Q, self.alpha, self.gamma,
                    window, ns, na
                )
                # slide the window
                self.trajectory.pop(0)

    def step(self, player, enemies, bullets, dt):
        """
        One control step:
          - decay epsilon
          - observe state
          - choose action
          - update Q from previous transition
          - move & possibly fire
          - compute shaping reward for this frame

        Returns (current_state, action, reward_step, fired_this_frame)
        """
        # Epsilon decay
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * (self.epsilon_decay ** dt)
        )

        # Observe current state
        current_state = get_state_bin(player, enemies)

        # Choose action under current policy
        action = self.select_action(current_state)

        # RL update for previous transition
        self._do_update(current_state, action)

        reward_step, fired_this_frame, self.time_since_last_shot = apply_agent_action(
            player,
            action,
            enemies,
            bullets,
            dt,
            self.speed,
            self.time_since_last_shot,
            self.bullet_cooldown,
            self.accuracy_focus,
            self.aim_acc_prob,
            self.miss_offset,
            self.bullet_owner,
        )

        # Q-table logging
        self.q_log_timer += dt
        if self.q_log_timer >= UI.q_log_interval:
            self.q_log_timer = 0.0
            print(f"=== Q-table snapshot ({self.name} {self.algo}) ===")
            for i, row in enumerate(self.Q):
                print(f"{self.name} state {i}: {['{:.2f}'.format(v) for v in row]}")

        # update internal prev_* for next step
        self.prev_state  = current_state
        self.prev_action = action
        # prev_reward will be set from outside after environment updates

        return current_state, action, reward_step, fired_this_frame


class BayesianQAgent:
    def __init__(
        self,
        name,
        q_mean,
        q_var,
        gamma,
        use_thompson=True,
        bullet_cooldown=0.10,
        speed=600.0,
        accuracy_focus=1.0,
        aim_acc_prob=1.0,
        miss_offset=0.0,
        bullet_owner="P4",
        epsilon=BAYES_EPSILON,
    ):
        self.name           = name
        self.q_mean         = q_mean
        self.q_var          = q_var
        self.gamma          = gamma
        self.use_thompson   = use_thompson

        self.bullet_cooldown = bullet_cooldown
        self.speed           = speed
        self.accuracy_focus  = accuracy_focus
        self.aim_acc_prob    = aim_acc_prob
        self.miss_offset     = miss_offset
        self.bullet_owner    = bullet_owner

        self.epsilon         = epsilon   # <--- store exploration rate
        self.epsilon_min     = BAYES_EPSILON_MIN
        self.epsilon_decay   = BAYES_EPSILON_DECAY

        self.prev_state  = None
        self.prev_action = None
        self.prev_reward = 0.0

        self.time_since_last_shot = 0.0

    def select_action(self, state):
        """
        Epsilon-greedy over a Bayesian value estimate:
        - with prob epsilon: uniform random action
        - otherwise: Thompson sampling (or greedy on mean)
        """
        # Explicit exploration
        if random.random() < self.epsilon:
            return random.randint(0, N_ACTIONS - 1)

        # Exploit according to Bayesian posterior
        if self.use_thompson:
            # Thompson sampling
            samples = []
            for a in range(N_ACTIONS):
                mu  = self.q_mean[state][a]
                var = self.q_var[state][a]
                std = math.sqrt(max(var, 1e-8))
                samples.append(random.gauss(mu, std))
            max_val = max(samples)
            best_actions = [i for i, v in enumerate(samples) if v == max_val]
            return random.choice(best_actions)
        else:
            # Greedy on posterior mean
            row = self.q_mean[state]
            max_q = max(row)
            best_actions = [i for i, q in enumerate(row) if q == max_q]
            return random.choice(best_actions)

    def step(self, player, enemies, bullets, dt):
        """
        Mirrors RLAgent.step:
          - observe state
          - choose action
          - Bayesian Q update using previous transition
          - move & fire
          - shaping rewards

        Returns (current_state, action, reward_step, fired_this_frame)
        """
        # Decay epsilon over time
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * (self.epsilon_decay ** dt)
        )

        # --- Observe current state ---
        current_state = get_state_bin(player, enemies)

        # --- Choose action ---
        action = self.select_action(current_state)

        # --- Bayesian Q update for previous transition ---
        if self.prev_state is not None and self.prev_action is not None:
            s  = self.prev_state
            a  = self.prev_action
            r  = self.prev_reward
            ns = current_state

            # TD target using posterior mean at next state
            next_row = self.q_mean[ns]
            target   = r + self.gamma * max(next_row)

            bayes_update_q(self.q_mean, self.q_var, s, a, target)

        reward_step, fired_this_frame, self.time_since_last_shot = apply_agent_action(
            player,
            action,
            enemies,
            bullets,
            dt,
            self.speed,
            self.time_since_last_shot,
            self.bullet_cooldown,
            self.accuracy_focus,
            self.aim_acc_prob,
            self.miss_offset,
            self.bullet_owner,
        )

        # Update previous transition info for next step
        self.prev_state  = current_state
        self.prev_action = action

        return current_state, action, reward_step, fired_this_frame


class ScorePopup:
    """Floating +10 / -10 text that drifts upward then disappears."""
    def __init__(self, x, y, text, color):
        self.x = x
        self.y = y
        self.text = text
        self.color = color
        self.age = 0.0
        self.lifetime = 0.8  # seconds
        self.speed_y = -40.0 # px/sec upward

    def update(self, dt):
        self.age += dt
        self.y += self.speed_y * dt

    def is_dead(self):
        return self.age >= self.lifetime

    def draw(self, surf, font):
        img = font.render(self.text, True, self.color)
        rect = img.get_rect(center=(self.x, self.y))
        surf.blit(img, rect)


# ------------------ Drawing helpers ------------------

def load_background():
    """
    Load and scale the desert background to the screen size.
    """
    img = pygame.image.load(ASSETS.background_image).convert()
    img = pygame.transform.scale(img, (DISPLAY.width, DISPLAY.height))
    return img
  

def load_cloud_layer():
    """
    Load a semi-transparent cloud layer and optionally scale it up/down.
    CLOUD_SCALE > 1.0  -> bigger clouds
    CLOUD_SCALE < 1.0  -> smaller clouds

    The resulting image is then scrolled top-to-bottom.
    """
    img = pygame.image.load(ASSETS.cloud_image).convert_alpha()

    # Apply scale factor to the original sprite size
    w, h = img.get_size()
    new_size = (int(w * ASSETS.cloud_scale), int(h * ASSETS.cloud_scale))
    img = pygame.transform.smoothscale(img, new_size)

    return img


def draw_scrolling_clouds_right(surf, cloud_image, offset_y):
    """
    Draw a vertically scrolling cloud image, anchored to the RIGHT edge
    of the screen. The image loops seamlessly in the vertical direction.
    """
    h = cloud_image.get_height()

    # Wrap offset into [0, h)
    offset_y = offset_y % h

    # Two stacked copies to create a continuous scroll
    y1 = -offset_y
    y2 = y1 + h

    # Anchor the RIGHT edge of the cloud image to the RIGHT edge of the screen
    x = DISPLAY.width - cloud_image.get_width()

    surf.blit(cloud_image, (x, int(y1)))
    surf.blit(cloud_image, (x, int(y2)))

  
def load_player_sprite():
    """
    Load the F-22 sprite with alpha (transparency preserved).
    Optionally scale to a nice on-screen size.
    """
    img = pygame.image.load(ASSETS.player_image).convert_alpha()

    # --- optional scaling so it matches your old ~40 px tall triangles ---
    target_height = ASSETS.player_target_height
    h = img.get_height()
    if h != 0:
        scale = target_height / float(h)
        new_size = (int(img.get_width() * scale), int(img.get_height() * scale))
        img = pygame.transform.smoothscale(img, new_size)

    return img


def load_enemy_sprite(flip_vertical=True, target_height=None):
    """
    Load the Su-57 enemy sprite with alpha, scale it, and optionally flip vertically
    so it points 'down' along the screen.
    """
    if target_height is None:
        target_height = ASSETS.enemy_target_height

    img = pygame.image.load(ASSETS.enemy_image).convert_alpha()

    # Scale to desired on-screen height
    h = img.get_height()
    if h != 0 and target_height is not None:
        scale = target_height / float(h)
        new_size = (int(img.get_width() * scale), int(img.get_height() * scale))
        img = pygame.transform.smoothscale(img, new_size)

    # Flip so the nose points downwards (enemies fly down the screen)
    if flip_vertical:
        img = pygame.transform.flip(img, False, True)

    return img


def draw_scrolling_background(surf, bg_image, offset_y):
    """
    Draw a vertically scrolling background image that loops seamlessly.

    offset_y increases over time; the image is tiled so that when one
    copy scrolls off the bottom, the next is already in place above it.
    """
    h = bg_image.get_height()

    # Wrap offset into [0, h)
    offset_y = offset_y % h

    # Draw two copies: one starting at -offset_y, one just below it
    y1 = -offset_y
    y2 = y1 + h

    surf.blit(bg_image, (0, int(y1)))
    surf.blit(bg_image, (0, int(y2)))


def draw_score_hud_p1(surf, font, score, acc_pct, kills, misses, mode, training_mode):
    base_color = (0, 255, 0)
    x = 10
    y = 10

    # P1 score
    text1 = font.render(f"P1 SCORE {score}", True, base_color)
    surf.blit(text1, (x, y))
    y += text1.get_height()

    # P1 kills
    text_k = font.render(f"KILLS {kills}", True, base_color)
    surf.blit(text_k, (x, y))
    y += text_k.get_height()

    # P1 misses (owned enemy escapes)
    text_m = font.render(f"MISSES {misses}", True, base_color)
    surf.blit(text_m, (x, y))
    y += text_m.get_height()

    # P1 accuracy
    text2 = font.render(f"ACC {int(acc_pct)}%", True, base_color)
    surf.blit(text2, (x, y))
    y += text2.get_height() + 4

    # MODE display
    mode_text = font.render(f"MODE: {mode}", True, (200, 200, 200))
    surf.blit(mode_text, (x, y))
    y += mode_text.get_height()

    # Training indicator (only if active)
    if training_mode:
        train_text = font.render("TRAINING MODE (P1 DISABLED)", True, (255, 200, 50))
        surf.blit(train_text, (x, y))


def draw_score_hud_rl_agents(
    surf, font,
    score_p2, acc_p2, kills_p2, misses_p2,
    score_p3, acc_p3, kills_p3, misses_p3,
    score_p4, acc_p4, kills_p4, misses_p4,
    epsilon_p2
):
    # P2 colors (blue-ish)
    p2_color = (0, 200, 255)
    # P3 colors (magenta-ish)
    p3_color = (255, 120, 255)
    # P4 colors (teal-ish)
    p4_color = (0, 255, 180)

    # --- P2 HUD (top-right) ---
    text1_p2 = font.render(f"P2 SCORE {score_p2}", True, p2_color)
    textk_p2 = font.render(f"KILLS {kills_p2}", True, p2_color)
    textm_p2 = font.render(f"MISSES {misses_p2}", True, p2_color)
    text2_p2 = font.render(f"ACC {int(acc_p2)}%", True, p2_color)

    rect1_p2 = text1_p2.get_rect(topright=(DISPLAY.width - 10, 10))
    rectk_p2 = textk_p2.get_rect(topright=(DISPLAY.width - 10, rect1_p2.bottom))
    rectm_p2 = textm_p2.get_rect(topright=(DISPLAY.width - 10, rectk_p2.bottom))
    rect2_p2 = text2_p2.get_rect(topright=(DISPLAY.width - 10, rectm_p2.bottom))

    surf.blit(text1_p2, rect1_p2)
    surf.blit(textk_p2, rectk_p2)
    surf.blit(textm_p2, rectm_p2)
    surf.blit(text2_p2, rect2_p2)

    # --- P3 HUD directly underneath P2 HUD ---
    top_y_p3 = rect2_p2.bottom + 8  # small gap below P2 ACC

    text1_p3 = font.render(f"P3 SCORE {score_p3}", True, p3_color)
    textk_p3 = font.render(f"KILLS {kills_p3}", True, p3_color)
    textm_p3 = font.render(f"MISSES {misses_p3}", True, p3_color)
    text2_p3 = font.render(f"ACC {int(acc_p3)}%", True, p3_color)

    rect1_p3 = text1_p3.get_rect(topright=(DISPLAY.width - 10, top_y_p3))
    rectk_p3 = textk_p3.get_rect(topright=(DISPLAY.width - 10, rect1_p3.bottom))
    rectm_p3 = textm_p3.get_rect(topright=(DISPLAY.width - 10, rectk_p3.bottom))
    rect2_p3 = text2_p3.get_rect(topright=(DISPLAY.width - 10, rectm_p3.bottom))

    surf.blit(text1_p3, rect1_p3)
    surf.blit(textk_p3, rectk_p3)
    surf.blit(textm_p3, rectm_p3)
    surf.blit(text2_p3, rect2_p3)

    # --- P4 HUD directly underneath P3 HUD ---
    top_y_p4 = rect2_p3.bottom + 8

    text1_p4 = font.render(f"P4 SCORE {score_p4}", True, p4_color)
    textk_p4 = font.render(f"KILLS {kills_p4}", True, p4_color)
    textm_p4 = font.render(f"MISSES {misses_p4}", True, p4_color)
    text2_p4 = font.render(f"ACC {int(acc_p4)}%", True, p4_color)

    rect1_p4 = text1_p4.get_rect(topright=(DISPLAY.width - 10, top_y_p4))
    rectk_p4 = textk_p4.get_rect(topright=(DISPLAY.width - 10, rect1_p4.bottom))
    rectm_p4 = textm_p4.get_rect(topright=(DISPLAY.width - 10, rectk_p4.bottom))
    rect2_p4 = text2_p4.get_rect(topright=(DISPLAY.width - 10, rectm_p4.bottom))

    surf.blit(text1_p4, rect1_p4)
    surf.blit(textk_p4, rectk_p4)
    surf.blit(textm_p4, rectm_p4)
    surf.blit(text2_p4, rect2_p4)

    # --- epsilon (P2) display under P4 HUD ---
    eps_text = font.render(f"ε={epsilon_p2:.2f}", True, (180, 180, 255))
    eps_rect = eps_text.get_rect(topright=(DISPLAY.width - 10, rect2_p4.bottom + 8))
    surf.blit(eps_text, eps_rect)



def palette_blue_red_yellow(t):
    """
    Map t in [0,1] to a color:
      0.0 -> blue   (0, 0, 255)
      0.5 -> red    (255, 0, 0)
      1.0 -> yellow (255, 255, 0)
    """
    t = max(0.0, min(1.0, float(t)))

    if t <= 0.5:
        # blue -> red
        u = t / 0.5  # 0..1
        r = int(u * 255)
        g = 0
        b = int((1.0 - u) * 255)
    else:
        # red -> yellow
        u = (t - 0.5) / 0.5  # 0..1
        r = 255
        g = int(u * 255)
        b = 0

    return (r, g, b)


def palette_blue_white_red(v):
    """
    Map correlation v in [-1, 1] to a colour:
        -1 -> blue   (0, 0, 255)
         0 -> white  (255, 255, 255)
        +1 -> red    (255, 0, 0)
    Negative values fade from white to blue,
    positive values fade from white to red.
    """
    v = max(-1.0, min(1.0, float(v)))

    if v < 0.0:
        # v in [-1, 0] -> u in [0, 1]
        u = v + 1.0
        r = int(255 * u)
        g = int(255 * u)
        b = 255
    else:
        # v in [0, 1]
        u = v
        r = 255
        g = int(255 * (1.0 - u))
        b = int(255 * (1.0 - u))

    return (r, g, b)


def draw_q_heatmap(
    surf,
    Q_table,
    top_y,
    left_x,
    width,
    label,
    current_state=None,   # optional: highlight this state
    highlight_color=(255, 255, 255),
):
    """
    Draw Q-table as a heatmap within a specified horizontal region.

    X-axis: state bins
    Y-axis: actions (0..3) with labels
    Color: blue (low) -> red (high)
      top_y : top pixel row of this heatmap
      left_x: left edge of this heatmap region
      width : total width of this heatmap region
    """

    # Flatten Q to find min/max
    q_min = min(min(row) for row in Q_table)
    q_max = max(max(row) for row in Q_table)
    span = q_max - q_min if q_max != q_min else 1.0

    heatmap_top = top_y
    usable_width = width - UI.heatmap_margin_left - UI.heatmap_margin_right

    cell_w = usable_width / float(N_STATE_BINS)
    cell_h = UI.heatmap_height / float(N_ACTIONS)

    # Title/label
    title_font = pygame.font.SysFont(None, 16)
    title_img = title_font.render(label, True, (220, 220, 220))
    surf.blit(title_img, (left_x + UI.heatmap_margin_left,
                          heatmap_top - title_img.get_height()))

    # Action labels
    action_labels = ["L", "R", "S", "F"]
    label_font = pygame.font.SysFont(None, 14)
    label_color = (200, 200, 200)

    for a in range(N_ACTIONS):
        # Action label
        row_top = heatmap_top + a * cell_h
        label_img = label_font.render(action_labels[a], True, label_color)
        label_x = left_x + 4
        label_y = int(row_top + cell_h / 2 - label_img.get_height() / 2)
        surf.blit(label_img, (label_x, label_y))

        # Heatmap cells
        for s in range(N_STATE_BINS):
            q = Q_table[s][a]
            t = (q - q_min) / span  # 0..1
            color = palette_blue_red_yellow(t)

            x = int(left_x + UI.heatmap_margin_left + s * cell_w)
            y = int(row_top)
            w = int(math.ceil(cell_w))
            h = int(math.ceil(cell_h))

            pygame.draw.rect(surf, color, pygame.Rect(x, y, w, h))

    # --- optional: draw vertical guide for current_state ---
    if current_state is not None:
        usable_width = width - UI.heatmap_margin_left - UI.heatmap_margin_right
        cell_w = usable_width / float(N_STATE_BINS)

        # x-position of this state's column
        x_state = int(left_x + UI.heatmap_margin_left + current_state * cell_w)

        # full height of the heatmap band
        y0 = heatmap_top
        y1 = heatmap_top + UI.heatmap_height

        pygame.draw.line(
            surf,
            highlight_color,
            (x_state, y0),
            (x_state, y1),
            1,
        )


def draw_target_box(surf, rect, color, thickness=2):
    """
    Draw a 'corner only' targeting box around rect:
    corners of a square with the middle third of each side cut out.

    The square has the same width as before (rect width + padding),
    but its height is forced to equal that width (true square),
    centered vertically on the enemy.
    """
    x, y, w, h = rect
    pad = 5  # small padding around the enemy

    # Horizontal span (unchanged)
    x0 = x - pad
    x1 = x + w + pad
    width_box = x1 - x0  # side length of the square

    # Vertically center the square on the enemy
    center_y = y + h / 2.0
    y0 = center_y - width_box / 2.0
    y1 = center_y + width_box / 2.0

    # Corner lengths (roughly 1/5 of each side)
    corner_len_x = width_box / 5.0
    corner_len_y = width_box / 5.0  # same, since it's a square

    # Top-left corner
    pygame.draw.line(surf, color, (x0, y0), (x0 + corner_len_x, y0), thickness)
    pygame.draw.line(surf, color, (x0, y0), (x0, y0 + corner_len_y), thickness)

    # Top-right corner
    pygame.draw.line(surf, color, (x1, y0), (x1 - corner_len_x, y0), thickness)
    pygame.draw.line(surf, color, (x1, y0), (x1, y0 + corner_len_y), thickness)

    # Bottom-left corner
    pygame.draw.line(surf, color, (x0, y1), (x0 + corner_len_x, y1), thickness)
    pygame.draw.line(surf, color, (x0, y1), (x0, y1 - corner_len_y), thickness)

    # Bottom-right corner
    pygame.draw.line(surf, color, (x1, y1), (x1 - corner_len_x, y1), thickness)
    pygame.draw.line(surf, color, (x1, y1), (x1, y1 - corner_len_y), thickness)



# ------------------ Main game loop ------------------

def main():
    screen = pygame.display.set_mode((DISPLAY.width, DISPLAY.height))
    pygame.display.set_caption("Team Spirit")
    clock = pygame.time.Clock()
    
    # Load scrolling background
    bg_image = load_background()
    bg_offset = 0.0
    
    # Load cloud layer
    cloud_image = load_cloud_layer()
    cloud_offset = 0.0
    
    # Load F-22 sprite
    player_sprite = load_player_sprite()
    
    # Load Su-57 enemy sprite
    enemy_sprite = load_enemy_sprite()
    
    # Optionally load Q-tables from previous run
    load_q_tables_if_enabled()

    hud_font = pygame.font.SysFont(None, UI.hud_font_size)
    popup_font = pygame.font.SysFont(None, UI.popup_font_size)

    # Player starting Y positions so they sit above the heatmaps
    base_y = UI.play_area_bottom - player_sprite.get_height()

    # sprite width for centering
    sprite_w, _ = player_sprite.get_size()

    # Player 1 (human) – yellow
    player1 = Player(
        DISPLAY.width / 2 - sprite_w / 2,
        base_y,
        color=PLAYER1_COLOR,
        sprite=player_sprite,
    )

    # Player 2 (RL bot) – blue
    player2 = Player(
        DISPLAY.width / 2 + 100,
        base_y - 40,
        color=PLAYER2_COLOR,
        sprite=player_sprite,
    )

    # Player 3 (RL bot #2) – magenta-ish
    player3 = Player(
        DISPLAY.width / 2 - 100,
        base_y - 40,
        color=PLAYER3_COLOR,
        sprite=player_sprite,
    )

    # Player 4 (Bayesian 'ideal observer') – teal-ish
    player4 = Player(
        DISPLAY.width / 2,
        base_y - 80,
        color=PLAYER4_COLOR,
        sprite=player_sprite,
    )


    # ------------------ RL Agents ------------------
    # P2: e.g. Expected SARSA (on-policy, smoother)
    agent_p2 = RLAgent(
        name="P2",
        algo=ALGO_N_STEP_SARSA,   # one of: ALGO_SARSA / ALGO_EXPECTED_SARSA / ALGO_N_STEP_SARSA / ALGO_Q_LEARNING
        Q_table=Q_P2,
        alpha=RL_ALPHA_P2,
        gamma=RL_GAMMA_P2,
        epsilon_start=RL_EPSILON_START_P2,
        epsilon_min=RL_EPSILON_MIN_P2,
        epsilon_decay=RL_EPSILON_DECAY_P2,
        bullet_cooldown=GAME.bot_bullet_cooldown,
        speed=GAME.bot_speed,
        accuracy_focus=GAME.p2_accuracy_focus,
        aim_acc_prob=GAME.p2_aim_accuracy,
        miss_offset=GAME.p2_miss_offset,
        bullet_owner="P2",
        n_step=N_STEP,
    )

    # P3: e.g. Q-learning baseline, accurate aim
    agent_p3 = RLAgent(
        name="P3",
        algo=ALGO_N_STEP_SARSA,       # one of: ALGO_SARSA / ALGO_EXPECTED_SARSA / ALGO_N_STEP_SARSA / ALGO_Q_LEARNING
        Q_table=Q_P3,
        alpha=RL_ALPHA_P3,
        gamma=RL_GAMMA_P3,
        epsilon_start=RL_EPSILON_START_P3,
        epsilon_min=RL_EPSILON_MIN_P3,
        epsilon_decay=RL_EPSILON_DECAY_P3,
        bullet_cooldown=GAME.bot_bullet_cooldown,
        speed=GAME.bot_speed,
        accuracy_focus=P3_ACCURACY_FOCUS,
        aim_acc_prob=1.0,          # perfect aim
        miss_offset=0.0,
        bullet_owner="P3",
        n_step=3,
    )

   # P4: Bayesian-Q "ideal observer"
    agent_p4 = BayesianQAgent(
        name="P4",
        q_mean=Q_BAYES_MEAN,
        q_var=Q_BAYES_VAR,
        gamma=RL_GAMMA_P3,
        use_thompson=True,
        bullet_cooldown=GAME.bot_bullet_cooldown,
        speed=GAME.bot_speed,
        accuracy_focus=P3_ACCURACY_FOCUS,
        aim_acc_prob=1.0,   # perfect aim
        miss_offset=0.0,
        bullet_owner="P4",
        # epsilon=BAYES_EPSILON  # optional, defaults to global
    )

    bullets = []
    enemies = []
    score_popups = []

    enemy_spawn_timer = 0.0
    time_since_last_shot_p1 = 0.0
    corr_history = []

    stats_by_owner = {owner: ScoreState() for owner in OWNER_IDS}
    players_by_owner = {
        "P1": player1,
        "P2": player2,
        "P3": player3,
        "P4": player4,
    }
    bot_states = {
        "P2": BotRuntime("P2", player2, agent_p2, Q_P2),
        "P3": BotRuntime("P3", player3, agent_p3, Q_P3),
        "P4": BotRuntime("P4", player4, agent_p4, Q_BAYES_MEAN),
    }

    mode = MODE_VERSUS
    training_mode = False

    running = True
    while running:
        dt = clock.tick(DISPLAY.fps) / 1000.0

        # -------------- Events --------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_m:
                    mode = MODE_VERSUS if mode == MODE_TEAM else MODE_TEAM
                elif event.key == pygame.K_t:
                    training_mode = not training_mode
                    print(f"Training mode: {training_mode}")

        keys = pygame.key.get_pressed()

        time_since_last_shot_p1, fired_this_frame_p1 = update_player1(
            player1,
            keys,
            bullets,
            dt,
            time_since_last_shot_p1,
            training_mode,
        )
        if fired_this_frame_p1:
            stats_by_owner["P1"].shots += 1

        for owner in BOT_OWNER_IDS:
            bot_state = bot_states[owner]
            bot_state.current_state, bot_state.current_action, bot_state.reward_step, fired_this_frame = bot_state.controller.step(
                bot_state.player,
                enemies,
                bullets,
                dt,
            )
            if fired_this_frame:
                stats_by_owner[owner].shots += 1

        corr_matrix = update_corr_history(
            corr_history,
            [players_by_owner[owner] for owner in OWNER_IDS],
        )


        # -------------- Update world --------------

        # grid_offset -= GRID_SCROLL_SPEED * dt
        
        # Update background scroll offset (negative = image moves down)
        bg_offset -= GAME.grid_scroll_speed * dt
        cloud_offset -= GAME.cloud_scroll_speed * dt

        bullets, bullet_reward_updates = update_projectiles(
            bullets,
            {owner: bot_states[owner].controller.accuracy_focus for owner in BOT_OWNER_IDS},
            dt,
        )
        for owner, reward_delta in bullet_reward_updates.items():
            bot_states[owner].reward_step += reward_delta

        enemy_spawn_timer += dt
        if enemy_spawn_timer >= GAME.enemy_spawn_interval:
            enemy_spawn_timer = 0.0
            ex = sample_spawn_x_gaussian_mixture()
            enemies.append(Enemy(ex, -40, enemy_sprite))

        enemies, escaped_enemies = update_enemies(enemies, dt)
        escape_reward_updates = apply_escape_consequences(
            escaped_enemies,
            stats_by_owner,
            bot_states,
            players_by_owner,
            training_mode,
            mode,
            score_popups,
        )
        for owner, reward_delta in escape_reward_updates.items():
            bot_states[owner].reward_step += reward_delta

        collision_reward_updates = resolve_bullet_collisions(
            bullets,
            enemies,
            stats_by_owner,
            bot_states,
            training_mode,
            score_popups,
        )
        for owner, reward_delta in collision_reward_updates.items():
            bot_states[owner].reward_step += reward_delta

        # Update score popups
        for p in score_popups:
            p.update(dt)
        score_popups = [p for p in score_popups if not p.is_dead()]

        # Finalise reward for this frame (used in next frame's RL update)
        for owner in BOT_OWNER_IDS:
            bot_states[owner].controller.prev_reward = bot_states[owner].reward_step


        # -------------- Draw --------------
        # draw_scrolling_grid(screen, grid_offset)
        draw_scrolling_background(screen, bg_image, bg_offset)
        
        # overlay cloud layer (semi-transparent, scrolls slowly)
        draw_scrolling_clouds_right(screen, cloud_image, cloud_offset)

        for b in bullets:
            b.draw(screen)

        for e in enemies:
            e.draw(screen)

        # --- RL agent targeting boxes ---
        for owner in BOT_OWNER_IDS:
            target = choose_target_enemy(bot_states[owner].player, enemies)
            if target is not None:
                draw_target_box(screen, target.rect, bot_states[owner].color, thickness=2)

        # Players
        if not training_mode:
            player1.draw(screen)
        for owner in BOT_OWNER_IDS:
            bot_states[owner].player.draw(screen)


        draw_score_hud_p1(
            screen, hud_font,
            stats_by_owner["P1"].score, stats_by_owner["P1"].accuracy, stats_by_owner["P1"].hits, stats_by_owner["P1"].misses,
            mode, training_mode
        )


        draw_score_hud_rl_agents(
            screen, hud_font,
            stats_by_owner["P2"].score, stats_by_owner["P2"].accuracy, stats_by_owner["P2"].hits, stats_by_owner["P2"].misses,
            stats_by_owner["P3"].score, stats_by_owner["P3"].accuracy, stats_by_owner["P3"].hits, stats_by_owner["P3"].misses,
            stats_by_owner["P4"].score, stats_by_owner["P4"].accuracy, stats_by_owner["P4"].hits, stats_by_owner["P4"].misses,
            agent_p2.epsilon
        )

        # Correlation heatmap for P2/P3/P4 x-positions
        # Place it on the right side, under the RL HUDs
        corr_size = 81  # px, tune as desired
        corr_margin_right = 5
        corr_margin_top = 400  # vertical offset from top; adjust to sit under HUDs

        corr_top_left_x = DISPLAY.width - corr_margin_right - corr_size
        corr_top_left_y = corr_margin_top

        draw_corr_heatmap(
            screen,
            corr_matrix,
            corr_top_left_x,
            corr_top_left_y,
            corr_size,
        )


        for p in score_popups:
            p.draw(screen, popup_font)


        # Q heatmaps: P2, P3, P4 side-by-side across the bottom band
        heatmap_top = DISPLAY.height - UI.heatmap_height

        third_width = DISPLAY.width // 3

        draw_q_heatmap(
            screen, bot_states["P2"].q_table,
            heatmap_top, 0, third_width,
            bot_states["P2"].heatmap_label,
            current_state=bot_states["P2"].current_state,
            highlight_color=bot_states["P2"].color,
        )

        draw_q_heatmap(
            screen, bot_states["P3"].q_table,
            heatmap_top, third_width, third_width,
            bot_states["P3"].heatmap_label,
            current_state=bot_states["P3"].current_state,
            highlight_color=bot_states["P3"].color,
        )

        draw_q_heatmap(
            screen, bot_states["P4"].q_table,
            heatmap_top, 2 * third_width,
            DISPLAY.width - 2 * third_width,
            bot_states["P4"].heatmap_label,
            current_state=bot_states["P4"].current_state,
            highlight_color=bot_states["P4"].color,
        )


        pygame.display.flip()

    # On exit, save the latest Q-tables
    save_q_tables()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
