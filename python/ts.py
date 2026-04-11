import pygame
import random
import sys
import math
import os
import json
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

pygame.init()

# ------------------ Config ------------------

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


@dataclass(frozen=True)
class DisplayConfig:
    width: int = 1200
    height: int = 740
    fps: int = 60


@dataclass(frozen=True)
class AssetConfig:
    background_image: Path = REPO_ROOT / "python" / "images" / "bg_forest_dark.png"
    player_image: Path = REPO_ROOT / "python" / "images" / "f22.png"
    enemy_image: Path = REPO_ROOT / "python" / "images" / "su57.png"
    cloud_image: Path = REPO_ROOT / "python" / "images" / "cloud5.png"
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
    start_from_saved_q: bool = True
    json_dir: Path = REPO_ROOT / "json"
    q_save_file_p2: Path = REPO_ROOT / "json" / "q_table_p2.json"
    q_save_file_p3: Path = REPO_ROOT / "json" / "q_table_p3.json"
    q_save_file_p3_b: Path = REPO_ROOT / "json" / "q_table_p3_b.json"
    q_save_file_p4_mean: Path = REPO_ROOT / "json" / "q_table_p4_mean.json"
    q_save_file_p4_var: Path = REPO_ROOT / "json" / "q_table_p4_var.json"
    q_save_file_p5: Path = REPO_ROOT / "json" / "q_table_p5.json"
    q_save_file_p5_b: Path = REPO_ROOT / "json" / "q_table_p5_b.json"
    q_save_file_p6: Path = REPO_ROOT / "json" / "q_table_p6.json"
    q_save_file_p6_b: Path = REPO_ROOT / "json" / "q_table_p6_b.json"
    q_save_file_p7: Path = REPO_ROOT / "json" / "q_table_p7.json"
    q_save_file_p7_b: Path = REPO_ROOT / "json" / "q_table_p7_b.json"


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
PLAYER5_COLOR = (255, 210, 80)   # P5: amber-ish
PLAYER6_COLOR = (180, 160, 255)  # P6: violet-ish
PLAYER7_COLOR = (120, 255, 120)  # P7: mint-ish
ENEMY_COLOR   = (255, 80, 80) # red
# ENEMY_COLOR   = (0, 0, 0) # black
BULLET_COLOR  = (0, 255, 0)

OWNER_COLORS = {
    "P1": PLAYER1_COLOR,
    "P2": PLAYER2_COLOR,
    "P3": PLAYER3_COLOR,
    "P4": PLAYER4_COLOR,
    "P5": PLAYER5_COLOR,
    "P6": PLAYER6_COLOR,
    "P7": PLAYER7_COLOR,
}

OWNER_IDS = ("P1", "P2", "P3", "P4", "P5", "P6")
BOT_OWNER_IDS = ("P2", "P3", "P4", "P5", "P6")


P3_ACCURACY_FOCUS = GAME.p2_accuracy_focus

# Scoring modes
MODE_TEAM   = "TEAM"
MODE_VERSUS = "VERSUS"

# Players should not be able to move into the Q-heatmap area at the bottom.
# Total heatmap height = UI.heatmap_height.


# ------------------ RL algorithms ------------------
ALGO_Q_LEARNING     = "Q_LEARNING"
ALGO_SARSA          = "SARSA"
ALGO_SARSA_LAMBDA   = "SARSA_LAMBDA"
ALGO_WATKINS_Q_LAMBDA = "WATKINS_Q_LAMBDA"
ALGO_DOUBLE_Q       = "DOUBLE_Q"
ALGO_EXPECTED_SARSA = "EXPECTED_SARSA"
ALGO_N_STEP_SARSA   = "N_STEP_SARSA"
ALGO_DIFFERENTIAL_SARSA = "DIFFERENTIAL_SARSA"
ALGO_R_LEARNING     = "R_LEARNING"
ALGO_DYNA_Q         = "DYNA_Q"
ALGO_UCB_Q          = "UCB_Q"
ALGO_OPTIMISTIC_Q   = "OPTIMISTIC_Q"
ALGO_HYSTERETIC_Q   = "HYSTERETIC_Q"
ALGO_LENIENT_Q      = "LENIENT_Q"
ALGO_BAYES_Q        = "BAYES_Q"   

# N-step horizon for N-STEP SARSA
N_STEP = 2

# --------------- RL config (RL players) ---------------
REL_STATE_BINS = 25   # horizontal target bins
TIME_TO_ESCAPE_BINS = 3
LOCK_CONFIDENCE_BINS = 2
NO_TARGET_STATE = REL_STATE_BINS * TIME_TO_ESCAPE_BINS * LOCK_CONFIDENCE_BINS
N_STATE_BINS = NO_TARGET_STATE + 1

A_MOVE_LEFT, A_MOVE_RIGHT, A_STAY, A_FIRE = range(4)
N_ACTIONS = 4


# ---------- P2 RL params ----------
RL_ALPHA_P2 = 0.20   # learning rate 
RL_GAMMA_P2 = 0.99   # discount rate

RL_EPSILON_START_P2 = 0.99   # initial exploration 
RL_EPSILON_MIN_P2   = 0.01   # floor 
RL_EPSILON_DECAY_P2 = 0.95  # multiplicative decay per second 
RL_LAMBDA_P2        = 0.85
RL_AVG_REWARD_STEP_SIZE_P2 = 0.02
RL_DYNA_PLANNING_STEPS_P2  = 8
RL_UCB_EXPLORATION_P2 = 2.0
RL_OPTIMISTIC_INIT_P2 = 5.0
RL_HYSTERETIC_NEGATIVE_RATIO_P2 = 0.1
RL_LENIENCY_TEMP_INIT_P2 = 1.5
RL_LENIENCY_TEMP_DECAY_P2 = 0.995


# ---------- P3 RL params ----------
RL_ALPHA_P3 = 0.20   # learning rate 
RL_GAMMA_P3 = 0.95   # discount rate

RL_EPSILON_START_P3 = 0.99   # initial exploration
RL_EPSILON_MIN_P3   = 0.01   # floor
RL_EPSILON_DECAY_P3 = 0.98  # multiplicative decay per second
RL_LAMBDA_P3        = 0.85
RL_AVG_REWARD_STEP_SIZE_P3 = 0.02
RL_DYNA_PLANNING_STEPS_P3  = 8
RL_UCB_EXPLORATION_P3 = 2.0
RL_OPTIMISTIC_INIT_P3 = 5.0
RL_HYSTERETIC_NEGATIVE_RATIO_P3 = 0.1
RL_LENIENCY_TEMP_INIT_P3 = 1.5
RL_LENIENCY_TEMP_DECAY_P3 = 0.995


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
MOVE_AWAY_PENALTY       = -0.1
MOVE_WHILE_ALIGNED_PENALTY = -0.08
MOVE_PROGRESS_WEIGHT    = 0.35
MOVE_DIRECTION_SWITCH_PENALTY = -0.08
STEP_REWARD             = 0.05
CENTERING_WEIGHT        = 0.15

# Target selection weights / hysteresis
TARGET_THREAT_WEIGHT       = 0.55
TARGET_ALIGNMENT_WEIGHT    = 0.35
TARGET_SHOT_WINDOW_WEIGHT  = 0.10
TARGET_CROWDING_WEIGHT     = 0.25
TARGET_SWITCH_SCORE_MARGIN = 0.08
TARGET_COORD_ASSIGNMENT_BONUS = 0.20
TARGET_LOCK_CONFIDENCE_SECONDS = 1.0
FIRE_GATE_ALIGNMENT_MULT = 1.2
FIRE_GATE_MIN_TARGET_SCORE = 0.10


# Primary tabular value tables plus an auxiliary table for Double Q-learning.
Q_P2 = [[0.0 for _ in range(N_ACTIONS)] for _ in range(N_STATE_BINS)]
Q_P3 = [[0.0 for _ in range(N_ACTIONS)] for _ in range(N_STATE_BINS)]
Q_P3_B = [[0.0 for _ in range(N_ACTIONS)] for _ in range(N_STATE_BINS)]
Q_P5 = [[0.0 for _ in range(N_ACTIONS)] for _ in range(N_STATE_BINS)]
Q_P5_B = [[0.0 for _ in range(N_ACTIONS)] for _ in range(N_STATE_BINS)]
Q_P6 = [[0.0 for _ in range(N_ACTIONS)] for _ in range(N_STATE_BINS)]
Q_P6_B = [[0.0 for _ in range(N_ACTIONS)] for _ in range(N_STATE_BINS)]
Q_P7 = [[0.0 for _ in range(N_ACTIONS)] for _ in range(N_STATE_BINS)]
Q_P7_B = [[0.0 for _ in range(N_ACTIONS)] for _ in range(N_STATE_BINS)]

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

NEXT_ENEMY_ID = 1


def allocate_enemy_id():
    global NEXT_ENEMY_ID
    enemy_id = NEXT_ENEMY_ID
    NEXT_ENEMY_ID += 1
    return enemy_id

AGENT_FAMILY_TABULAR = "TABULAR"
AGENT_FAMILY_BAYES = "BAYES"

AGENT_VARIANT_OVERRIDES = {
    "Q_LEARNING": {
        "family": AGENT_FAMILY_TABULAR,
        "algo": ALGO_Q_LEARNING,
    },
    "SARSA": {
        "family": AGENT_FAMILY_TABULAR,
        "algo": ALGO_SARSA,
    },
    "SARSA_LAMBDA": {
        "family": AGENT_FAMILY_TABULAR,
        "algo": ALGO_SARSA_LAMBDA,
    },
    "WATKINS_Q_LAMBDA": {
        "family": AGENT_FAMILY_TABULAR,
        "algo": ALGO_WATKINS_Q_LAMBDA,
    },
    "DOUBLE_Q": {
        "family": AGENT_FAMILY_TABULAR,
        "algo": ALGO_DOUBLE_Q,
    },
    "EXPECTED_SARSA": {
        "family": AGENT_FAMILY_TABULAR,
        "algo": ALGO_EXPECTED_SARSA,
    },
    "N_STEP_SARSA": {
        "family": AGENT_FAMILY_TABULAR,
        "algo": ALGO_N_STEP_SARSA,
        "n_step": N_STEP,
    },
    "DIFFERENTIAL_SARSA": {
        "family": AGENT_FAMILY_TABULAR,
        "algo": ALGO_DIFFERENTIAL_SARSA,
    },
    "R_LEARNING": {
        "family": AGENT_FAMILY_TABULAR,
        "algo": ALGO_R_LEARNING,
    },
    "DYNA_Q": {
        "family": AGENT_FAMILY_TABULAR,
        "algo": ALGO_DYNA_Q,
    },
    "UCB_Q_LEARNING": {
        "family": AGENT_FAMILY_TABULAR,
        "algo": ALGO_UCB_Q,
    },
    "OPTIMISTIC_Q_LEARNING": {
        "family": AGENT_FAMILY_TABULAR,
        "algo": ALGO_OPTIMISTIC_Q,
    },
    "HYSTERETIC_Q_LEARNING": {
        "family": AGENT_FAMILY_TABULAR,
        "algo": ALGO_HYSTERETIC_Q,
    },
    "LENIENT_Q_LEARNING": {
        "family": AGENT_FAMILY_TABULAR,
        "algo": ALGO_LENIENT_Q,
    },
    "BAYES_Q": {
        "family": AGENT_FAMILY_BAYES,
        "algo": ALGO_BAYES_Q,
        "use_thompson": True,
    },
}

BOT_BASE_CONFIGS = {
    "P2": {
        "family": AGENT_FAMILY_TABULAR,
        "q_table_name": "Q_P2",
        "secondary_q_name": None,
        "alpha": RL_ALPHA_P2,
        "gamma": RL_GAMMA_P2,
        "epsilon_start": RL_EPSILON_START_P2,
        "epsilon_min": RL_EPSILON_MIN_P2,
        "epsilon_decay": RL_EPSILON_DECAY_P2,
        "trace_lambda": RL_LAMBDA_P2,
        "n_step": N_STEP,
        "avg_reward_step_size": RL_AVG_REWARD_STEP_SIZE_P2,
        "dyna_planning_steps": RL_DYNA_PLANNING_STEPS_P2,
        "ucb_exploration": RL_UCB_EXPLORATION_P2,
        "optimistic_init_value": RL_OPTIMISTIC_INIT_P2,
        "hysteretic_negative_ratio": RL_HYSTERETIC_NEGATIVE_RATIO_P2,
        "leniency_temperature_init": RL_LENIENCY_TEMP_INIT_P2,
        "leniency_temperature_decay": RL_LENIENCY_TEMP_DECAY_P2,
        "bullet_cooldown": GAME.bot_bullet_cooldown,
        "speed": GAME.bot_speed,
        "accuracy_focus": GAME.p2_accuracy_focus,
        "aim_acc_prob": GAME.p2_aim_accuracy,
        "miss_offset": GAME.p2_miss_offset,
        "bullet_owner": "P2",
    },
    "P3": {
        "family": AGENT_FAMILY_TABULAR,
        "q_table_name": "Q_P3",
        "secondary_q_name": "Q_P3_B",
        "alpha": RL_ALPHA_P3,
        "gamma": RL_GAMMA_P3,
        "epsilon_start": RL_EPSILON_START_P3,
        "epsilon_min": RL_EPSILON_MIN_P3,
        "epsilon_decay": RL_EPSILON_DECAY_P3,
        "trace_lambda": RL_LAMBDA_P3,
        "n_step": N_STEP,
        "avg_reward_step_size": RL_AVG_REWARD_STEP_SIZE_P3,
        "dyna_planning_steps": RL_DYNA_PLANNING_STEPS_P3,
        "ucb_exploration": RL_UCB_EXPLORATION_P3,
        "optimistic_init_value": RL_OPTIMISTIC_INIT_P3,
        "hysteretic_negative_ratio": RL_HYSTERETIC_NEGATIVE_RATIO_P3,
        "leniency_temperature_init": RL_LENIENCY_TEMP_INIT_P3,
        "leniency_temperature_decay": RL_LENIENCY_TEMP_DECAY_P3,
        "bullet_cooldown": GAME.bot_bullet_cooldown,
        "speed": GAME.bot_speed,
        "accuracy_focus": P3_ACCURACY_FOCUS,
        "aim_acc_prob": 1.0,
        "miss_offset": 0.0,
        "bullet_owner": "P3",
    },
    "P4": {
        "family": AGENT_FAMILY_BAYES,
        "q_mean_name": "Q_BAYES_MEAN",
        "q_var_name": "Q_BAYES_VAR",
        "gamma": RL_GAMMA_P3,
        "use_thompson": True,
        "epsilon_start": BAYES_EPSILON_START,
        "epsilon_min": BAYES_EPSILON_MIN,
        "epsilon_decay": BAYES_EPSILON_DECAY,
        "bullet_cooldown": GAME.bot_bullet_cooldown,
        "speed": GAME.bot_speed,
        "accuracy_focus": P3_ACCURACY_FOCUS,
        "aim_acc_prob": 1.0,
        "miss_offset": 0.0,
        "bullet_owner": "P4",
    },
    "P5": {
        "family": AGENT_FAMILY_TABULAR,
        "q_table_name": "Q_P5",
        "secondary_q_name": "Q_P5_B",
        "alpha": RL_ALPHA_P2,
        "gamma": RL_GAMMA_P2,
        "epsilon_start": RL_EPSILON_START_P2,
        "epsilon_min": RL_EPSILON_MIN_P2,
        "epsilon_decay": RL_EPSILON_DECAY_P2,
        "trace_lambda": RL_LAMBDA_P2,
        "n_step": N_STEP,
        "avg_reward_step_size": RL_AVG_REWARD_STEP_SIZE_P2,
        "dyna_planning_steps": RL_DYNA_PLANNING_STEPS_P2,
        "ucb_exploration": RL_UCB_EXPLORATION_P2,
        "optimistic_init_value": RL_OPTIMISTIC_INIT_P2,
        "hysteretic_negative_ratio": RL_HYSTERETIC_NEGATIVE_RATIO_P2,
        "leniency_temperature_init": RL_LENIENCY_TEMP_INIT_P2,
        "leniency_temperature_decay": RL_LENIENCY_TEMP_DECAY_P2,
        "bullet_cooldown": GAME.bot_bullet_cooldown,
        "speed": GAME.bot_speed,
        "accuracy_focus": GAME.p2_accuracy_focus,
        "aim_acc_prob": GAME.p2_aim_accuracy,
        "miss_offset": GAME.p2_miss_offset,
        "bullet_owner": "P5",
    },
    "P6": {
        "family": AGENT_FAMILY_TABULAR,
        "q_table_name": "Q_P6",
        "secondary_q_name": "Q_P6_B",
        "alpha": RL_ALPHA_P3,
        "gamma": RL_GAMMA_P3,
        "epsilon_start": RL_EPSILON_START_P3,
        "epsilon_min": RL_EPSILON_MIN_P3,
        "epsilon_decay": RL_EPSILON_DECAY_P3,
        "trace_lambda": RL_LAMBDA_P3,
        "n_step": N_STEP,
        "avg_reward_step_size": RL_AVG_REWARD_STEP_SIZE_P3,
        "dyna_planning_steps": RL_DYNA_PLANNING_STEPS_P3,
        "ucb_exploration": RL_UCB_EXPLORATION_P3,
        "optimistic_init_value": RL_OPTIMISTIC_INIT_P3,
        "hysteretic_negative_ratio": RL_HYSTERETIC_NEGATIVE_RATIO_P3,
        "leniency_temperature_init": RL_LENIENCY_TEMP_INIT_P3,
        "leniency_temperature_decay": RL_LENIENCY_TEMP_DECAY_P3,
        "bullet_cooldown": GAME.bot_bullet_cooldown,
        "speed": GAME.bot_speed,
        "accuracy_focus": P3_ACCURACY_FOCUS,
        "aim_acc_prob": 1.0,
        "miss_offset": 0.0,
        "bullet_owner": "P6",
    },
    "P7": {
        "family": AGENT_FAMILY_TABULAR,
        "q_table_name": "Q_P7",
        "secondary_q_name": "Q_P7_B",
        "alpha": RL_ALPHA_P3,
        "gamma": RL_GAMMA_P3,
        "epsilon_start": RL_EPSILON_START_P3,
        "epsilon_min": RL_EPSILON_MIN_P3,
        "epsilon_decay": RL_EPSILON_DECAY_P3,
        "trace_lambda": RL_LAMBDA_P3,
        "n_step": N_STEP,
        "avg_reward_step_size": RL_AVG_REWARD_STEP_SIZE_P3,
        "dyna_planning_steps": RL_DYNA_PLANNING_STEPS_P3,
        "ucb_exploration": RL_UCB_EXPLORATION_P3,
        "optimistic_init_value": RL_OPTIMISTIC_INIT_P3,
        "hysteretic_negative_ratio": RL_HYSTERETIC_NEGATIVE_RATIO_P3,
        "leniency_temperature_init": RL_LENIENCY_TEMP_INIT_P3,
        "leniency_temperature_decay": RL_LENIENCY_TEMP_DECAY_P3,
        "bullet_cooldown": GAME.bot_bullet_cooldown,
        "speed": GAME.bot_speed,
        "accuracy_focus": P3_ACCURACY_FOCUS,
        "aim_acc_prob": 1.0,
        "miss_offset": 0.0,
        "bullet_owner": "P7",
    },
}

ACTIVE_AGENT_VARIANTS = {
    "P2": "SARSA_LAMBDA",
    "P3": "WATKINS_Q_LAMBDA",
    "P4": "BAYES_Q",
    "P5": "DIFFERENTIAL_SARSA",
    "P6": "DYNA_Q",
}


def resolve_active_agent_config(owner):
    base = BOT_BASE_CONFIGS.get(owner)
    if base is None:
        raise ValueError(f"No base config for owner={owner}")

    variant_name = ACTIVE_AGENT_VARIANTS.get(owner)
    if variant_name is None:
        raise ValueError(f"No active variant selected for owner={owner}")

    variant = AGENT_VARIANT_OVERRIDES.get(variant_name)
    if variant is None:
        raise ValueError(f"Unknown variant '{variant_name}' for owner={owner}")

    merged = dict(base)
    if merged["family"] != variant["family"]:
        raise ValueError(
            f"Variant '{variant_name}' has family={variant['family']} but owner {owner} expects family={merged['family']}"
        )

    merged.update(variant)
    merged["variant_name"] = variant_name
    return merged


def _resolve_table_by_name(table_name):
    table = globals().get(table_name)
    if table is None:
        raise ValueError(f"Unknown table '{table_name}'")
    return table


def build_agent_controller(owner):
    cfg = resolve_active_agent_config(owner)

    if cfg["family"] == AGENT_FAMILY_TABULAR:
        secondary_q = None
        if cfg["algo"] == ALGO_DOUBLE_Q:
            secondary_q_name = cfg.get("secondary_q_name")
            if not secondary_q_name:
                raise ValueError(f"Double Q variant for {owner} requires secondary_q_name.")
            secondary_q = _resolve_table_by_name(secondary_q_name)

        return RLAgent(
            name=owner,
            algo=cfg["algo"],
            Q_table=_resolve_table_by_name(cfg["q_table_name"]),
            alpha=cfg["alpha"],
            gamma=cfg["gamma"],
            epsilon_start=cfg["epsilon_start"],
            epsilon_min=cfg["epsilon_min"],
            epsilon_decay=cfg["epsilon_decay"],
            bullet_cooldown=cfg["bullet_cooldown"],
            speed=cfg["speed"],
            accuracy_focus=cfg["accuracy_focus"],
            aim_acc_prob=cfg["aim_acc_prob"],
            miss_offset=cfg["miss_offset"],
            bullet_owner=cfg["bullet_owner"],
            n_step=cfg.get("n_step", N_STEP),
            trace_lambda=cfg.get("trace_lambda", 0.0),
            secondary_q_table=secondary_q,
            avg_reward_step_size=cfg.get("avg_reward_step_size", 0.01),
            dyna_planning_steps=cfg.get("dyna_planning_steps", 0),
            ucb_exploration=cfg.get("ucb_exploration", 1.0),
            optimistic_init_value=cfg.get("optimistic_init_value", 5.0),
            hysteretic_negative_ratio=cfg.get("hysteretic_negative_ratio", 0.1),
            leniency_temperature_init=cfg.get("leniency_temperature_init", 1.5),
            leniency_temperature_decay=cfg.get("leniency_temperature_decay", 0.995),
        )

    return BayesianQAgent(
        name=owner,
        q_mean=_resolve_table_by_name(cfg["q_mean_name"]),
        q_var=_resolve_table_by_name(cfg["q_var_name"]),
        gamma=cfg["gamma"],
        use_thompson=cfg.get("use_thompson", True),
        bullet_cooldown=cfg["bullet_cooldown"],
        speed=cfg["speed"],
        accuracy_focus=cfg["accuracy_focus"],
        aim_acc_prob=cfg["aim_acc_prob"],
        miss_offset=cfg["miss_offset"],
        bullet_owner=cfg["bullet_owner"],
        epsilon=cfg["epsilon_start"],
        epsilon_min=cfg["epsilon_min"],
        epsilon_decay=cfg["epsilon_decay"],
    )


def log_active_agent_lineup():
    print("Active bot lineup:")
    for owner in BOT_OWNER_IDS:
        cfg = resolve_active_agent_config(owner)
        print(f"  {owner}: {cfg['variant_name']}")


def _write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, temp_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except Exception:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        raise


def save_q_tables():
    """Save tabular and Bayesian value tables to disk as JSON."""
    try:
        save_id = uuid.uuid4().hex
        payloads = {
            PERSISTENCE.q_save_file_p2: {"save_id": save_id, "table": Q_P2},
            PERSISTENCE.q_save_file_p3: {"save_id": save_id, "table": Q_P3},
            PERSISTENCE.q_save_file_p3_b: {"save_id": save_id, "table": Q_P3_B},
            PERSISTENCE.q_save_file_p4_mean: {"save_id": save_id, "table": Q_BAYES_MEAN},
            PERSISTENCE.q_save_file_p4_var: {"save_id": save_id, "table": Q_BAYES_VAR},
            PERSISTENCE.q_save_file_p5: {"save_id": save_id, "table": Q_P5},
            PERSISTENCE.q_save_file_p5_b: {"save_id": save_id, "table": Q_P5_B},
            PERSISTENCE.q_save_file_p6: {"save_id": save_id, "table": Q_P6},
            PERSISTENCE.q_save_file_p6_b: {"save_id": save_id, "table": Q_P6_B},
            PERSISTENCE.q_save_file_p7: {"save_id": save_id, "table": Q_P7},
            PERSISTENCE.q_save_file_p7_b: {"save_id": save_id, "table": Q_P7_B},
        }
        for path, payload in payloads.items():
            _write_json_atomic(path, payload)
        print("Saved value tables to disk.")
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
    If saved-Q loading is enabled and files exist, load the tabular and Bayesian tables from disk.
    Otherwise leave them as freshly initialised.
    """
    global Q_P2, Q_P3, Q_P3_B, Q_BAYES_MEAN, Q_BAYES_VAR, Q_P5, Q_P5_B, Q_P6, Q_P6_B, Q_P7, Q_P7_B

    if not PERSISTENCE.start_from_saved_q:
        print(f"Saved Q-table loading disabled for {PERSISTENCE.json_dir}/; starting from fresh Q-tables.")
        return

    table_paths = {
        "Q_P2": PERSISTENCE.q_save_file_p2,
        "Q_P3": PERSISTENCE.q_save_file_p3,
        "Q_P3_B": PERSISTENCE.q_save_file_p3_b,
        "Q_BAYES_MEAN": PERSISTENCE.q_save_file_p4_mean,
        "Q_BAYES_VAR": PERSISTENCE.q_save_file_p4_var,
        "Q_P5": PERSISTENCE.q_save_file_p5,
        "Q_P5_B": PERSISTENCE.q_save_file_p5_b,
        "Q_P6": PERSISTENCE.q_save_file_p6,
        "Q_P6_B": PERSISTENCE.q_save_file_p6_b,
        "Q_P7": PERSISTENCE.q_save_file_p7,
        "Q_P7_B": PERSISTENCE.q_save_file_p7_b,
    }

    missing_paths = [path for path in table_paths.values() if not path.exists()]
    if missing_paths:
        print(f"No complete saved Q-table set found in {PERSISTENCE.json_dir}/; starting from fresh Q-tables.")
        return

    loaded_tables = {}
    save_ids = set()
    saw_transactional_format = False

    try:
        for name, path in table_paths.items():
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)

            if isinstance(payload, dict) and "table" in payload:
                saw_transactional_format = True
                save_id = payload.get("save_id")
                table = payload["table"]
                if not isinstance(save_id, str) or not save_id:
                    print(f"Saved {name} is missing its save identifier; starting from fresh Q-tables.")
                    return
                save_ids.add(save_id)
            else:
                table = payload

            if not _shape_ok(table):
                print(f"Saved {name} has incompatible shape; starting from fresh Q-tables.")
                return

            loaded_tables[name] = table

        if saw_transactional_format and len(save_ids) != 1:
            print("Saved Q-table files do not belong to the same save set; starting from fresh Q-tables.")
            return

        Q_P2 = loaded_tables["Q_P2"]
        Q_P3 = loaded_tables["Q_P3"]
        Q_P3_B = loaded_tables["Q_P3_B"]
        Q_BAYES_MEAN = loaded_tables["Q_BAYES_MEAN"]
        Q_BAYES_VAR = loaded_tables["Q_BAYES_VAR"]
        Q_P5 = loaded_tables["Q_P5"]
        Q_P5_B = loaded_tables["Q_P5_B"]
        Q_P6 = loaded_tables["Q_P6"]
        Q_P6_B = loaded_tables["Q_P6_B"]
        Q_P7 = loaded_tables["Q_P7"]
        Q_P7_B = loaded_tables["Q_P7_B"]
        print("Loaded Q-tables from disk.")

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



def draw_corr_heatmap(surf, corr, top_left_x, top_left_y, size, labels=None):
    """
    Draw an NxN correlation heatmap in a square region.

    corr: NxN list of floats in [-1, 1]
    Color map: -1 -> blue, 0 -> blackish, +1 -> red
    Rows/cols correspond to the provided labels.
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
    if labels is None:
        labels = [f"P{i + 1}" for i in range(dim)]
    else:
        labels = labels[:dim]
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


def rl_update_differential_sarsa(
    Q_table,
    alpha,
    avg_reward_step_size,
    avg_reward_estimate,
    prev_state,
    prev_action,
    reward,
    next_state,
    next_action,
):
    """
    Differential SARSA for continuing tasks:
        delta = r - r_bar + Q(s',a') - Q(s,a)
        Q(s,a) <- Q(s,a) + alpha * delta
        r_bar  <- r_bar + beta * delta
    """
    if (
        prev_state is None
        or prev_action is None
        or next_state is None
        or next_action is None
    ):
        return avg_reward_estimate

    q_old = Q_table[prev_state][prev_action]
    q_next = Q_table[next_state][next_action]
    delta = reward - avg_reward_estimate + q_next - q_old

    Q_table[prev_state][prev_action] = q_old + alpha * delta
    return avg_reward_estimate + avg_reward_step_size * delta


def rl_update_r_learning(
    Q_table,
    alpha,
    avg_reward_step_size,
    avg_reward_estimate,
    prev_state,
    prev_action,
    reward,
    next_state,
):
    """
    R-learning (average-reward off-policy control):
        delta = r - r_bar + max_a' Q(s',a') - Q(s,a)
        Q(s,a) <- Q(s,a) + alpha * delta
        r_bar  <- r_bar + beta * delta   (on greedy updates)
    """
    if prev_state is None or prev_action is None or next_state is None:
        return avg_reward_estimate

    q_old = Q_table[prev_state][prev_action]
    max_next = max(Q_table[next_state])
    delta = reward - avg_reward_estimate + max_next - q_old

    Q_table[prev_state][prev_action] = q_old + alpha * delta

    prev_row = Q_table[prev_state]
    if prev_row[prev_action] == max(prev_row):
        return avg_reward_estimate + avg_reward_step_size * delta
    return avg_reward_estimate


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


def td_update_from_target(Q_table, alpha, state, action, target):
    q_old = Q_table[state][action]
    Q_table[state][action] = q_old + alpha * (target - q_old)


def greedy_action_from_values(values):
    max_q = max(values)
    best_actions = [i for i, q in enumerate(values) if q == max_q]
    return random.choice(best_actions)


def rl_update_double_q(q_a, q_b, alpha, gamma, prev_state, prev_action, reward, next_state):
    """
    Double Q-learning:
      - update one table at random
      - select the next greedy action from that table
      - evaluate it using the other table
    """
    if prev_state is None or prev_action is None or next_state is None:
        return

    if random.random() < 0.5:
        next_action = greedy_action_from_values(q_a[next_state])
        target = reward + gamma * q_b[next_state][next_action]
        td_update_from_target(q_a, alpha, prev_state, prev_action, target)
    else:
        next_action = greedy_action_from_values(q_b[next_state])
        target = reward + gamma * q_a[next_state][next_action]
        td_update_from_target(q_b, alpha, prev_state, prev_action, target)


def double_q_apply_target(q_a, q_b, alpha, state, action, target):
    if random.random() < 0.5:
        td_update_from_target(q_a, alpha, state, action, target)
    else:
        td_update_from_target(q_b, alpha, state, action, target)


def rl_update_dyna_q(
    Q_table,
    model,
    model_keys,
    alpha,
    gamma,
    prev_state,
    prev_action,
    reward,
    next_state,
    planning_steps,
):
    """
    Dyna-Q:
      1) Real Q-learning update from experience
      2) Store one-step model transition
      3) Run planning updates from sampled model transitions
    """
    if prev_state is None or prev_action is None or next_state is None:
        return

    rl_update(Q_table, alpha, gamma, prev_state, prev_action, reward, next_state)

    key = (prev_state, prev_action)
    if key not in model:
        model_keys.append(key)
    model[key] = (reward, next_state)

    if planning_steps <= 0 or not model_keys:
        return

    for _ in range(planning_steps):
        model_state, model_action = random.choice(model_keys)
        model_reward, model_next_state = model[(model_state, model_action)]
        rl_update(
            Q_table,
            alpha,
            gamma,
            model_state,
            model_action,
            model_reward,
            model_next_state,
        )


def rl_update_hysteretic_q(
    Q_table,
    alpha,
    gamma,
    negative_ratio,
    prev_state,
    prev_action,
    reward,
    next_state,
):
    """
    Hysteretic Q-learning:
      - positive TD errors use alpha
      - negative TD errors use alpha * negative_ratio
    """
    if prev_state is None or prev_action is None or next_state is None:
        return

    q_old = Q_table[prev_state][prev_action]
    max_next = max(Q_table[next_state])
    target = reward + gamma * max_next
    delta = target - q_old
    effective_alpha = alpha if delta >= 0.0 else (alpha * negative_ratio)
    Q_table[prev_state][prev_action] = q_old + effective_alpha * delta


def rl_update_lenient_q(
    Q_table,
    temperature_table,
    alpha,
    gamma,
    prev_state,
    prev_action,
    reward,
    next_state,
    temperature_decay,
):
    """
    Lenient Q-learning:
      - maintain per-(s,a) leniency temperature
      - probabilistically ignore negative TD updates while temperature is high
    """
    if prev_state is None or prev_action is None or next_state is None:
        return

    q_old = Q_table[prev_state][prev_action]
    max_next = max(Q_table[next_state])
    target = reward + gamma * max_next
    delta = target - q_old

    temp = max(0.0, temperature_table[prev_state][prev_action])
    leniency = 1.0 - math.exp(-temp)

    if delta >= 0.0 or random.random() >= leniency:
        Q_table[prev_state][prev_action] = q_old + alpha * delta

    temperature_table[prev_state][prev_action] = temp * temperature_decay


def replacing_trace_update(Q_table, traces, alpha, delta, state, action):
    traces[state][action] = 1.0
    for s in range(N_STATE_BINS):
        for a in range(N_ACTIONS):
            trace = traces[s][a]
            if trace != 0.0:
                Q_table[s][a] += alpha * delta * trace


def decay_traces(traces, gamma, trace_lambda):
    factor = gamma * trace_lambda
    for s in range(N_STATE_BINS):
        for a in range(N_ACTIONS):
            traces[s][a] *= factor


def clear_traces(traces):
    for s in range(N_STATE_BINS):
        for a in range(N_ACTIONS):
            traces[s][a] = 0.0


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
    var_post_raw = 1.0 / (1.0 / var_prior + 1.0 / var_obs)

    # 4) Posterior mean uses the unconstrained posterior precision.
    mu_post = var_post_raw * (mu_old / var_prior + target / var_obs)

    # 5) Enforce a minimum stored variance so we never become completely rigid.
    var_post = max(var_post_raw, BAYES_MIN_VAR)

    q_mean[s][a] = mu_post
    q_var[s][a]  = var_post

    # Optional debug
    if s == REL_STATE_BINS // 2 and a == A_FIRE and random.random() < 0.001:
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


def get_alignment_threshold():
    bin_width = DISPLAY.width / REL_STATE_BINS
    return 0.4 * bin_width if bin_width > 0 else 0.0


def score_enemy_target(player, enemy, target_counts_last_frame, preferred_target_id=None):
    player_center_x = player.x + player.width / 2.0
    rel_x = enemy.rect.centerx - player_center_x
    abs_rel = abs(rel_x)

    half_width = DISPLAY.width / 2.0
    alignment = 1.0 - min(abs_rel / half_width, 1.0)

    time_to_escape = (DISPLAY.height - enemy.y) / max(enemy.speed, 1e-6)
    time_to_escape = max(0.0, time_to_escape)
    threat = 1.0 / (1.0 + time_to_escape)

    aligned = abs_rel < get_alignment_threshold()
    shot_window = 1.0 if aligned else 0.0

    crowd_count = target_counts_last_frame.get(enemy.enemy_id, 0)
    crowd_penalty = TARGET_CROWDING_WEIGHT * crowd_count
    preferred_bonus = TARGET_COORD_ASSIGNMENT_BONUS if enemy.enemy_id == preferred_target_id else 0.0

    score = (
        TARGET_THREAT_WEIGHT * threat
        + TARGET_ALIGNMENT_WEIGHT * alignment
        + TARGET_SHOT_WINDOW_WEIGHT * shot_window
        - crowd_penalty
        + preferred_bonus
    )
    return score, rel_x, aligned


def choose_target_enemy_with_memory(
    player,
    enemies,
    current_target_id=None,
    target_counts_last_frame=None,
    preferred_target_id=None,
):
    if target_counts_last_frame is None:
        target_counts_last_frame = {}

    candidates = [e for e in enemies if e.rect.bottom < player.y]
    if not candidates:
        return None, None, None, False

    scored = []
    for enemy in candidates:
        score, rel_x, aligned = score_enemy_target(
            player,
            enemy,
            target_counts_last_frame,
            preferred_target_id=preferred_target_id,
        )
        scored.append((enemy, score, rel_x, aligned))

    best_enemy, best_score, best_rel_x, best_aligned = max(scored, key=lambda t: t[1])

    if current_target_id is None:
        return best_enemy, best_score, best_rel_x, best_aligned

    current_tuple = None
    for tup in scored:
        if tup[0].enemy_id == current_target_id:
            current_tuple = tup
            break

    if current_tuple is None:
        return best_enemy, best_score, best_rel_x, best_aligned

    current_enemy, current_score, current_rel_x, current_aligned = current_tuple
    if best_score <= (current_score + TARGET_SWITCH_SCORE_MARGIN):
        return current_enemy, current_score, current_rel_x, current_aligned

    return best_enemy, best_score, best_rel_x, best_aligned


def find_enemy_by_id(enemies, enemy_id):
    if enemy_id is None:
        return None
    for enemy in enemies:
        if enemy.enemy_id == enemy_id:
            return enemy
    return None


def build_greedy_target_assignment(players_by_owner, enemies):
    assignments = {owner: None for owner in BOT_OWNER_IDS}
    available_enemy_ids = {enemy.enemy_id for enemy in enemies}
    unassigned_owners = set(BOT_OWNER_IDS)

    while unassigned_owners and available_enemy_ids:
        best_owner = None
        best_enemy_id = None
        best_score = None

        for owner in unassigned_owners:
            player = players_by_owner[owner]
            for enemy in enemies:
                if enemy.enemy_id not in available_enemy_ids:
                    continue
                if enemy.rect.bottom >= player.y:
                    continue
                score, _, _ = score_enemy_target(player, enemy, {})
                if best_score is None or score > best_score:
                    best_owner = owner
                    best_enemy_id = enemy.enemy_id
                    best_score = score

        if best_owner is None or best_enemy_id is None:
            break

        assignments[best_owner] = best_enemy_id
        unassigned_owners.remove(best_owner)
        available_enemy_ids.remove(best_enemy_id)

    return assignments


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


def get_time_to_escape_bin(target):
    time_to_escape = (DISPLAY.height - target.y) / max(target.speed, 1e-6)
    if time_to_escape < 1.2:
        return 0
    if time_to_escape < 2.5:
        return 1
    return 2


def get_lock_confidence_bin(lock_duration):
    return 1 if lock_duration >= TARGET_LOCK_CONFIDENCE_SECONDS else 0


def get_state_bin(p2, enemies, target=None, lock_duration=0.0):
    """
    Discretise horizontal relation between P2 and its chosen target into bins.
    state = 0 .. REL_STATE_BINS-1 for valid targets, NO_TARGET_STATE otherwise.
    """
    if not enemies:
        return NO_TARGET_STATE

    if target is None:
        target = choose_target_enemy(p2, enemies)
    if target is None:
        return NO_TARGET_STATE

    p2_center_x = p2.x + p2.width / 2
    rel = target.rect.centerx - p2_center_x  # negative: enemy left, positive: enemy right

    rel_norm = (rel + DISPLAY.width / 2) / DISPLAY.width
    rel_norm = max(0.0, min(1.0, rel_norm))

    idx = int(rel_norm * REL_STATE_BINS)
    if idx >= REL_STATE_BINS:
        idx = REL_STATE_BINS - 1

    time_bin = get_time_to_escape_bin(target)
    lock_bin = get_lock_confidence_bin(lock_duration)
    state_idx = ((lock_bin * TIME_TO_ESCAPE_BINS + time_bin) * REL_STATE_BINS) + idx
    return max(0, min(NO_TARGET_STATE - 1, state_idx))


def get_rel_x(p2, enemies, target=None):
    """
    Return horizontal offset (enemy_x - p2_center_x) for P2's chosen target.
    If no suitable target, returns None.
    """
    if not enemies:
        return None
    if target is None:
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
    elif algo == ALGO_SARSA_LAMBDA:
        return "SARSA-LAMBDA"
    elif algo == ALGO_WATKINS_Q_LAMBDA:
        return "WATKINS-Q-LAMBDA"
    elif algo == ALGO_DOUBLE_Q:
        return "DOUBLE Q"
    elif algo == ALGO_EXPECTED_SARSA:
        return "EXPECTED SARSA"
    elif algo == ALGO_N_STEP_SARSA:
        return "N-STEP SARSA"
    elif algo == ALGO_DIFFERENTIAL_SARSA:
        return "DIFFERENTIAL SARSA"
    elif algo == ALGO_R_LEARNING:
        return "R-LEARNING"
    elif algo == ALGO_DYNA_Q:
        return "DYNA-Q"
    elif algo == ALGO_UCB_Q:
        return "UCB Q-LEARNING"
    elif algo == ALGO_OPTIMISTIC_Q:
        return "OPTIMISTIC Q-LEARNING"
    elif algo == ALGO_HYSTERETIC_Q:
        return "HYSTERETIC Q-LEARNING"
    elif algo == ALGO_LENIENT_Q:
        return "LENIENT Q-LEARNING"
    elif algo == ALGO_BAYES_Q:
        return "BAYES-Q"
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
    current_state: int | None = None
    current_action: int | None = None
    reward_step: float = 0.0

    @property
    def color(self):
        return OWNER_COLORS[self.owner]

    @property
    def q_table(self):
        return self.controller.display_q_table

    @property
    def heatmap_label(self):
        if self.controller.algo == ALGO_BAYES_Q:
            return f"{self.owner} BAYES-Q"
        if self.controller.algo == ALGO_N_STEP_SARSA:
            return f"{self.owner} {self.controller.n_step}-STEP SARSA"
        return f"{self.owner} {algo_display_name(self.controller.algo)}"


def init_targeting_runtime(controller):
    controller.elapsed_time = 0.0
    controller.current_target_id = None
    controller.current_target_score = None
    controller.current_target_rel_x = None
    controller.current_target_aligned = False
    controller.current_target_lock_started = None
    controller.current_target_first_hit_recorded = False

    controller.target_switches = 0
    controller.target_frames = 0
    controller.shared_target_frames = 0
    controller.shots_fired = 0
    controller.aligned_shots = 0
    controller.lock_to_first_hit_samples = []
    controller.targeted_enemy_ids = set()
    controller.targeted_enemy_escapes = 0

def update_controller_target_selection(
    controller,
    player,
    enemies,
    target_counts_last_frame,
    preferred_target_id=None,
):
    target, score, rel_x, aligned = choose_target_enemy_with_memory(
        player,
        enemies,
        current_target_id=controller.current_target_id,
        target_counts_last_frame=target_counts_last_frame,
        preferred_target_id=preferred_target_id,
    )

    previous_target_id = controller.current_target_id
    next_target_id = target.enemy_id if target is not None else None

    if (
        previous_target_id is not None
        and next_target_id is not None
        and previous_target_id != next_target_id
    ):
        controller.target_switches += 1

    if previous_target_id != next_target_id:
        controller.current_target_lock_started = (
            controller.elapsed_time if next_target_id is not None else None
        )
        controller.current_target_first_hit_recorded = False

    controller.current_target_id = next_target_id
    controller.current_target_score = score
    controller.current_target_rel_x = rel_x
    controller.current_target_aligned = bool(aligned)

    if next_target_id is not None:
        controller.target_frames += 1
        controller.targeted_enemy_ids.add(next_target_id)

    return target, rel_x


def register_controller_shared_target(controller, share_count):
    if controller.current_target_id is None:
        return
    if share_count > 1:
        controller.shared_target_frames += 1


def record_controller_shot(controller, aligned_on_fire):
    controller.shots_fired += 1
    if aligned_on_fire:
        controller.aligned_shots += 1


def record_controller_target_hit(controller, enemy_id):
    if controller.current_target_id != enemy_id:
        return
    if controller.current_target_first_hit_recorded:
        return
    if controller.current_target_lock_started is None:
        return

    latency = max(0.0, controller.elapsed_time - controller.current_target_lock_started)
    controller.lock_to_first_hit_samples.append(latency)
    controller.current_target_first_hit_recorded = True


def record_controller_target_escape(controller, enemy_id):
    if enemy_id in controller.targeted_enemy_ids:
        controller.targeted_enemy_escapes += 1


def targeting_metrics_snapshot(controller):
    runtime_minutes = max(controller.elapsed_time / 60.0, 1e-8)
    shared_target_rate = (
        controller.shared_target_frames / controller.target_frames
        if controller.target_frames > 0
        else 0.0
    )
    shots_when_aligned_rate = (
        controller.aligned_shots / controller.shots_fired
        if controller.shots_fired > 0
        else 0.0
    )
    mean_time_to_first_hit = (
        sum(controller.lock_to_first_hit_samples) / len(controller.lock_to_first_hit_samples)
        if controller.lock_to_first_hit_samples
        else None
    )
    targeted_enemy_count = len(controller.targeted_enemy_ids)
    escape_rate_of_targeted_enemies = (
        controller.targeted_enemy_escapes / targeted_enemy_count
        if targeted_enemy_count > 0
        else 0.0
    )

    return {
        "target_switches_per_min": controller.target_switches / runtime_minutes,
        "shared_target_rate": shared_target_rate,
        "shots_when_aligned_rate": shots_when_aligned_rate,
        "time_to_first_hit_after_lock": mean_time_to_first_hit,
        "escape_rate_of_targeted_enemies": escape_rate_of_targeted_enemies,
    }


def clamp_player_x(player):
    player.x = max(0, min(DISPLAY.width - player.width, player.x))
    player.rect.topleft = (player.x, player.y)


def compute_alignment_reward(action, rel, accuracy_focus):
    reward_step = 0.0
    bin_width = DISPLAY.width / REL_STATE_BINS
    aligned_thresh = 0.4 * bin_width if bin_width > 0 else 0

    if action == A_FIRE and rel is None:
        return MISALIGNED_FIRE_PENALTY * accuracy_focus

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
            reward_step += MOVE_AWAY_PENALTY
        if rel > 0 and action == A_MOVE_LEFT:
            reward_step += MOVE_AWAY_PENALTY
        if rel < 0 and action == A_MOVE_LEFT:
            reward_step += TOWARD_MOVE_BONUS
        if rel > 0 and action == A_MOVE_RIGHT:
            reward_step += TOWARD_MOVE_BONUS
        if abs_rel < aligned_thresh and action in (A_MOVE_LEFT, A_MOVE_RIGHT):
            reward_step += MOVE_WHILE_ALIGNED_PENALTY

    return reward_step


def compute_movement_progress_reward(rel_before, rel_after):
    if rel_before is None or rel_after is None:
        return 0.0

    start_abs = abs(rel_before)
    end_abs = abs(rel_after)
    progress = start_abs - end_abs
    max_scale = max(DISPLAY.width / 2.0, 1.0)
    return MOVE_PROGRESS_WEIGHT * (progress / max_scale)


def compute_centering_reward(player, enemies):
    if enemies:
        return 0.0

    center_x = DISPLAY.width / 2.0
    agent_center_x = player.x + player.width / 2.0
    dist_from_center = abs(agent_center_x - center_x)
    norm = min(dist_from_center / (DISPLAY.width / 2.0), 1.0)
    return CENTERING_WEIGHT * (1.0 - norm)


def maybe_fire_bullet(player, action, enemies, bullets, cooldown_timer, bullet_cooldown,
                      aim_acc_prob, miss_offset, bullet_owner, credit_state=None,
                      credit_action=None, target_enemy=None, rel_x=None, target_score=None):
    fired_this_frame = False
    aligned_on_fire = False

    if action != A_FIRE or cooldown_timer < bullet_cooldown:
        return cooldown_timer, fired_this_frame, aligned_on_fire

    target = target_enemy if target_enemy is not None else choose_target_enemy(player, enemies)
    if target is None or target.rect.bottom >= player.y:
        return cooldown_timer, fired_this_frame, aligned_on_fire

    center_x = player.x + player.width / 2

    if rel_x is None:
        rel_x = target.rect.centerx - center_x
    aligned_on_fire = (rel_x is not None and abs(rel_x) < get_alignment_threshold())

    fire_thresh = FIRE_GATE_ALIGNMENT_MULT * get_alignment_threshold()
    if rel_x is None or abs(rel_x) > fire_thresh:
        return cooldown_timer, False, False
    if target_score is not None and target_score < FIRE_GATE_MIN_TARGET_SCORE:
        return cooldown_timer, False, False

    if random.random() < aim_acc_prob:
        bullet_x = center_x
    else:
        bullet_x = center_x + miss_offset * random.choice([-1, 1])

    bullets.append(
        Bullet(
            bullet_x,
            player.y,
            owner=bullet_owner,
            credit_state=credit_state,
            credit_action=credit_action,
        )
    )
    return 0.0, True, aligned_on_fire


def apply_agent_action(player, action, enemies, bullets, dt, speed, time_since_last_shot,
                       bullet_cooldown, accuracy_focus, aim_acc_prob, miss_offset, bullet_owner,
                       credit_state=None, credit_action=None, target_enemy=None, rel_x=None,
                       target_score=None):
    reward_step = STEP_REWARD
    rel_before = rel_x if rel_x is not None else get_rel_x(player, enemies, target_enemy)

    if action == A_MOVE_LEFT:
        player.x -= speed * dt
    elif action == A_MOVE_RIGHT:
        player.x += speed * dt

    clamp_player_x(player)

    rel_after = get_rel_x(player, enemies, target_enemy)
    reward_step += compute_alignment_reward(action, rel_after, accuracy_focus)
    reward_step += compute_movement_progress_reward(rel_before, rel_after)
    reward_step += compute_centering_reward(player, enemies)

    time_since_last_shot += dt
    time_since_last_shot, fired_this_frame, aligned_on_fire = maybe_fire_bullet(
        player,
        action,
        enemies,
        bullets,
        time_since_last_shot,
        bullet_cooldown,
        aim_acc_prob,
        miss_offset,
        bullet_owner,
        credit_state,
        credit_action,
        target_enemy=target_enemy,
        rel_x=rel_after,
        target_score=target_score,
    )
    if fired_this_frame:
        reward_step += ACCURACY_FIRE_COST * accuracy_focus

    if action == A_FIRE and not fired_this_frame:
        reward_step += MISALIGNED_FIRE_PENALTY * accuracy_focus

    return reward_step, fired_this_frame, time_since_last_shot, aligned_on_fire


def update_corr_history(history, players):
    centers = tuple(player.x + player.width / 2.0 for player in players)
    history.append(centers)
    if len(history) > UI.corr_window_size:
        history.pop(0)
    return compute_corr_matrix(history)


def update_projectiles(bullets, bot_states, dt):
    new_bullets = []

    for bullet in bullets:
        bullet.update(dt)
        if bullet.off_screen():
            if bullet.owner in BOT_OWNER_IDS:
                reward = MISS_BULLET_PENALTY * bot_states[bullet.owner].controller.accuracy_focus
                bot_states[bullet.owner].controller.credit_projectile_outcome(
                    bullet.credit_state,
                    bullet.credit_action,
                    reward,
                )
        else:
            new_bullets.append(bullet)

    return new_bullets


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
        for owner in BOT_OWNER_IDS:
            bot_states[owner].controller.register_target_escape(enemy.enemy_id)

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
    for bullet in bullets[:]:
        for enemy in enemies[:]:
            if not bullet.rect.colliderect(enemy.rect):
                continue

            if bullet.owner != "P1" or not training_mode:
                stats_by_owner[bullet.owner].score += 10
                stats_by_owner[bullet.owner].hits += 1

            popup_color = OWNER_COLORS.get(bullet.owner, BULLET_COLOR)
            if bullet.owner in BOT_OWNER_IDS:
                reward = HIT_REWARD * bot_states[bullet.owner].controller.accuracy_focus
                bot_states[bullet.owner].controller.credit_projectile_outcome(
                    bullet.credit_state,
                    bullet.credit_action,
                    reward,
                )
                bot_states[bullet.owner].controller.register_target_hit(enemy.enemy_id)

            score_popups.append(
                ScorePopup(enemy.rect.centerx, enemy.rect.centery, "+10", popup_color)
            )
            bullets.remove(bullet)
            enemies.remove(enemy)
            break


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
    def __init__(self, x, y, owner="P1", credit_state=None, credit_action=None):
        self.width  = 4
        self.height = 12
        self.x = x - self.width / 2
        self.y = y
        self.owner = owner  # "P1", "P2", "P3", "P4"
        self.credit_state = credit_state
        self.credit_action = credit_action
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
        self.enemy_id = allocate_enemy_id()

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
      - SARSA(lambda)
      - Watkins Q(lambda)
      - Double Q-learning
      - Expected SARSA
      - N-step SARSA
      - Differential SARSA
      - Dyna-Q
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
        trace_lambda=0.0,
        secondary_q_table=None,
        avg_reward_step_size=0.01,
        dyna_planning_steps=0,
        ucb_exploration=1.0,
        optimistic_init_value=5.0,
        hysteretic_negative_ratio=0.1,
        leniency_temperature_init=1.5,
        leniency_temperature_decay=0.995,
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
        self.trace_lambda   = trace_lambda
        self.secondary_q    = secondary_q_table
        self.avg_reward_step_size = avg_reward_step_size
        self.avg_reward_estimate = 0.0
        self.dyna_planning_steps = dyna_planning_steps
        self.dyna_model = {}
        self.dyna_model_keys = []
        self.ucb_exploration = ucb_exploration
        self.optimistic_init_value = optimistic_init_value
        self.hysteretic_negative_ratio = hysteretic_negative_ratio
        self.leniency_temperature_decay = leniency_temperature_decay
        self.state_visits = [0 for _ in range(N_STATE_BINS)]
        self.action_visits = [
            [0 for _ in range(N_ACTIONS)]
            for _ in range(N_STATE_BINS)
        ]
        self.leniency_temperature = None
        if self.algo == ALGO_LENIENT_Q:
            self.leniency_temperature = [
                [leniency_temperature_init for _ in range(N_ACTIONS)]
                for _ in range(N_STATE_BINS)
            ]
        if self.algo == ALGO_OPTIMISTIC_Q:
            self._initialize_optimistic_q_if_fresh()

        if self.algo == ALGO_DOUBLE_Q and self.secondary_q is None:
            raise ValueError("Double Q agent requires a secondary_q_table.")

        # Internal RL state
        self.prev_state  = None
        self.prev_action = None
        self.prev_reward = 0.0

        self.time_since_last_shot = 0.0
        self.q_log_timer          = 0.0
        self.last_move_action = None
        init_targeting_runtime(self)

        # For N-step SARSA
        self.trajectory = []
        self.eligibility = None
        if self.algo in (ALGO_SARSA_LAMBDA, ALGO_WATKINS_Q_LAMBDA):
            self.eligibility = [
                [0.0 for _ in range(N_ACTIONS)]
                for _ in range(N_STATE_BINS)
            ]

    def select_action(self, state):
        """ε-greedy action selection."""
        if self.algo == ALGO_UCB_Q:
            return self.select_action_ucb(state)

        if random.random() < self.epsilon:
            return random.randint(0, N_ACTIONS - 1)

        if self.algo == ALGO_DOUBLE_Q and self.secondary_q is not None:
            q_row = [self.Q[state][a] + self.secondary_q[state][a] for a in range(N_ACTIONS)]
        else:
            q_row = self.Q[state]
        return greedy_action_from_values(q_row)

    def select_action_ucb(self, state):
        unvisited_actions = [a for a in range(N_ACTIONS) if self.action_visits[state][a] == 0]
        if unvisited_actions:
            return random.choice(unvisited_actions)

        state_count = max(1, self.state_visits[state])
        log_term = math.log(state_count + 1.0)
        ucb_values = []
        for a in range(N_ACTIONS):
            n_sa = max(1, self.action_visits[state][a])
            bonus = self.ucb_exploration * math.sqrt(log_term / n_sa)
            ucb_values.append(self.Q[state][a] + bonus)
        return greedy_action_from_values(ucb_values)

    def _initialize_optimistic_q_if_fresh(self):
        if any(any(value != 0.0 for value in row) for row in self.Q):
            return

        for s in range(N_STATE_BINS):
            for a in range(N_ACTIONS):
                self.Q[s][a] = self.optimistic_init_value

    @property
    def display_q_table(self):
        if self.algo != ALGO_DOUBLE_Q or self.secondary_q is None:
            return self.Q

        return [
            [
                0.5 * (self.Q[s][a] + self.secondary_q[s][a])
                for a in range(N_ACTIONS)
            ]
            for s in range(N_STATE_BINS)
        ]

    def register_shared_target(self, share_count):
        register_controller_shared_target(self, share_count)

    def register_shot(self, aligned_on_fire):
        record_controller_shot(self, aligned_on_fire)

    def register_target_hit(self, enemy_id):
        record_controller_target_hit(self, enemy_id)

    def register_target_escape(self, enemy_id):
        record_controller_target_escape(self, enemy_id)

    def targeting_metrics(self):
        return targeting_metrics_snapshot(self)

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

        elif self.algo == ALGO_SARSA_LAMBDA:
            q_old = self.Q[ps][pa]
            q_next = self.Q[ns][na]
            delta = r + self.gamma * q_next - q_old
            replacing_trace_update(self.Q, self.eligibility, self.alpha, delta, ps, pa)
            decay_traces(self.eligibility, self.gamma, self.trace_lambda)

        elif self.algo == ALGO_WATKINS_Q_LAMBDA:
            q_old = self.Q[ps][pa]
            next_row = self.Q[ns]
            max_next = max(next_row)
            delta = r + self.gamma * max_next - q_old
            replacing_trace_update(self.Q, self.eligibility, self.alpha, delta, ps, pa)
            if next_row[na] == max_next:
                decay_traces(self.eligibility, self.gamma, self.trace_lambda)
            else:
                clear_traces(self.eligibility)

        elif self.algo == ALGO_DOUBLE_Q:
            rl_update_double_q(self.Q, self.secondary_q, self.alpha, self.gamma, ps, pa, r, ns)

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

        elif self.algo == ALGO_DIFFERENTIAL_SARSA:
            self.avg_reward_estimate = rl_update_differential_sarsa(
                self.Q,
                self.alpha,
                self.avg_reward_step_size,
                self.avg_reward_estimate,
                ps,
                pa,
                r,
                ns,
                na,
            )

        elif self.algo == ALGO_DYNA_Q:
            rl_update_dyna_q(
                self.Q,
                self.dyna_model,
                self.dyna_model_keys,
                self.alpha,
                self.gamma,
                ps,
                pa,
                r,
                ns,
                self.dyna_planning_steps,
            )

        elif self.algo == ALGO_R_LEARNING:
            self.avg_reward_estimate = rl_update_r_learning(
                self.Q,
                self.alpha,
                self.avg_reward_step_size,
                self.avg_reward_estimate,
                ps,
                pa,
                r,
                ns,
            )

        elif self.algo == ALGO_UCB_Q:
            rl_update(self.Q, self.alpha, self.gamma, ps, pa, r, ns)

        elif self.algo == ALGO_OPTIMISTIC_Q:
            rl_update(self.Q, self.alpha, self.gamma, ps, pa, r, ns)

        elif self.algo == ALGO_HYSTERETIC_Q:
            rl_update_hysteretic_q(
                self.Q,
                self.alpha,
                self.gamma,
                self.hysteretic_negative_ratio,
                ps,
                pa,
                r,
                ns,
            )

        elif self.algo == ALGO_LENIENT_Q:
            rl_update_lenient_q(
                self.Q,
                self.leniency_temperature,
                self.alpha,
                self.gamma,
                ps,
                pa,
                r,
                ns,
                self.leniency_temperature_decay,
            )

    def credit_projectile_outcome(self, state, action, reward):
        """
        Credit delayed projectile outcomes directly to the firing decision.
        The compact agent state does not track bullets in flight, so these
        outcomes should not be reassigned to whatever action happens next.
        """
        if state is None or action is None:
            return
        if self.algo == ALGO_DOUBLE_Q and self.secondary_q is not None:
            double_q_apply_target(self.Q, self.secondary_q, self.alpha, state, action, reward)
            return

        if self.algo == ALGO_HYSTERETIC_Q:
            q_old = self.Q[state][action]
            delta = reward - q_old
            effective_alpha = self.alpha if delta >= 0.0 else (self.alpha * self.hysteretic_negative_ratio)
            self.Q[state][action] = q_old + effective_alpha * delta
            return

        if self.algo == ALGO_LENIENT_Q:
            temp = max(0.0, self.leniency_temperature[state][action])
            leniency = 1.0 - math.exp(-temp)
            q_old = self.Q[state][action]
            delta = reward - q_old
            if delta >= 0.0 or random.random() >= leniency:
                self.Q[state][action] = q_old + self.alpha * delta
            self.leniency_temperature[state][action] = temp * self.leniency_temperature_decay
            return

        td_update_from_target(self.Q, self.alpha, state, action, reward)

    def step(self, player, enemies, bullets, dt, target_counts_last_frame=None, preferred_target_id=None):
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
        self.elapsed_time += dt

        # Epsilon decay
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * (self.epsilon_decay ** dt)
        )

        if target_counts_last_frame is None:
            target_counts_last_frame = {}

        target_enemy, target_rel_x = update_controller_target_selection(
            self, player, enemies, target_counts_last_frame, preferred_target_id=preferred_target_id
        )

        lock_duration = 0.0
        if self.current_target_lock_started is not None:
            lock_duration = max(0.0, self.elapsed_time - self.current_target_lock_started)

        # Observe current state
        current_state = get_state_bin(player, enemies, target_enemy, lock_duration=lock_duration)

        # Choose action under current policy
        action = self.select_action(current_state)
        self.state_visits[current_state] += 1
        self.action_visits[current_state][action] += 1

        # RL update for previous transition
        self._do_update(current_state, action)

        reward_step, fired_this_frame, self.time_since_last_shot, aligned_on_fire = apply_agent_action(
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
            credit_state=current_state,
            credit_action=action,
            target_enemy=target_enemy,
            rel_x=target_rel_x,
            target_score=self.current_target_score,
        )
        if action in (A_MOVE_LEFT, A_MOVE_RIGHT):
            if (
                self.last_move_action in (A_MOVE_LEFT, A_MOVE_RIGHT)
                and self.last_move_action != action
            ):
                reward_step += MOVE_DIRECTION_SWITCH_PENALTY
            self.last_move_action = action
        else:
            self.last_move_action = None
        if fired_this_frame:
            self.register_shot(aligned_on_fire)

        # Q-table logging
        self.q_log_timer += dt
        if self.q_log_timer >= UI.q_log_interval:
            self.q_log_timer = 0.0
            print(f"=== Q-table snapshot ({self.name} {self.algo}) ===")
            for i, row in enumerate(self.display_q_table):
                print(f"{self.name} state {i}: {['{:.2f}'.format(v) for v in row]}")
            tm = self.targeting_metrics()
            ttfh = tm["time_to_first_hit_after_lock"]
            ttfh_str = f"{ttfh:.2f}s" if ttfh is not None else "n/a"
            print(
                f"{self.name} targeting: switches/min={tm['target_switches_per_min']:.2f}, "
                f"shared={tm['shared_target_rate']:.2f}, aligned_shots={tm['shots_when_aligned_rate']:.2f}, "
                f"lock_to_hit={ttfh_str}, targeted_escape={tm['escape_rate_of_targeted_enemies']:.2f}"
            )

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
        epsilon_min=BAYES_EPSILON_MIN,
        epsilon_decay=BAYES_EPSILON_DECAY,
    ):
        self.name           = name
        self.algo           = ALGO_BAYES_Q
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
        self.epsilon_min     = epsilon_min
        self.epsilon_decay   = epsilon_decay

        self.prev_state  = None
        self.prev_action = None
        self.prev_reward = 0.0

        self.time_since_last_shot = 0.0
        self.last_move_action = None
        init_targeting_runtime(self)

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

    def credit_projectile_outcome(self, state, action, reward):
        if state is None or action is None:
            return
        bayes_update_q(self.q_mean, self.q_var, state, action, reward)

    @property
    def display_q_table(self):
        return self.q_mean

    def register_shared_target(self, share_count):
        register_controller_shared_target(self, share_count)

    def register_shot(self, aligned_on_fire):
        record_controller_shot(self, aligned_on_fire)

    def register_target_hit(self, enemy_id):
        record_controller_target_hit(self, enemy_id)

    def register_target_escape(self, enemy_id):
        record_controller_target_escape(self, enemy_id)

    def targeting_metrics(self):
        return targeting_metrics_snapshot(self)

    def step(self, player, enemies, bullets, dt, target_counts_last_frame=None, preferred_target_id=None):
        """
        Mirrors RLAgent.step:
          - observe state
          - choose action
          - Bayesian Q update using previous transition
          - move & fire
          - shaping rewards

        Returns (current_state, action, reward_step, fired_this_frame)
        """
        self.elapsed_time += dt

        # Decay epsilon over time
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * (self.epsilon_decay ** dt)
        )

        if target_counts_last_frame is None:
            target_counts_last_frame = {}

        target_enemy, target_rel_x = update_controller_target_selection(
            self, player, enemies, target_counts_last_frame, preferred_target_id=preferred_target_id
        )

        lock_duration = 0.0
        if self.current_target_lock_started is not None:
            lock_duration = max(0.0, self.elapsed_time - self.current_target_lock_started)

        # --- Observe current state ---
        current_state = get_state_bin(player, enemies, target_enemy, lock_duration=lock_duration)

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

        reward_step, fired_this_frame, self.time_since_last_shot, aligned_on_fire = apply_agent_action(
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
            credit_state=current_state,
            credit_action=action,
            target_enemy=target_enemy,
            rel_x=target_rel_x,
            target_score=self.current_target_score,
        )
        if action in (A_MOVE_LEFT, A_MOVE_RIGHT):
            if (
                self.last_move_action in (A_MOVE_LEFT, A_MOVE_RIGHT)
                and self.last_move_action != action
            ):
                reward_step += MOVE_DIRECTION_SWITCH_PENALTY
            self.last_move_action = action
        else:
            self.last_move_action = None
        if fired_this_frame:
            self.register_shot(aligned_on_fire)

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
    img = pygame.image.load(str(ASSETS.background_image)).convert()
    img = pygame.transform.scale(img, (DISPLAY.width, DISPLAY.height))
    return img
  

def load_cloud_layer():
    """
    Load a semi-transparent cloud layer and optionally scale it up/down.
    CLOUD_SCALE > 1.0  -> bigger clouds
    CLOUD_SCALE < 1.0  -> smaller clouds

    The resulting image is then scrolled top-to-bottom.
    """
    img = pygame.image.load(str(ASSETS.cloud_image)).convert_alpha()

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
    img = pygame.image.load(str(ASSETS.player_image)).convert_alpha()

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

    img = pygame.image.load(str(ASSETS.enemy_image)).convert_alpha()

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


def draw_score_hud_rl_agents(surf, font, stats_by_owner, bot_states):
    x_right = DISPLAY.width - 10
    y = 10
    compact_font = pygame.font.SysFont(None, max(16, font.get_height() - 8))

    for owner in BOT_OWNER_IDS:
        stats = stats_by_owner[owner]
        color = OWNER_COLORS.get(owner, (220, 220, 220))
        controller = bot_states[owner].controller

        epsilon = getattr(controller, "epsilon", None)
        epsilon_text = f"{epsilon:.2f}" if epsilon is not None else "-"
        line = f"{owner} S{stats.score} K{stats.hits} M{stats.misses} A{int(stats.accuracy)}% ε{epsilon_text}"

        text = compact_font.render(line, True, color)
        rect = text.get_rect(topright=(x_right, y))
        surf.blit(text, rect)
        y = rect.bottom + 4


def draw_targeting_metrics_panel(surf, font, bot_states, top_left_x=10, top_left_y=190):
    title_font = pygame.font.SysFont(None, max(18, font.get_height() - 6))
    row_font = pygame.font.SysFont(None, max(14, font.get_height() - 12))

    rows = []
    for owner in BOT_OWNER_IDS:
        metrics = bot_states[owner].controller.targeting_metrics()
        ttfh = metrics["time_to_first_hit_after_lock"]
        ttfh_text = f"{ttfh:.2f}s" if ttfh is not None else "-"
        rows.append(
            (
                owner,
                f"{owner} sw:{metrics['target_switches_per_min']:.1f}/m "
                f"share:{metrics['shared_target_rate']:.2f} "
                f"align:{metrics['shots_when_aligned_rate']:.2f} "
                f"hit:{ttfh_text} "
                f"esc:{metrics['escape_rate_of_targeted_enemies']:.2f}",
            )
        )

    panel_width = max(
        title_font.size("TARGETING METRICS")[0],
        max((row_font.size(text)[0] for _, text in rows), default=0),
    ) + 16
    line_height = row_font.get_linesize()
    panel_height = 10 + title_font.get_linesize() + 6 + len(rows) * line_height + 8

    panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
    panel_surface.fill((12, 12, 20, 185))
    surf.blit(panel_surface, (top_left_x, top_left_y))
    pygame.draw.rect(surf, (120, 120, 140), pygame.Rect(top_left_x, top_left_y, panel_width, panel_height), 1)

    title = title_font.render("TARGETING METRICS", True, (230, 230, 230))
    surf.blit(title, (top_left_x + 8, top_left_y + 6))

    y = top_left_y + 10 + title_font.get_linesize() + 2
    for owner, text in rows:
        color = OWNER_COLORS.get(owner, (220, 220, 220))
        img = row_font.render(text, True, color)
        surf.blit(img, (top_left_x + 8, y))
        y += line_height



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
    title_img = title_font.render(label, True, highlight_color)
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
    log_active_agent_lineup()

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
    player5 = Player(
        DISPLAY.width / 2 + 260,
        base_y - 80,
        color=PLAYER5_COLOR,
        sprite=player_sprite,
    )
    player6 = Player(
        DISPLAY.width / 2 - 260,
        base_y - 80,
        color=PLAYER6_COLOR,
        sprite=player_sprite,
    )


    # ------------------ RL Agents ------------------
    agent_p2 = build_agent_controller("P2")
    agent_p3 = build_agent_controller("P3")
    agent_p4 = build_agent_controller("P4")
    agent_p5 = build_agent_controller("P5")
    agent_p6 = build_agent_controller("P6")

    bullets = []
    enemies = []
    score_popups = []

    enemy_spawn_timer = 0.0
    time_since_last_shot_p1 = 0.0
    target_counts_last_frame = {}

    stats_by_owner = {owner: ScoreState() for owner in OWNER_IDS}
    players_by_owner = {
        "P1": player1,
        "P2": player2,
        "P3": player3,
        "P4": player4,
        "P5": player5,
        "P6": player6,
    }
    bot_states = {
        "P2": BotRuntime("P2", player2, agent_p2),
        "P3": BotRuntime("P3", player3, agent_p3),
        "P4": BotRuntime("P4", player4, agent_p4),
        "P5": BotRuntime("P5", player5, agent_p5),
        "P6": BotRuntime("P6", player6, agent_p6),
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

        preferred_target_by_owner = build_greedy_target_assignment(players_by_owner, enemies)

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
                target_counts_last_frame,
                preferred_target_id=preferred_target_by_owner.get(owner),
            )
            if fired_this_frame:
                stats_by_owner[owner].shots += 1

        target_counts_current = {}
        for owner in BOT_OWNER_IDS:
            target_id = bot_states[owner].controller.current_target_id
            if target_id is None:
                continue
            target_counts_current[target_id] = target_counts_current.get(target_id, 0) + 1

        for owner in BOT_OWNER_IDS:
            target_id = bot_states[owner].controller.current_target_id
            share_count = target_counts_current.get(target_id, 0) if target_id is not None else 0
            bot_states[owner].controller.register_shared_target(share_count)

        target_counts_last_frame = target_counts_current

        # -------------- Update world --------------

        # grid_offset -= GRID_SCROLL_SPEED * dt
        
        # Update background scroll offset (negative = image moves down)
        bg_offset -= GAME.grid_scroll_speed * dt
        cloud_offset -= GAME.cloud_scroll_speed * dt

        bullets = update_projectiles(
            bullets,
            bot_states,
            dt,
        )

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

        resolve_bullet_collisions(
            bullets,
            enemies,
            stats_by_owner,
            bot_states,
            training_mode,
            score_popups,
        )

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
            target = find_enemy_by_id(enemies, bot_states[owner].controller.current_target_id)
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


        draw_score_hud_rl_agents(screen, hud_font, stats_by_owner, bot_states)
        draw_targeting_metrics_panel(screen, hud_font, bot_states)


        for p in score_popups:
            p.draw(screen, popup_font)


        # Q heatmaps: six bot lanes across the bottom band
        heatmap_top = DISPLAY.height - UI.heatmap_height
        lane_count = len(BOT_OWNER_IDS)
        lane_width = DISPLAY.width // lane_count

        for idx, owner in enumerate(BOT_OWNER_IDS):
            left_x = idx * lane_width
            width = lane_width if idx < lane_count - 1 else (DISPLAY.width - left_x)
            draw_q_heatmap(
                screen,
                bot_states[owner].q_table,
                heatmap_top,
                left_x,
                width,
                bot_states[owner].heatmap_label,
                current_state=bot_states[owner].current_state,
                highlight_color=bot_states[owner].color,
            )


        pygame.display.flip()

    # On exit, save the latest Q-tables
    save_q_tables()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
