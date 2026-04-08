"""
graph_generator.py — Misinformation Cascade Containment Environment

Responsibilities:
  1. Build a seeded NetworkX graph (Erdos-Renyi or Barabasi-Albert).
  2. Assign GraphNode attributes to every node (influence, virality, spread prob).
  3. Seed initial infections deterministically.
  4. Pre-compute the null trajectory: simulate LATENT_DURATION steps of
     unchecked spread with no agent actions, producing null_trajectory[0..max_steps].
  5. Return the annotated graph and null trajectory to env.py for use in reset().

Design contracts:
  - All randomness flows through a single random.Random(seed) instance.
  - numpy.random is NOT used — avoids global seed state side effects.
  - The same seed always produces the same graph, same infections, same trajectory.
  - null_trajectory has exactly max_steps + 1 entries (validated by CascadeState).
  - No NetworkX node is left without a GraphNode attribute.
"""

from __future__ import annotations

import random
from typing import Optional

import networkx as nx

try:
    from .models import LATENT_DURATION, TASK_CONFIG, TASK_SEEDS, GraphNode
except ImportError:
    from models import LATENT_DURATION, TASK_CONFIG, TASK_SEEDS, GraphNode


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_graph(
    difficulty: str,
    seed: Optional[int] = None,
) -> tuple[nx.Graph, list[GraphNode], list[float]]:
    """Build and annotate a graph for the given difficulty.

    Returns:
        G               — NetworkX graph with node IDs as strings ("n_0" … "n_N-1")
        nodes           — list of GraphNode objects in node-id order
        null_trajectory — list of cumulative null-agent weighted damage,
                          length = max_steps + 1
    """
    if difficulty not in TASK_CONFIG:
        raise ValueError(f"Unknown difficulty '{difficulty}'. Must be one of {list(TASK_CONFIG)}")

    cfg  = TASK_CONFIG[difficulty]
    seed = TASK_SEEDS[difficulty] if seed is None else seed
    rng  = random.Random(seed)

    G = _build_topology(cfg, seed)
    G = _relabel_nodes(G)
    nodes = _assign_node_attributes(G, cfg, rng, difficulty)
    nodes = _seed_infections(nodes, cfg, rng)
    null_trajectory = _compute_null_trajectory(G, nodes, cfg, seed)

    return G, nodes, null_trajectory


# ---------------------------------------------------------------------------
# Step 1: Topology
# ---------------------------------------------------------------------------

def _build_topology(cfg: dict, seed: int) -> nx.Graph:
    """Build raw NetworkX graph. Nodes are integers at this stage."""
    graph_type = cfg["graph_type"]

    if graph_type == "erdos_renyi":
        G = nx.erdos_renyi_graph(
            n    = cfg["n_nodes"],
            p    = cfg["er_edge_prob"],
            seed = seed,
        )
        # Erdos-Renyi can produce isolated nodes. Connect each isolate to a
        # random non-isolate so the graph is fully reachable.
        isolates = list(nx.isolates(G))
        non_isolates = [n for n in G.nodes if G.degree(n) > 0]
        rng = random.Random(seed + 1)  # offset so reconnection doesn't alter main sequence
        for iso in isolates:
            target = rng.choice(non_isolates)
            G.add_edge(iso, target)

    elif graph_type == "barabasi_albert":
        G = nx.barabasi_albert_graph(
            n    = cfg["n_nodes"],
            m    = cfg["ba_edges"],
            seed = seed,
        )
    else:
        raise ValueError(f"Unknown graph_type '{graph_type}'")

    return G


# ---------------------------------------------------------------------------
# Step 2: Relabel nodes to string IDs
# ---------------------------------------------------------------------------

def _relabel_nodes(G: nx.Graph) -> nx.Graph:
    """Relabel integer nodes to string IDs: 'n_0', 'n_1', …"""
    mapping = {i: f"n_{i}" for i in G.nodes}
    return nx.relabel_nodes(G, mapping)


# ---------------------------------------------------------------------------
# Step 3: Assign node attributes
# ---------------------------------------------------------------------------

def _assign_node_attributes(
    G: nx.Graph,
    cfg: dict,
    rng: random.Random,
    difficulty: str,
) -> list[GraphNode]:
    """Compute and attach GraphNode attributes to every node in G.

    Influence score: normalized degree centrality [0.0, 1.0].
        influence = degree / (total_nodes - 1)

    Virality modifier:
        Hard (BA graph): degree-mapped linearly from 0.5 (min degree) to 2.0 (max degree).
            Hubs are inherently more viral — this is the structural danger of BA graphs.
        Easy / Medium (ER graph): randomly assigned in [0.5, 1.5] using seeded rng.
            Prevents the agent from inferring virality purely from degree.

    Effective spread prob: base_spread_prob * virality_modifier, clamped to [0.0, 1.0].
    """
    n_nodes         = cfg["n_nodes"]
    base_spread     = cfg["base_spread_prob"]
    graph_type      = cfg["graph_type"]

    degrees         = dict(G.degree())
    max_degree      = max(degrees.values()) if degrees else 1
    min_degree      = min(degrees.values()) if degrees else 0
    degree_range    = max(max_degree - min_degree, 1)  # avoid div-by-zero

    nodes: list[GraphNode] = []

    for node_id in G.nodes:
        degree = degrees[node_id]
        influence_score = degree / (n_nodes - 1) if n_nodes > 1 else 0.0
        influence_score = min(max(influence_score, 0.0), 1.0)

        if graph_type == "barabasi_albert":
            # Linear map: min_degree → 0.5, max_degree → 2.0
            virality_modifier = 0.5 + 1.5 * ((degree - min_degree) / degree_range)
        else:
            virality_modifier = rng.uniform(0.5, 1.5)

        virality_modifier = min(max(virality_modifier, 0.5), 2.0)
        effective_prob    = min(base_spread * virality_modifier, 1.0)

        gn = GraphNode(
            node_id               = node_id,
            degree                = degree,
            influence_score       = round(influence_score, 4),
            virality_modifier     = round(virality_modifier, 4),
            base_spread_prob      = base_spread,
            effective_spread_prob = round(effective_prob, 4),
            status                = "SUSCEPTIBLE",
            at_risk               = False,
            turns_at_risk         = 0,
            infected_neighbor     = None,
            boost_turns_remaining = 0,
        )
        nodes.append(gn)

        # Attach GraphNode to the NetworkX node for engine lookups
        G.nodes[node_id]["data"] = gn

    return nodes


# ---------------------------------------------------------------------------
# Step 4: Seed initial infections
# ---------------------------------------------------------------------------

def _seed_infections(
    nodes: list[GraphNode],
    cfg: dict,
    rng: random.Random,
) -> list[GraphNode]:
    """Mark n_initial_infected nodes as CONFIRMED_INFECTED.

    Selection: highest-influence nodes are preferred for easy/medium to give
    the agent a visible starting threat. For hard, seed randomly to prevent
    trivial hub-blocking strategies from dominating.
    """
    n_seed      = cfg["n_initial_infected"]
    graph_type  = cfg["graph_type"]

    if graph_type == "barabasi_albert":
        # Random seeding on hard — structural difficulty comes from topology
        candidates = rng.sample(nodes, n_seed)
    else:
        # Seed the top-influence nodes on easy/medium for a clear starting signal
        sorted_nodes = sorted(nodes, key=lambda n: n.influence_score, reverse=True)
        candidates   = sorted_nodes[:n_seed]

    for node in candidates:
        node.status = "CONFIRMED_INFECTED"

    return nodes


# ---------------------------------------------------------------------------
# Step 5: Pre-compute null trajectory
# ---------------------------------------------------------------------------

def _compute_null_trajectory(
    G: nx.Graph,
    nodes: list[GraphNode],
    cfg: dict,
    seed: int,
) -> list[float]:
    """Simulate max_steps of unchecked spread (no agent actions).

    Uses a DEEP COPY of node states so the original nodes are not mutated.
    The null agent does nothing — no FACTCHECK, no QUARANTINE, no INOCULATE.
    Latent mechanics apply: newly exposed nodes enter LATENT for LATENT_DURATION
    steps before becoming CONFIRMED_INFECTED.

    Returns null_trajectory: list of length max_steps + 1.
        null_trajectory[0] = initial weighted damage (before any steps).
        null_trajectory[t] = cumulative weighted damage after t steps.

    Weighted damage per step = sum of influence_score for all
    LATENT + CONFIRMED_INFECTED nodes at that moment.
    The null trajectory is used by env.py to:
        (a) compute step-specific spread deltas for last_action_effect strings.
        (b) compute the terminal counterfactual score.
    """
    max_steps   = cfg["max_steps"]
    base_spread = cfg["base_spread_prob"]
    ext_interval: Optional[int] = cfg.get("external_seed_interval")

    # Deep-copy node states into a simple dict for simulation speed
    sim: dict[str, dict] = {
        n.node_id: {
            "status":           n.status,
            "turns_at_risk":    n.turns_at_risk,
            "infected_neighbor": n.infected_neighbor,
            "influence_score":  n.influence_score,
            "effective_spread_prob": n.effective_spread_prob,
            "at_risk":          n.at_risk,
        }
        for n in nodes
    }

    # Separate RNG for null simulation — must not share state with graph rng
    null_rng = random.Random(seed + 999)

    def _weighted_damage(sim_state: dict) -> float:
        return sum(
            v["influence_score"]
            for v in sim_state.values()
            if v["status"] in ("LATENT", "CONFIRMED_INFECTED")
        )

    trajectory: list[float] = [_weighted_damage(sim)]  # index 0: initial state

    for step in range(1, max_steps + 1):
        # --- Advance LATENT timers ---
        for node_id, node in sim.items():
            if node["status"] == "LATENT":
                node["turns_at_risk"] += 1
                if node["turns_at_risk"] >= LATENT_DURATION:
                    node["status"] = "CONFIRMED_INFECTED"

        # --- Spread from CONFIRMED_INFECTED and LATENT nodes ---
        # Collect new exposures before applying them (synchronous update)
        new_exposures: dict[str, str] = {}  # {newly_exposed_id: infector_id}

        for node_id, node in sim.items():
            if node["status"] not in ("CONFIRMED_INFECTED", "LATENT"):
                continue
            for neighbor_id in G.neighbors(node_id):
                neighbor = sim[neighbor_id]
                if neighbor["status"] != "SUSCEPTIBLE":
                    continue
                if neighbor_id in new_exposures:
                    continue  # already being exposed this step
                if null_rng.random() < node["effective_spread_prob"]:
                    new_exposures[neighbor_id] = node_id

        # Apply new exposures — newly exposed enter LATENT immediately
        for exposed_id, infector_id in new_exposures.items():
            sim[exposed_id]["status"]           = "LATENT"
            sim[exposed_id]["at_risk"]          = True
            sim[exposed_id]["turns_at_risk"]    = 0
            sim[exposed_id]["infected_neighbor"] = infector_id

        # --- Hard mode: inject external seed every ext_interval steps ---
        if ext_interval and step % ext_interval == 0:
            susceptible_ids = [
                nid for nid, nd in sim.items()
                if nd["status"] == "SUSCEPTIBLE"
            ]
            if susceptible_ids:
                target_id = null_rng.choice(susceptible_ids)
                sim[target_id]["status"]        = "LATENT"
                sim[target_id]["at_risk"]       = True
                sim[target_id]["turns_at_risk"] = 0

        trajectory.append(_weighted_damage(sim))

    assert len(trajectory) == max_steps + 1, (
        f"null_trajectory length {len(trajectory)} != max_steps+1 {max_steps + 1}"
    )
    return trajectory


# ---------------------------------------------------------------------------
# Utility: node lookup by id
# ---------------------------------------------------------------------------

def get_node(G: nx.Graph, node_id: str) -> GraphNode:
    """Retrieve the live GraphNode attached to a NetworkX node."""
    data = G.nodes.get(node_id)
    if data is None:
        raise KeyError(f"Node '{node_id}' not found in graph.")
    return data["data"]


def get_all_nodes(G: nx.Graph) -> list[GraphNode]:
    """Return all GraphNode objects from the graph in node-id order."""
    return [G.nodes[nid]["data"] for nid in sorted(G.nodes)]


# ---------------------------------------------------------------------------
# Quick smoke test (run directly: python graph_generator.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for diff in ("easy", "medium", "hard"):
        cfg = TASK_CONFIG[diff]
        G, nodes, traj = build_graph(diff)

        n_infected = sum(1 for n in nodes if n.status == "CONFIRMED_INFECTED")
        max_inf    = cfg["n_initial_infected"]
        traj_len   = cfg["max_steps"] + 1

        assert len(nodes)  == cfg["n_nodes"],  f"{diff}: node count mismatch"
        assert n_infected  == max_inf,         f"{diff}: initial infection count mismatch"
        assert len(traj)   == traj_len,        f"{diff}: null_trajectory length mismatch"
        assert traj[0] > 0,                    f"{diff}: initial damage must be > 0"
        assert traj[-1] >= traj[0],            f"{diff}: null damage must be non-decreasing"

        print(
            f"[{diff:6}] nodes={len(nodes):2}  "
            f"initial_infected={n_infected}  "
            f"traj_len={len(traj)}  "
            f"null_damage_0={traj[0]:.3f}  "
            f"null_damage_final={traj[-1]:.3f}"
        )
    print("All smoke tests passed.")
