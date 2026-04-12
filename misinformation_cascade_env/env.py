"""Misinformation Cascade Containment — OpenEnv environment.

API: reset() → Observation, step(action) → Observation, state() → dict.
Physics: validate → apply → spread → boost decay → external seed → terminate.
"""

from __future__ import annotations

import random
import uuid
from typing import Optional

import networkx as nx

try:
    from .models import (
        ACTION_COSTS,
        BOOST_DURATION,
        LATENT_DURATION,
        SATURATION_MERCY_THRESHOLD,
        STARTING_BUDGET,
        TASK_CONFIG,
        TASK_SEEDS,
        CascadeAction,
        CascadeObservation,
        CascadeState,
        GraphNode,
        NodeSummary,
    )
    from .graph_generator import build_graph
    from .task_grader import clamp_score
except ImportError:
    from models import (
        ACTION_COSTS,
        BOOST_DURATION,
        LATENT_DURATION,
        SATURATION_MERCY_THRESHOLD,
        STARTING_BUDGET,
        TASK_CONFIG,
        TASK_SEEDS,
        CascadeAction,
        CascadeObservation,
        CascadeState,
        GraphNode,
        NodeSummary,
    )
    from graph_generator import build_graph
    from task_grader import clamp_score


# ── Environment ───────────────────────────────────────────────────────────

class MisinformationCascadeEnv:
    """
    Misinformation Cascade Containment — OpenEnv environment.

    Usage:
        env = MisinformationCascadeEnv(difficulty="hard")
        obs = env.reset()
        while not obs.done:
            action = agent.act(obs)
            obs = env.step(action)
        score = obs.reward
    """
    def __init__(self, difficulty: str = "easy", seed: Optional[int] = None) -> None:
        if difficulty not in TASK_CONFIG:
            raise ValueError(f"difficulty must be one of {list(TASK_CONFIG)}")
        self.difficulty = difficulty
        self._cfg = TASK_CONFIG[difficulty]
        self._seed = seed if seed is not None else TASK_SEEDS[difficulty]

        # Populated by reset()
        self._G: Optional[nx.Graph] = None
        self._nodes: Optional[dict[str, GraphNode]] = None
        self._null_trajectory: Optional[list[float]] = None
        self._step_count: int = 0
        self._budget: int = STARTING_BUDGET
        self._episode_id: str = ""
        self._spread_rng: Optional[random.Random] = None
        self._done: bool = False
        self._termination_reason: Optional[str] = None
        self._prev_damage: float = 0.0
        self._prev_infected_count: int = 0
        self._isolated_subgraph_node_ids: set[str] = set()

    # ── OpenEnv API ────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> CascadeObservation:
        """Initialise a fresh episode and return the first observation."""
        if seed is not None:
            self._seed = seed

        G, node_list, null_traj = build_graph(self.difficulty, seed=self._seed)

        self._G = G
        self._nodes = {n.node_id: n for n in node_list}
        self._null_trajectory = null_traj
        self._step_count = 0
        self._budget = STARTING_BUDGET
        self._episode_id = str(uuid.uuid4())
        self._done = False
        self._termination_reason = None
        self._spread_rng = random.Random(self._seed + 999)

        expected = self._cfg["max_steps"] + 1
        if len(null_traj) != expected:
            raise ValueError(f"null_trajectory length {len(null_traj)} != {expected}")
        if self._cfg["n_initial_infected"] > 0 and null_traj[0] <= 0.0:
            raise ValueError("Null trajectory initial damage must be > 0 when infections exist.")

        for nid, gn in self._nodes.items():
            self._G.nodes[nid]["data"] = gn

        self._update_at_risk_flags()
        self._prev_damage = self._weighted_damage()
        self._prev_infected_count = sum(
            1 for gn in self._nodes.values() if gn.status == "CONFIRMED_INFECTED"
        )

        return self._build_observation(
            last_action_effect="Episode started. Misinformation is spreading. Contain it."
        )

    def step(self, action: CascadeAction) -> CascadeObservation:
        """Apply one action, advance physics, return next observation."""
        if self._done:
            raise RuntimeError("Episode is done. Call reset() before stepping.")

        self._step_count += 1
        valid, err = self._validate_action(action)

        if not valid:
            self._advance_physics()
            return self._finalize_step(
                f"INVALID ACTION \u2014 {err} Step consumed. Infection spread unchecked."
            )

        effect = self._apply_action(action)
        self._budget -= ACTION_COSTS[action.action_type]
        self._advance_physics()
        return self._finalize_step(effect)

    def state(self) -> dict:
        """Full internal state for orchestrator replay / judge auditing."""
        if self._G is None:
            return {"error": "Environment not initialised. Call reset() first."}

        by_status: dict[str, list[str]] = {
            s: [] for s in ("susceptible", "latent", "confirmed_infected",
                           "inoculated", "quarantined")
        }
        for nid, gn in self._nodes.items():
            by_status[gn.status.lower()].append(nid)

        return CascadeState(
            episode_id=self._episode_id,
            difficulty=self.difficulty,
            step_count=self._step_count,
            budget=self._budget,
            max_steps=self._cfg["max_steps"],
            **by_status,
            total_damage_accumulated=self._prev_damage,
            null_trajectory=self._null_trajectory,
            graph_node_link_data=nx.node_link_data(self._G),
            termination_reason=self._termination_reason,
        ).model_dump()

    # ── Action Validation ──────────────────────────────────────────────

    def _validate_action(self, action: CascadeAction) -> tuple[bool, str]:
        if action.action_type == "WAIT":
            return True, ""

        node_id = action.target_node_id

        if node_id not in self._nodes:
            return False, f"Node '{node_id}' does not exist in this graph."

        gn   = self._nodes[node_id]
        cost = ACTION_COSTS[action.action_type]

        if cost > self._budget:
            return (
                False,
                f"Insufficient budget. '{action.action_type}' costs {cost}, "
                f"remaining: {self._budget}.",
            )

        if action.action_type == "FACTCHECK":
            if gn.status in ("QUARANTINED", "INOCULATED"):
                return False, f"Cannot FACTCHECK '{node_id}': already {gn.status}."
            if gn.status == "CONFIRMED_INFECTED":
                return False, (
                    f"Cannot FACTCHECK '{node_id}': already CONFIRMED_INFECTED. "
                    f"Use QUARANTINE."
                )

        if action.action_type == "QUARANTINE":
            if gn.status in ("QUARANTINED", "INOCULATED"):
                return False, f"Cannot QUARANTINE '{node_id}': already {gn.status}."

        if action.action_type == "INOCULATE":
            if gn.status in ("QUARANTINED", "INOCULATED"):
                return False, f"Cannot INOCULATE '{node_id}': already {gn.status}."

        if action.action_type == "BOOST_CORRECTION":
            if gn.status in ("QUARANTINED", "INOCULATED"):
                return False, f"Cannot BOOST '{node_id}': already {gn.status}."

        return True, ""

    # ── Action Application ─────────────────────────────────────────────

    def _apply_action(self, action: CascadeAction) -> str:
        """Apply a validated action. Returns shaped feedback string."""
        if action.action_type == "WAIT":
            return "WAIT — no action taken. Infection spreading."

        node_id = action.target_node_id
        gn      = self._nodes[node_id]

        if action.action_type == "FACTCHECK":
            return self._do_factcheck(gn)

        if action.action_type == "QUARANTINE":
            return self._do_quarantine(gn)

        if action.action_type == "INOCULATE":
            return self._do_inoculate(gn)

        if action.action_type == "BOOST_CORRECTION":
            return self._do_boost(gn)

        return "Unknown action. No effect."

    def _do_factcheck(self, gn: GraphNode) -> str:
        true_status = gn.status  # SUSCEPTIBLE, LATENT, or AT_RISK equivalent
        if gn.status == "LATENT":
            return (
                f"FACTCHECK {gn.node_id}: Node is LATENT — already infected but not yet "
                f"showing. It has been spreading for {gn.turns_at_risk} turn(s). "
                f"Recommend immediate QUARANTINE (cost 5)."
            )
        return (
            f"FACTCHECK {gn.node_id}: Node status confirmed {gn.status}. "
            f"Influence: {gn.influence_score:.2f}. "
            f"Spread prob: {gn.effective_spread_prob:.2f}."
        )

    def _do_quarantine(self, gn: GraphNode) -> str:
        old_status = gn.status
        gn.status  = "QUARANTINED"
        gn.at_risk = False
        # Remove all edges from this node in the graph
        neighbors = list(self._G.neighbors(gn.node_id))
        self._G.remove_edges_from([(gn.node_id, nb) for nb in neighbors])
        # Update AT_RISK flags for former neighbors
        self._update_at_risk_flags()
        return (
            f"QUARANTINE {gn.node_id}: Node removed from network (was {old_status}). "
            f"{len(neighbors)} edge(s) severed. "
            f"Influence suppressed: {gn.influence_score:.2f}."
        )

    def _do_inoculate(self, gn: GraphNode) -> str:
        """
        Inoculate a node.
        If node is LATENT (secretly infected): Option 3 — botched procedure.
            Budget consumed. Node flips to CONFIRMED_INFECTED (status reveal).
            Agent overpaid for a FACTCHECK and still needs to QUARANTINE.
        If node is SUSCEPTIBLE or AT_RISK: inoculation succeeds, node immune.
        """
        if gn.status == "LATENT":
            # Botched procedure — Option 3
            gn.status  = "CONFIRMED_INFECTED"
            gn.at_risk = False
            self._update_at_risk_flags()
            return (
                f"INOCULATE {gn.node_id}: BOTCHED — node was already latent-infected. "
                f"Procedure failed. Infection now CONFIRMED_INFECTED and active. "
                f"3 budget consumed. Recommend QUARANTINE next turn (cost 5)."
            )

        gn.status  = "INOCULATED"
        gn.at_risk = False
        gn.boost_turns_remaining = 0
        self._update_at_risk_flags()
        return (
            f"INOCULATE {gn.node_id}: Node permanently immunised. "
            f"Cannot be infected. Influence score: {gn.influence_score:.2f}."
        )

    def _do_boost(self, gn: GraphNode) -> str:
        # Reset (not stack) boost counter on re-application
        gn.boost_turns_remaining  = BOOST_DURATION
        gn.effective_spread_prob  = round(gn.base_spread_prob * gn.virality_modifier * 0.5, 4)
        return (
            f"BOOST_CORRECTION {gn.node_id}: Spread probability halved to "
            f"{gn.effective_spread_prob:.2f} for {BOOST_DURATION} turns."
        )

    # ── Physics Engine ─────────────────────────────────────────────────

    def _advance_physics(self) -> None:
        """Run one step of spread physics after the action has been applied."""
        self._advance_latent_timers()
        self._spread()
        self._decrement_boost_timers()
        self._inject_external_seed()
        self._update_at_risk_flags()

    def _advance_latent_timers(self) -> None:
        """Increment turns_at_risk for all LATENT nodes.
        Flip LATENT → CONFIRMED_INFECTED when turns_at_risk >= LATENT_DURATION.
        """
        for gn in self._nodes.values():
            if gn.status == "LATENT":
                gn.turns_at_risk += 1
                if gn.turns_at_risk >= LATENT_DURATION:
                    gn.status = "CONFIRMED_INFECTED"

    def _spread(self) -> None:
        """Probabilistic spread from CONFIRMED_INFECTED and LATENT nodes.
        Synchronous update: collect all new exposures before applying any.
        First infector wins — subsequent exposures to the same node are ignored.
        """
        new_exposures: dict[str, str] = {}  # {target_node_id: infector_node_id}

        for node_id, gn in self._nodes.items():
            if gn.status not in ("CONFIRMED_INFECTED", "LATENT"):
                continue
            for neighbor_id in self._G.neighbors(node_id):
                nb = self._nodes[neighbor_id]
                if nb.status != "SUSCEPTIBLE":
                    continue
                if neighbor_id in new_exposures:
                    continue  # already being exposed this step
                if self._spread_rng.random() < gn.effective_spread_prob:
                    new_exposures[neighbor_id] = node_id

        for target_id, infector_id in new_exposures.items():
            target = self._nodes[target_id]
            target.status           = "LATENT"
            target.at_risk          = True
            target.turns_at_risk    = 0
            target.infected_neighbor = infector_id  # first exposure only

    def _decrement_boost_timers(self) -> None:
        """Decrement boost counters. Restore spread prob when counter hits 0."""
        for gn in self._nodes.values():
            if gn.boost_turns_remaining > 0:
                gn.boost_turns_remaining -= 1
                if gn.boost_turns_remaining == 0:
                    gn.effective_spread_prob = round(
                        gn.base_spread_prob * gn.virality_modifier, 4
                    )

    def _inject_external_seed(self) -> None:
        """Hard difficulty only: inject a new infection seed every ext_interval steps."""
        ext_interval = self._cfg.get("external_seed_interval")
        if not ext_interval:
            return
        if self._step_count % ext_interval != 0:
            return

        susceptible = [
            gn for gn in self._nodes.values()
            if gn.status == "SUSCEPTIBLE"
        ]
        if not susceptible:
            return

        target = self._spread_rng.choice(susceptible)
        target.status        = "LATENT"
        target.at_risk       = True
        target.turns_at_risk = 0
        target.infected_neighbor = "__external__"

    def _update_at_risk_flags(self) -> None:
        """Recompute at_risk on all SUSCEPTIBLE nodes.
        A node is AT_RISK if it has at least one CONFIRMED_INFECTED or LATENT neighbor.
        First infected_neighbor is preserved — not overwritten by later infected neighbors.
        
        FIXED: LATENT nodes now have their at_risk flags re-evaluated in case their
        infector was quarantined (edge removed). This prevents stale at_risk observations.
        """
        for node_id, gn in self._nodes.items():
            if gn.status not in ("SUSCEPTIBLE", "LATENT"):
                # Terminal nodes: CONFIRMED_INFECTED, INOCULATED, QUARANTINED are never AT_RISK
                gn.at_risk = False
                continue

            infected_neighbors = [
                nb_id for nb_id in self._G.neighbors(node_id)
                if self._nodes[nb_id].status in ("CONFIRMED_INFECTED", "LATENT")
            ]

            if infected_neighbors:
                gn.at_risk = True
                if gn.infected_neighbor is None:
                    gn.infected_neighbor = infected_neighbors[0]
                # Clock does not reset on additional infected neighbors
            else:
                gn.at_risk = False
                # Do not clear infected_neighbor — preserves first-exposure record

    # ── Termination & Scoring ──────────────────────────────────────────

    def _detect_isolated_infected_subgraphs(self) -> list[set[str]]:
        """Detect connected components containing CONFIRMED_INFECTED nodes that are
        unreachable from other infected nodes (graph fragmentation from quarantines).
        
        Returns list of isolated infected component IDs.
        Used for alerting about silent spread in disconnected subgraphs.
        """
        confirmed_nodes = {
            gn.node_id for gn in self._nodes.values()
            if gn.status == "CONFIRMED_INFECTED"
        }
        
        if not confirmed_nodes:
            return []
        
        visited = set()
        isolated_components = []
        
        for start_node in confirmed_nodes:
            if start_node in visited:
                continue
            
            # BFS to find connected component
            component = set()
            queue = [start_node]
            while queue:
                node_id = queue.pop(0)
                if node_id in visited:
                    continue
                visited.add(node_id)
                component.add(node_id)
                
                for neighbor_id in self._G.neighbors(node_id):
                    if neighbor_id not in visited:
                        queue.append(neighbor_id)
            
            # Check if this component is isolated from other infected nodes
            has_external_contact = any(
                neighbor_id not in component
                for node_id in component
                for neighbor_id in self._G.neighbors(node_id)
            )
            
            if not has_external_contact and len(component) < len(confirmed_nodes):
                isolated_components.append(component)
        
        return isolated_components

    def _check_termination(self) -> tuple[bool, Optional[str]]:
        """Check all three termination conditions in priority order."""
        confirmed = [
            gn for gn in self._nodes.values()
            if gn.status == "CONFIRMED_INFECTED"
        ]
        total = len(self._nodes)

        # 1. Perfect eradication
        if len(confirmed) == 0:
            return True, "eradication"

        # 2. Saturation mercy kill
        infected_ratio = len(confirmed) / total
        if infected_ratio >= SATURATION_MERCY_THRESHOLD and self._budget == 0:
            return True, "saturation"

        # 3. Max steps reached
        if self._step_count >= self._cfg["max_steps"]:
            return True, "max_steps"

        return False, None

    def _compute_counterfactual_score(self) -> float:
        """Terminal score: how much damage did the agent prevent vs null agent?

        score = (null_final_damage_adjusted - agent_damage_adjusted) / null_final_damage_adjusted
        
        Adjustment: Subtract damage from isolated infected subgraphs (unreachable islands).
        These nodes cannot spread beyond their island boundary, so they don't represent
        true containment failure — they're mechanically isolated by the graph topology.
        
        The null trajectory includes this damage because it was computed at reset() time
        before any quarantines fragmented the graph. We retroactively adjust by calculating
        current isolated damage and subtracting from both null_final and agent_damage.
        """
        null_final = self._null_trajectory[-1]
        agent_dmg = self._weighted_damage()

        # Calculate damage in isolated infected islands
        isolated_damage = sum(
            gn.influence_score
            for node_id in self._isolated_subgraph_node_ids
            for gn in [self._nodes[node_id]]
            if gn.status in ("LATENT", "CONFIRMED_INFECTED")
        )

        # Adjust both baselines by isolated damage
        null_final_adjusted = max(0.0, null_final - isolated_damage)
        agent_dmg_adjusted = max(0.0, agent_dmg - isolated_damage)

        if null_final_adjusted <= 0:
            # Keep terminal score strictly inside (0, 1) for evaluator compatibility.
            return round(clamp_score(1.0), 4)

        score = (null_final_adjusted - agent_dmg_adjusted) / null_final_adjusted
        return round(clamp_score(score), 4)

    def _weighted_damage(self) -> float:
        """Current weighted infection mass: sum of influence scores for
        LATENT + CONFIRMED_INFECTED nodes."""
        return sum(
            gn.influence_score
            for gn in self._nodes.values()
            if gn.status in ("LATENT", "CONFIRMED_INFECTED")
        )

    # ── Observation Builder ────────────────────────────────────────────

    def _finalize_step(self, base_effect: str) -> CascadeObservation:
        """Check termination, compute reward, build and return observation."""
        # Update isolated subgraph tracking (for counterfactual score adjustment)
        isolated_components = self._detect_isolated_infected_subgraphs()
        self._isolated_subgraph_node_ids = set()
        for component in isolated_components:
            self._isolated_subgraph_node_ids.update(component)

        done, reason = self._check_termination()
        self._done = done
        self._termination_reason = reason

        current_damage = self._weighted_damage()
        t = min(self._step_count, len(self._null_trajectory) - 1)
        null_delta = self._null_trajectory[t] - self._null_trajectory[max(t - 1, 0)]
        damage_delta = current_damage - self._prev_damage
        null_final = max(self._null_trajectory[-1], 1e-6)
        step_reward = (null_delta - damage_delta) / null_final
        step_reward = round(max(min(step_reward, 0.25), -0.25), 4)

        reward = step_reward
        effect = base_effect

        containment_signal = (
            f" | Δdamage this step: {damage_delta:+.3f} "
            f"(null expected: +{null_delta:.3f}). "
            f"Step reward: {step_reward:+.4f}."
        )
        effect += containment_signal

        if done:
            terminal_score = self._compute_counterfactual_score()
            reward = terminal_score
            effect += (
                f" | EPISODE ENDED ({reason}). "
                f"Null damage: {self._null_trajectory[-1]:.3f}. "
                f"Agent damage: {current_damage:.3f}. "
                f"Final score: {terminal_score:.4f}."
            )

        self._prev_damage = current_damage

        if isolated_components:
            effect += (
                f" | WARNING: {len(isolated_components)} isolated infected component(s) exists. "
                f"Spread is silent—no AT_RISK nodes surface. Silent damage continues."
            )

        return self._build_observation(
            last_action_effect=effect,
            reward=reward,
            done=done,
        )

    def _build_observation(
        self,
        last_action_effect: str,
        reward: float = 0.0,
        done: bool = False,
    ) -> CascadeObservation:
        """Construct CascadeObservation from current internal graph state."""
        all_nodes = list(self._nodes.values())

        # --- Top 10 by influence ---
        sorted_by_influence = sorted(
            all_nodes, key=lambda n: n.influence_score, reverse=True
        )
        top_10 = sorted_by_influence[:10]

        # --- Confirmed infected ---
        confirmed = [
            gn for gn in all_nodes
            if gn.status == "CONFIRMED_INFECTED"
        ]

        # --- AT_RISK nodes (all, not just top 10) ---
        at_risk = [
            gn for gn in all_nodes
            if gn.at_risk and gn.status in ("SUSCEPTIBLE", "LATENT")
        ]

        def to_summary(gn: GraphNode) -> NodeSummary:
            """Convert internal GraphNode to agent-visible NodeSummary.
            LATENT nodes appear as AT_RISK if they have a known infected neighbor,
            otherwise SUSCEPTIBLE. Never exposed as LATENT.
            """
            if gn.status == "LATENT":
                if gn.at_risk and gn.infected_neighbor:
                    visible_status = "AT_RISK"
                else:
                    visible_status = "SUSCEPTIBLE"
            elif gn.status == "SUSCEPTIBLE" and gn.at_risk:
                visible_status = "AT_RISK"
            else:
                visible_status = gn.status  # CONFIRMED_INFECTED, INOCULATED, QUARANTINED

            return NodeSummary(
                node_id          = gn.node_id,
                influence_score  = gn.influence_score,
                status           = visible_status,
                boost_active     = gn.boost_turns_remaining > 0,
                infected_neighbor = gn.infected_neighbor if visible_status == "AT_RISK" else None,
                turns_at_risk    = gn.turns_at_risk if visible_status == "AT_RISK" else None,
            )

        top_summaries       = [to_summary(gn) for gn in top_10]
        confirmed_summaries = [to_summary(gn) for gn in confirmed]
        at_risk_summaries   = [to_summary(gn) for gn in at_risk]

        # Count stats
        n_infected    = sum(1 for gn in all_nodes if gn.status == "CONFIRMED_INFECTED")
        n_inoculated  = sum(1 for gn in all_nodes if gn.status == "INOCULATED")
        n_quarantined = sum(1 for gn in all_nodes if gn.status == "QUARANTINED")

        # FIXED: Correctly calculate spread delta (how many new CONFIRMED_INFECTED this step)
        spread_delta = max(0, n_infected - self._prev_infected_count)
        self._prev_infected_count = n_infected  # Update for next step

        return CascadeObservation(
            top_nodes          = top_summaries,
            confirmed_infected = confirmed_summaries,
            at_risk_nodes      = at_risk_summaries,
            budget_remaining   = self._budget,
            step               = max(self._step_count, 1),
            max_steps          = self._cfg["max_steps"],
            total_nodes        = len(self._nodes),
            infected_count     = n_infected,
            inoculated_count   = n_inoculated,
            quarantined_count  = n_quarantined,
            spread_delta_last_step = spread_delta,  # FIXED: Now reflects actual spread
            last_action_effect = last_action_effect,
            reward             = reward,
            done               = done,
        )
