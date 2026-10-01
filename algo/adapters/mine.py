"""Greedy construction with repeated one-site local search."""

import random
import time

from adapter import Solver, dist_table


class MySolver(Solver):
    def solve(self, instance, submit_candidate):
        started = time.perf_counter()
        table = dist_table(instance)
        weights = instance.weights
        n, m, k = instance.size, len(instance.sites), instance.k
        penalty = instance.penalty
        # Clustered instances benefit from a separate repeatable restart stream.
        seed_offset = 7 if instance.profile["family"] == "clustered" else 0
        rng = random.Random(int(instance.digest, 16) + seed_offset)

        def greedy(randomized=False):
            opened = []
            selected = set()
            current = [penalty] * n
            for _ in range(k):
                gains = []
                for s in range(m):
                    if s in selected:
                        continue
                    gain = sum(w * (before - min(before, row[s]))
                               for w, before, row in zip(weights, current, table))
                    gains.append((gain, s))
                gains.sort(reverse=True)
                if randomized and len(gains) > 1:
                    # Restrict the choice to strong greedy candidates while varying
                    # the construction enough to escape one greedy path.
                    limit = min(10, len(gains))
                    pick = rng.randrange(limit)
                    s = gains[pick][1]
                else:
                    s = gains[0][1]
                selected.add(s)
                opened.append(s)
                current = [min(before, row[s])
                           for before, row in zip(current, table)]
            return opened, sum(w * d for w, d in zip(weights, current))

        def best_swap(opened, current_cost):
            opened_set = set(opened)
            best_cost = current_cost
            best_move = None
            for pos in range(k):
                remaining_sites = opened[:pos] + opened[pos + 1:]
                base = [min((row[s] for s in remaining_sites), default=penalty)
                        for row in table]
                for s in range(m):
                    if s in opened_set:
                        continue
                    candidate_cost = sum(
                        w * min(before, row[s])
                        for w, before, row in zip(weights, base, table)
                    )
                    if candidate_cost < best_cost:
                        best_cost = candidate_cost
                        best_move = (pos, s)
            return best_cost, best_move

        def improve(opened, current_cost):
            # Apply best improvements until a one-swap local optimum or the
            # evaluator's final checkpoint is close.
            while time.perf_counter() - started < 4.55:
                value, move = best_swap(opened, current_cost)
                if move is None:
                    break
                opened[move[0]] = move[1]
                current_cost = value
                submit_candidate({"sites": list(opened)})
            return opened, current_cost

        best, best_cost = greedy()
        submit_candidate({"sites": list(best)})
        best, best_cost = improve(best, best_cost)

        # Spend remaining time on greedy starts with controlled variation,
        # polishing each with the same deterministic exchange search.
        while time.perf_counter() - started < 4.55:
            candidate, value = greedy(randomized=True)
            if value < best_cost:
                best, best_cost = candidate, value
                submit_candidate({"sites": list(best)})
            candidate, value = improve(candidate, value)
            if value < best_cost:
                best, best_cost = candidate, value

        return {"sites": list(best)}
