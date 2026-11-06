"""Das Modell: Handwerte, Aufzählung als Gegenprobe (alle Zuordnungen und Reihenfolgen), Held–Karp, Ungarische Methode, wechselweise Optimierung, Relaxation."""

import itertools

import numpy as np
import pytest

import zrf_constants as C
import zrf_model as M


def _all(inst):
    """Alle (Kosten, σ, Reihenfolge) durch Aufzählung."""
    return [(M.tour_cost(inst, sg, o), sg, o) for sg in itertools.permutations(range(inst.T)) for o in itertools.permutations(range(inst.P))]


# --- Handbeispiel ---------------------------------------------------------------------------------------------------------------------------

def test_hand_cost_of_the_optimum_by_hand(hand):
    """σ = (0, 1), Reihenfolge ABCD. Ruhe (5, −2) → Platz 0 (1, −2) = 4; → A (0, 0) = max(1, 2) = 2; A → Platz 0 = 2; → B (4, 4) = 6; B → Platz 1 (9, −2) = 6; → C (10, 0) = 2;
    C → Platz 1 = 2; → D (9, 3) = 5; D → Ruhe = max(4, 5) = 5. Summe: 4 + 2 + 2 + 6 + 6 + 2 + 2 + 5 + 5 = 34."""
    assert [d for _, _, d in M.legs(hand, (0, 1), (0, 1, 2, 3))] == [4, 2, 2, 6, 6, 2, 2, 5, 5]
    assert M.tour_cost(hand, (0, 1), (0, 1, 2, 3)) == 34


def test_hand_fixpoint_by_hand(hand):
    """σ = (1, 0) (Typ 0 auf Platz 1 bei x = 9, Typ 1 auf Platz 0 bei x = 1), Reihenfolge D A C B. Ruhe → Platz 0 = 4; → D (9, 3) = 8; D → Platz 1 = 5; → A (0, 0) = 9;
    A → Platz 0 = 2; → C (10, 0) = 9; C → Platz 1 = 2; → B (4, 4) = 6; B → Ruhe = 6. Summe: 4 + 8 + 5 + 9 + 2 + 9 + 2 + 6 + 6 = 51."""
    assert [d for _, _, d in M.legs(hand, (1, 0), (3, 0, 2, 1))] == [4, 8, 5, 9, 2, 9, 2, 6, 6]
    assert M.tour_cost(hand, (1, 0), (3, 0, 2, 1)) == 51


def test_hand_joint_and_alternating(hand):
    cost, sigma, order = M.joint_exact(hand)
    assert cost == 34 and list(sigma) == [0, 1] and list(order) == [0, 1, 2, 3]
    good = M.alternating(hand, (0, 1))
    bad = M.alternating(hand, (1, 0))
    assert good["cost"] == 34 and bad["cost"] == 51 and list(bad["sigma"]) == [1, 0] and list(bad["order"]) == [3, 0, 2, 1] and bad["rounds"] == 2


def test_hand_fixpoint_is_a_local_optimum_of_both_steps(hand):
    bad = M.alternating(hand, (1, 0))
    assert M.best_order(hand, bad["sigma"])[0] == bad["cost"] and M.best_assignment(hand, bad["order"])[0] >= bad["cost"] - 1e-9


# --- Aufzählung als Gegenprobe --------------------------------------------------------------------------------------------------------------

def test_held_karp_equals_enumeration_for_every_assignment(small):
    for inst in small:
        for sg in itertools.permutations(range(inst.T)):
            c, o = M.best_order(inst, sg)
            assert c == pytest.approx(min(M.tour_cost(inst, sg, p) for p in itertools.permutations(range(inst.P)))) and M.tour_cost(inst, sg, o) == pytest.approx(c)


def test_hungarian_equals_enumeration_for_every_order(small):
    for inst in small:
        for o in list(itertools.permutations(range(inst.P)))[::37]:
            c, sg = M.best_assignment(inst, o)
            assert c == pytest.approx(min(M.tour_cost(inst, s, o) for s in itertools.permutations(range(inst.T)))) and M.tour_cost(inst, sg, o) == pytest.approx(c)


def test_joint_exact_equals_full_enumeration(small):
    for inst in small:
        c, sg, o = M.joint_exact(inst)
        assert c == pytest.approx(min(x[0] for x in _all(inst))) and M.tour_cost(inst, sg, o) == pytest.approx(c)


def test_order_through_placements_equals_enumeration(small):
    for inst in small:
        c, o = M.order_through_placements(inst)
        brute = min(inst.dhp[p[0]] + sum(inst.dpp[a, b] for a, b in zip(p, p[1:])) + inst.dhp[p[-1]] for p in itertools.permutations(range(inst.P)))
        assert c == pytest.approx(brute) and sorted(o) == list(range(inst.P))


def test_legs_sum_to_the_tour_cost(small):
    for inst in small:
        for sg in list(itertools.permutations(range(inst.T)))[:2]:
            o = list(range(inst.P))
            assert sum(d for _, _, d in M.legs(inst, sg, o)) == pytest.approx(M.tour_cost(inst, sg, o))


# --- wechselweise Optimierung ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("kind", C.KINDS)
def test_alternating_never_increases_and_ends_at_a_fixpoint(kind):
    inst = M.make(kind, 8, 4, 100003)
    for sg in itertools.permutations(range(inst.T)):
        a = M.alternating(inst, sg)
        costs = [x for _, c1, c2 in a["trace"] for x in (c1, c2)]
        assert all(b <= a_ + 1e-9 for a_, b in zip(costs, costs[1:]))
        assert M.tour_cost(inst, a["sigma"], a["order"]) == pytest.approx(a["cost"])
        assert M.best_assignment(inst, a["order"])[0] >= a["cost"] - 1e-9 and M.best_order(inst, a["sigma"])[0] == pytest.approx(a["cost"])


def test_alternating_cost_is_never_below_the_joint_optimum():
    for kind in C.KINDS:
        inst = M.make(kind, 8, 4, 100004)
        j = M.joint_exact(inst)[0]
        for sg in itertools.permutations(range(inst.T)):
            assert M.alternating(inst, sg)["cost"] >= j - 1e-9


def test_basins_cover_every_start_and_contain_the_optimum_end():
    inst = M.make("clustered", 8, 4, 41)
    b = M.basins(inst)
    assert sum(b.values()) == 24 and min(b) >= M.joint_exact(inst)[0] - 1e-6


# --- Relaxation und Startzuordnungen --------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("kind", C.KINDS)
def test_lower_bound_is_below_the_optimum(kind):
    for s in range(10):
        inst = M.make(kind, 8, 4, 100000 + s)
        assert M.lower_bound(inst) <= M.joint_exact(inst)[0] + 1e-9


def test_start_assignments_are_permutations_and_deterministic():
    inst = M.make("random", 10, 5, 7)
    for f in (M.sigma_practice, M.sigma_centroid):
        assert sorted(f(inst)) == list(range(5))
    assert sorted(M.sigma_random(inst, 7)) == list(range(5)) and list(M.sigma_random(inst, 7)) == list(M.sigma_random(inst, 7))
    assert any(list(M.sigma_random(inst, s)) != list(M.sigma_random(inst, 7)) for s in range(8, 30))


def test_centroid_assignment_minimizes_the_leg_sum():
    """Die Schwerpunkt-Zuordnung minimiert Σ_i d(s_σ(typ i), p_i) über alle Zuordnungen."""
    for kind in C.KINDS:
        inst = M.make(kind, 8, 4, 100010)
        legsum = lambda sg: sum(inst.dps[i, sg[inst.typ[i]]] for i in range(inst.P))
        sg = M.sigma_centroid(inst)
        assert legsum(sg) == pytest.approx(min(legsum(s) for s in itertools.permutations(range(inst.T))))


def test_practice_puts_the_most_frequent_type_next_to_the_rest_position():
    inst = M.make("skewed", 10, 4, 100000)
    top = int(np.argmax(np.bincount(inst.typ, minlength=inst.T)))
    assert int(M.sigma_practice(inst)[top]) == int(np.argmin(inst.dhs))


# --- Erzeugung ------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("kind", C.KINDS)
@pytest.mark.parametrize("P,T", [(6, 3), (8, 4), (10, 5), (12, 4)])
def test_instances_are_valid_and_deterministic(kind, P, T):
    a, b = M.make(kind, P, T, 5), M.make(kind, P, T, 5)
    assert np.array_equal(a.pos, b.pos) and np.array_equal(a.typ, b.typ) and a.P == P and a.T == T
    assert set(a.typ.tolist()) == set(range(T)) and (a.pos >= 0).all() and (a.pos[:, 0] <= C.BOARD[0] + 1).all() and (a.pos[:, 1] <= C.BOARD[1] + 1).all()
    assert not np.array_equal(a.pos, M.make(kind, P, T, 6).pos)


def test_make_rejects_bad_arguments():
    with pytest.raises(ValueError):
        M.make("unknown", 8, 4, 1)
    with pytest.raises(ValueError):
        M.make("random", 3, 4, 1)


def test_the_families_differ_as_described():
    sk = M.make("skewed", 12, 4, 100000)
    cnt = np.bincount(sk.typ, minlength=4)
    assert cnt[0] == 1 + (12 - 4) // 2 and cnt[0] > cnt[1:].max()
    cl = M.make("clustered", 12, 4, 100000)
    spread = lambda inst: np.mean([inst.pos[inst.typ == t].std(axis=0).mean() for t in range(inst.T) if (inst.typ == t).sum() > 1])
    assert spread(cl) < spread(M.make("random", 12, 4, 100000))
