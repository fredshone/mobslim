"""MATSim's default (Charypar-Nagel) scoring function; see :mod:`mobslim.scoring`."""

from typing import Mapping, Optional, Sequence

import numpy as np

from mobslim.attributes import PersonAttributes
from mobslim.experience import (
    ExperiencedActivity,
    ExperiencedLeg,
    ExperiencedPlan,
)
from mobslim.scoring.params import (
    ACT_RELATIVE,
    ACT_T0,
    A,
    M,
    ScoringParams,
    act_row,
    act_table,
    mode_table,
    walk_table,
    zero_utility_duration,
)
from mobslim.scoring.person import (
    PersonParams,
    Subpopulation,
    resolve_person_params,
)
from mobslim.timeutils import HOUR
from mobslim.trajectories import Trajectories

_T, _PRIO = A["typical_duration"], A["priority"]
_OPEN, _CLOSE = A["opening_time"], A["closing_time"]
_LATEST, _EARLIEST, _MIN = A["latest_start_time"], A["earliest_end_time"], A["minimal_duration"]
_CONST, _BETA_TT = M["constant"], M["marginal_utility_of_traveling"]
_BETA_D, _RATE = M["marginal_utility_of_distance"], M["monetary_distance_rate"]
_DAILY, _DAILY_MONEY = M["daily_constant"], M["daily_monetary_constant"]


def _priced(beta: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """As MATSim, distance only counts if it is priced (an unknown, NaN
    distance does not spoil the score then)."""
    return np.where(beta != 0.0, beta * np.where(np.isnan(distance), 0.0, distance), 0.0)


def activity_kernel(rows: np.ndarray, arrival: np.ndarray, departure: np.ndarray,
                    performing, waiting, late_arrival, early_departure) -> np.ndarray:
    """MATSim's ``CharyparNagelActivityScoring.calcActScore`` for many
    intervals at once.

    ``rows`` is ``(n, ACT_COLUMNS)`` activity params (see
    :func:`~mobslim.scoring.params.act_table`), one per interval; the
    utilities [util/h] are scalars or one per interval. Terms whose
    constraints no row sets are skipped.
    """
    t_typ, t_0 = rows[:, _T], rows[:, ACT_T0]
    open_, close = rows[:, _OPEN], rows[:, _CLOSE]
    has_open, has_close = ~np.isnan(open_), ~np.isnan(close)
    windows = has_open.any() or has_close.any()
    if windows:
        start = np.where(has_open & (arrival < open_), open_, arrival)
        end = np.where(has_close & (close < departure), close, departure)
        closed = has_open & has_close & ((open_ > departure) | (close < arrival))
        start = np.where(closed, departure, start)
        end = np.where(closed, departure, end)
    else:
        start, end = arrival, departure
    duration = end - start

    beta_perf = np.asarray(performing) / HOUR  # util / s
    with np.errstate(divide="ignore", invalid="ignore"):
        log_part = beta_perf * t_typ * np.log(duration / t_0)
        linear = beta_perf * t_typ / t_0 * (duration - t_0)
    u = np.where(duration >= t_0, log_part, linear)
    if windows:
        u += np.asarray(waiting) / HOUR * (np.maximum(start - arrival, 0.0)
                                           + np.maximum(departure - end, 0.0))
    latest = rows[:, _LATEST]
    late = ~np.isnan(latest) & (start > latest)
    if late.any():
        u += np.asarray(late_arrival) / HOUR * np.where(late, start - latest, 0.0)
    earliest, min_dur = rows[:, _EARLIEST], rows[:, _MIN]
    early = ~np.isnan(earliest) & (end < earliest)
    short = ~np.isnan(min_dur) & (duration < min_dur)
    if early.any() or short.any():
        u += np.asarray(early_departure) / HOUR * (np.where(early, earliest - end, 0.0)
                                                   + np.where(short, min_dur - duration, 0.0))
    return u


def _check_known(rows: np.ndarray, codes: np.ndarray, act_types: Sequence[str]) -> None:
    unknown = np.isnan(rows[:, _T])
    if unknown.any():
        t = act_types[int(codes[np.argmax(unknown)])]
        raise KeyError(f"No ActivityParams for activity type '{t}'.")


def _activity_intervals(acts: Sequence[ExperiencedActivity], day: float):
    """``(type, arrival, departure)`` of each scored activity, as MATSim:
    middle activities as performed; first and last wrapped overnight into
    one ``[last.start, first.end + day]`` if their types match, else
    ``[0, first.end]`` and ``[last.start, day]``. A lone activity lasts the day."""
    if not acts:
        return
    if len(acts) == 1:
        yield acts[0].type, 0.0, day
        return
    first, last = acts[0], acts[-1]
    for a in acts[1:-1]:
        if a.start is not None and a.end is not None:
            yield a.type, a.start, a.end
    first_end = first.end if first.end is not None else day
    last_start = last.start if last.start is not None else day
    if first.type == last.type:
        yield first.type, last_start, first_end + day
    else:
        yield first.type, 0.0, first_end
        yield last.type, last_start, day


class CharyparNagelScorer:
    """MATSim's default scoring (see :mod:`mobslim.scoring`).

    ``subpopulations`` maps a value of the ``subpopulation_attribute``
    person attribute to the :class:`ScoringParams` of those persons (MATSim's
    ``scoringParameters`` sets); persons without the attribute use
    ``params``. A person whose subpopulation has no params is an error, as
    in MATSim.

    ``person_rules`` then adjust each person's parameters from their
    attributes (see :mod:`mobslim.scoring.person`). With subpopulations, a
    :class:`~mobslim.scoring.person.Subpopulation` rule runs first unless
    ``person_rules`` contains one.
    """

    def __init__(self, params: ScoringParams = ScoringParams(),
                 subpopulations: Optional[Mapping[str, ScoringParams]] = None,
                 subpopulation_attribute: str = "subpopulation",
                 person_rules: Sequence = ()):
        self.params = params
        self.subpopulations = dict(subpopulations or {})
        self.subpopulation_attribute = subpopulation_attribute
        self.person_rules = tuple(person_rules)
        self.sets: tuple[ScoringParams, ...] = (params, *self.subpopulations.values())
        self.set_names: tuple = (None, *self.subpopulations)
        auto = ()
        if self.subpopulations and not any(isinstance(r, Subpopulation) for r in self.person_rules):
            auto = (Subpopulation(subpopulation_attribute),)
        self._rules = auto + self.person_rules
        self._mode_tables: dict = {}
        self._act_tables: dict = {}
        self._act_rows: dict = {}  # per-plan path: {id(set): {type: row}}
        self._walk = walk_table(self.sets)
        self._walk_modes = np.array([p.walk_mode for p in self.sets], dtype=object)
        self._person_cache = None  # (attributes or n_persons, PersonParams)
        self._subscorers: dict = {}

    def for_subpopulation(self, name) -> "CharyparNagelScorer":
        """A scorer with the params of subpopulation ``name`` (``None``: this one)."""
        if name is None:
            return self
        if name not in self.subpopulations:
            raise KeyError(f"No ScoringParams for subpopulation {name!r}.")
        if name not in self._subscorers:
            self._subscorers[name] = CharyparNagelScorer(self.subpopulations[name])
        return self._subscorers[name]

    # -- person parameters ------------------------------------------------------
    def person_params(self, attributes: Optional[PersonAttributes],
                      n_persons: Optional[int] = None) -> PersonParams:
        """Every person's parameters (cached for the last ``attributes``,
        which never change during a run). Without ``attributes`` (then
        ``n_persons`` is needed) the rules see an empty attribute table."""
        if attributes is None:
            if n_persons is None:
                raise ValueError("person_params needs attributes or n_persons.")
            key, attributes = int(n_persons), None
        else:
            if n_persons is not None and attributes.n_persons != n_persons:
                raise ValueError(f"Person attributes have {attributes.n_persons} rows "
                                 f"for {n_persons} persons.")
            key = attributes
        cached = self._person_cache
        if cached is not None and (cached[0] is key or (isinstance(key, int) and cached[0] == key)):
            return cached[1]
        n = key if attributes is None else attributes.n_persons
        if self._rules:
            attrs = PersonAttributes.empty(n) if attributes is None else attributes
            pp = resolve_person_params(attrs, self.sets, self.set_names, self._rules)
        else:
            pp = PersonParams(self.sets, self.set_names, np.zeros(n, dtype=np.int32))
        self._person_cache = (key, pp)
        return pp

    def params_for(self, attributes: Mapping) -> ScoringParams:
        """The params of one person with ``attributes`` (a dict), resolved on
        their own: rules that depend on the population (such as
        :class:`IncomeDependentMoney` without a ``reference``) see only them."""
        if not self._rules or not attributes:
            return self.params
        attrs = PersonAttributes(1, {k: [v] for k, v in attributes.items()})
        return resolve_person_params(attrs, self.sets, self.set_names, self._rules).params_of(0)

    def transit_weights(self, attributes: Optional[PersonAttributes], mode: str = "pt"):
        """Per-person transit router weights matching this scorer
        (:class:`~mobslim.transit.PersonRaptorWeights`), as MATSim's
        ``DefaultRaptorParametersForPerson``. Persons whose weights equal
        those of ``params`` get ``-1`` (the router's own, which the
        controller derives from ``params``). ``None`` if nobody differs."""
        from mobslim.transit.raptor import PersonRaptorWeights, RaptorWeights

        if attributes is None:
            return None
        pp = self.person_params(attributes)
        if pp.is_uniform:
            return None
        perf = pp.value("performing")
        walk = np.empty(pp.n_persons)
        for code, p in enumerate(pp.sets):
            on = pp.set_index == code
            if on.any():
                walk[on] = pp.value(f"modes.{p.walk_mode}.marginal_utility_of_traveling")[on]
        table = np.column_stack([
            pp.value(f"modes.{mode}.marginal_utility_of_traveling") - perf,
            pp.value("marginal_utility_of_waiting_pt") - perf,
            walk - perf,
            pp.value("utility_of_line_switch"),
        ])
        rows, inverse = np.unique(table, axis=0, return_inverse=True)
        default = RaptorWeights.from_scoring(self.params, mode)
        weights = [RaptorWeights(*(float(x) for x in row)) for row in rows]
        keep = [w != default for w in weights]
        new_code = np.cumsum(keep) - 1
        codes = np.where(np.array(keep)[inverse], new_code[inverse], -1)
        out = PersonRaptorWeights([w for w, k in zip(weights, keep) if k], codes)
        return out if out.any else None

    # -- per plan (the reference) -----------------------------------------------
    def score(self, plan: ExperiencedPlan, attributes: Optional[Mapping] = None,
              params: Optional[ScoringParams] = None) -> float:
        """One plan's utility, in plain Python. ``params`` (e.g.
        ``person_params(...).params_of(i)``) or else ``attributes`` (the
        person's attributes as a dict, see :meth:`params_for`) pick the
        parameters; default ``params``."""
        p = params if params is not None else (
            self.params_for(attributes) if attributes else self.params)
        utility = sum(self.leg_utility(leg, p) for leg in plan.legs)
        utility += sum(p.mode(m).daily_constant + p.mode(m).daily_monetary_constant * p.marginal_utility_of_money
                       for m in {leg.mode for leg in plan.legs})
        intervals = list(_activity_intervals(plan.activities, p.day_length))
        if intervals:
            types = [t for t, _, _ in intervals]
            arrival = np.array([a for _, a, _ in intervals], dtype=float)
            departure = np.array([d for _, _, d in intervals], dtype=float)
            utility += float(self._activity_scores(p, types, arrival, departure).sum())
        if plan.stuck:
            utility += p.stuck_penalty
        return utility

    def leg_utility(self, leg: ExperiencedLeg, params: Optional[ScoringParams] = None) -> float:
        p = self.params if params is None else params
        m = p.mode(leg.mode)
        tt = leg.travel_time
        if tt is None:  # stuck mid-leg: penalty is applied at plan level
            return m.constant
        if leg.rides is not None:
            return self._pt_leg_utility(p, leg, m, tt)
        return (m.constant + m.marginal_utility_of_traveling * tt / HOUR
                + self._distance_utility(p, m, leg.distance))

    @staticmethod
    def _distance_utility(p: ScoringParams, m, distance: float) -> float:
        beta = m.marginal_utility_of_distance + m.monetary_distance_rate * p.marginal_utility_of_money
        return beta * distance if beta != 0.0 else 0.0

    def _pt_leg_utility(self, p: ScoringParams, leg: ExperiencedLeg, m, tt: float) -> float:
        w = p.mode(p.walk_mode)
        wait = ride = ride_dist = 0.0
        for rd in leg.rides:
            wait += rd.wait_time or 0.0
            ride += rd.in_vehicle_time or 0.0
            ride_dist += rd.distance
        walk = tt - wait - ride
        return (m.constant + m.marginal_utility_of_traveling * ride / HOUR
                + p.marginal_utility_of_waiting_pt * wait / HOUR
                + w.marginal_utility_of_traveling * walk / HOUR
                + self._distance_utility(p, w, leg.distance - ride_dist)
                + self._distance_utility(p, m, ride_dist)
                + p.utility_of_line_switch * max(len(leg.rides) - 1, 0))

    def _activity_scores(self, p: ScoringParams, types: Sequence[str], arrival: np.ndarray,
                         departure: np.ndarray) -> np.ndarray:
        if any(p is s for s in self.sets):  # cache rows of the scorer's own sets
            cache = self._act_rows.setdefault(id(p), {})
            rows = []
            for t in types:
                row = cache.get(t)
                if row is None:
                    row = cache[t] = act_row(p.activities.get(t, p.default_activity))
                rows.append(row)
        else:
            rows = [act_row(p.activities.get(t, p.default_activity)) for t in types]
        rows = np.array(rows, dtype=float).reshape(len(types), -1)
        _check_known(rows, np.arange(len(types)), types)
        return activity_kernel(rows, arrival, departure, p.performing, p.waiting,
                               p.late_arrival, p.early_departure)

    def activity_utility(self, act_type: str, duration: float) -> float:
        """Utility of performing ``act_type`` for ``duration`` seconds, from
        midnight (so opening and closing times apply to ``[0, duration]``)."""
        return self.activity_score(act_type, 0.0, duration)

    def activity_score(self, act_type: str, arrival: float, departure: float) -> float:
        """MATSim's ``calcActScore``: utility of being at ``act_type`` from
        ``arrival`` to ``departure`` (seconds of the day, may exceed 24 h)."""
        return float(self._activity_scores(self.params, [act_type], np.array([float(arrival)]),
                                           np.array([float(departure)]))[0])

    # -- all persons at once ----------------------------------------------------
    def _mode_table(self, pp: PersonParams, modes: tuple) -> np.ndarray:
        if pp.sets is not self.sets:
            return mode_table(pp.sets, modes)
        table = self._mode_tables.get(modes)
        if table is None:
            table = self._mode_tables[modes] = mode_table(self.sets, modes)
        return table

    def _act_table(self, pp: PersonParams, act_types: tuple) -> np.ndarray:
        if pp.sets is not self.sets:
            return act_table(pp.sets, act_types)
        table = self._act_tables.get(act_types)
        if table is None:
            table = self._act_tables[act_types] = act_table(self.sets, act_types)
        return table

    def score_trajectories(self, tr: Trajectories,
                           attributes: Optional[PersonAttributes] = None) -> np.ndarray:
        """:meth:`score` for every person at once, from the mobsim's arrays:
        one vectorised pass whatever the number of subpopulations and
        person-specific parameters. Same utility as :meth:`score` with
        ``params=person_params(attributes).params_of(i)`` (up to
        floating-point summation order)."""
        return self.score_with(tr, self.person_params(attributes, tr.plans.n_plans))

    def score_with(self, tr: Trajectories, pp: PersonParams) -> np.ndarray:
        """Every person's utility under ``pp``."""
        b = tr.plans
        n = b.n_plans
        if pp.n_persons != n:
            raise ValueError(f"PersonParams has {pp.n_persons} persons for {n} plans.")
        set_index = pp.set_index
        utility = np.zeros(n)

        # -- legs that were started
        if b.n_legs:
            leg_plan = b.plan_of_leg()
            leg_set = set_index[leg_plan]
            P = self._mode_table(pp, b.modes)[leg_set, b.leg_mode]  # (n_legs, fields), a copy
            for name, fld, col in pp.of_kind("modes"):
                if name in b.modes:
                    on = np.flatnonzero(b.leg_mode == b.modes.index(name))
                    v = col[leg_plan[on]]
                    ok = ~np.isnan(v)
                    P[on[ok], M[fld]] = v[ok]
            money = pp.scalar("marginal_utility_of_money", leg_plan)
            const, beta_tt = P[:, _CONST], P[:, _BETA_TT]
            beta_d = P[:, _BETA_D] + P[:, _RATE] * money
            departed = ~np.isnan(tr.leg_departure)
            travel = tr.leg_arrival - tr.leg_departure
            arrived = ~np.isnan(travel)  # implies departed
            travel0 = np.where(arrived, travel, 0.0)
            distance = tr.leg_distance
            leg_u = const + beta_tt * travel0 / HOUR + _priced(beta_d, distance)
            if tr.transit_modes:
                pt = np.flatnonzero(b.mode_legs(tr.transit_modes))
                if len(pt):
                    leg_u[pt] = self._pt_utility(tr, pp, pt, leg_plan, leg_set, const, beta_tt,
                                                 beta_d, money, travel0)
            leg_u = np.where(arrived, leg_u, const)
            utility += np.bincount(leg_plan[departed], weights=leg_u[departed], minlength=n)

            # daily constants: once per (person, mode used)
            daily = P[:, _DAILY] + P[:, _DAILY_MONEY] * money
            if np.any(daily != 0.0):
                legs = np.flatnonzero(departed)
                _, once = np.unique(leg_plan[legs] * len(b.modes) + b.leg_mode[legs], return_index=True)
                legs = legs[once]
                utility += np.bincount(leg_plan[legs], weights=daily[legs], minlength=n)

        # -- activities: every scored interval, then one kernel call
        if b.n_acts:
            utility += self._activity_utility(tr, pp, n)

        stuck = np.flatnonzero(tr.stuck)
        utility[stuck] += pp.scalar("stuck_penalty", stuck)
        return utility

    def _pt_utility(self, tr, pp, pt, leg_plan, leg_set, const, beta_tt, beta_d, money, travel0):
        """Utility of pt legs ``pt`` (indices) if arrived; see :mod:`mobslim.scoring`."""
        b = tr.plans
        lor = b.leg_of_ride()
        n_legs = b.n_legs
        board, alight = tr.ride_board, tr.ride_alight
        wait_r = board - tr.ride_wait_start
        ride_r = alight - board
        wait = np.bincount(lor, weights=np.where(np.isnan(wait_r), 0.0, wait_r), minlength=n_legs)[pt]
        ride = np.bincount(lor, weights=np.where(np.isnan(ride_r), 0.0, ride_r), minlength=n_legs)[pt]
        ride_d = np.bincount(lor, weights=tr.ride_distance(), minlength=n_legs)[pt]
        n_rides = np.diff(b.ride_offsets)[pt]
        plan = leg_plan[pt]
        if pp.sets is self.sets:
            walk_rows, walk_modes = self._walk, self._walk_modes
        else:
            walk_rows = walk_table(pp.sets)
            walk_modes = np.array([p.walk_mode for p in pp.sets], dtype=object)
        W = walk_rows[leg_set[pt]]
        overrides = pp.of_kind("modes")
        if overrides:
            leg_walk_mode = walk_modes[leg_set[pt]]
            for name, fld, col in overrides:
                on = np.flatnonzero(leg_walk_mode == name)
                v = col[plan[on]]
                ok = ~np.isnan(v)
                W[on[ok], M[fld]] = v[ok]
        walk_beta_d = W[:, _BETA_D] + W[:, _RATE] * (money[pt] if np.ndim(money) else money)
        return (const[pt] + beta_tt[pt] * ride / HOUR
                + pp.scalar("marginal_utility_of_waiting_pt", plan) * wait / HOUR
                + W[:, _BETA_TT] * (travel0[pt] - wait - ride) / HOUR
                + _priced(walk_beta_d, tr.leg_distance[pt] - ride_d)
                + _priced(beta_d[pt], ride_d)
                + pp.scalar("utility_of_line_switch", plan) * np.maximum(n_rides - 1, 0))

    def _activity_utility(self, tr: Trajectories, pp: PersonParams, n: int) -> np.ndarray:
        b = tr.plans
        types = b.act_type
        act_plan = b.plan_of_act()
        first = b.act_offsets[:-1]
        has_acts = np.diff(b.act_offsets) > 0
        reached = ~np.isnan(tr.act_start)
        reached[first[has_acts]] = True
        n_reached = np.bincount(act_plan[reached], minlength=n)
        last = first + n_reached - 1  # reached activities are a prefix
        day = np.broadcast_to(pp.scalar("day_length", slice(None)), (n,))

        idx = np.arange(b.n_acts)
        mid = np.flatnonzero(reached & (idx > first[act_plan]) & (idx < last[act_plan])
                             & ~np.isnan(tr.act_start) & ~np.isnan(tr.act_end))
        single = np.flatnonzero(n_reached == 1)
        multi = np.flatnonzero(n_reached > 1)
        f, la = first[multi], last[multi]
        first_end = tr.act_end[f]
        first_end = np.where(np.isnan(first_end), day[multi], first_end)
        last_start = tr.act_start[la]
        last_start = np.where(np.isnan(last_start), day[multi], last_start)
        same = types[f] == types[la]
        diff = ~same
        wrap, split = multi[same], multi[diff]
        k_single, k_split = len(single), len(split)

        owner = np.concatenate([act_plan[mid], single, wrap, split, split])
        codes = np.concatenate([types[mid], types[first[single]], types[f[same]], types[f[diff]],
                                types[la[diff]]])
        arrival = np.concatenate([tr.act_start[mid], np.zeros(k_single), last_start[same],
                                  np.zeros(k_split), last_start[diff]])
        departure = np.concatenate([tr.act_end[mid], day[single], first_end[same] + day[wrap],
                                    first_end[diff], day[split]])

        rows = self._act_table(pp, b.act_types)[pp.set_index[owner], codes]
        _check_known(rows, codes, b.act_types)
        for name, fld, col in pp.of_kind("activities"):
            if name in b.act_types:
                on = np.flatnonzero(codes == b.act_types.index(name))
                v = col[owner[on]]
                ok = ~np.isnan(v)
                rows[on[ok], A[fld]] = v[ok]
                if fld in ("typical_duration", "priority"):
                    r = rows[on[ok]]
                    rows[on[ok], ACT_T0] = zero_utility_duration(r[:, _T], r[:, _PRIO],
                                                                 r[:, ACT_RELATIVE] == 1.0)
        u = activity_kernel(rows, arrival, departure, *(pp.scalar(f, owner) for f in
                            ("performing", "waiting", "late_arrival", "early_departure")))
        return np.bincount(owner, weights=u, minlength=n)

