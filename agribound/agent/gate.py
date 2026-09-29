"""
Human confirmation gate for agent-proposed runs.

No plan is executed unless a :class:`ConfirmationGate` holds an **unused
approval bound to that plan's hash**:

- An approval is created either by the gate's *callback* (called from
  :meth:`ConfirmationGate.request_approval`) or explicitly with
  :meth:`ConfirmationGate.record_approval` (used by the MCP server after an
  elicitation, or by Python code that approves a plan it has inspected).
- :meth:`ConfirmationGate.authorize_execution` recomputes the plan hash from
  the stored configuration and the *current* input files, requires a matching
  unused approval, marks it used (approvals are single-use) and counts the
  execution against ``max_executions`` (default 1 per session). When the
  hash check fails, the unused approvals of that hash are revoked, and
  :meth:`ConfirmationGate.request_approval` always asks the callback anew,
  so an approval that could not be used never authorises a later run.
- :func:`check_plan_current` performs the same hash check on its own; the
  ``execute_plan`` tools call it *before* asking the reviewer, so nobody is
  asked to approve a plan whose inputs already changed.

Callbacks
---------
A callback receives the :class:`~agribound.agent.plans.Plan` and returns
``True`` to approve. Ready-made callbacks:

- :func:`prompt_confirm` -- prints the full resolved configuration and
  requires the reviewer to type ``yes`` (anything else denies).
- :func:`deny_all` -- denies every plan (the default for non-interactive
  sessions).

:func:`default_callback` picks :func:`prompt_confirm` when standard input is
an interactive terminal and :func:`deny_all` otherwise.
"""

from __future__ import annotations

import getpass
import logging
import sys
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, TextIO

from agribound.agent.errors import (
    ExecutionDeniedError,
    ExecutionLimitError,
    ExecutionNotApprovedError,
    PlanChangedError,
)

logger = logging.getLogger(__name__)

ConfirmCallback = Callable[[Any], bool]
"""``callback(plan) -> bool``; *True* approves the plan."""


def _utc_now() -> str:
    import datetime as _dt

    return _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _current_user() -> str:
    try:
        return getpass.getuser()
    except Exception:  # pragma: no cover - platform dependent
        return "unknown"


@dataclass
class Approval:
    """An approval of one plan hash (single-use)."""

    plan_id: str
    plan_hash: str
    approver: str
    method: str
    approved_utc: str
    used_utc: str | None = None
    revoked_utc: str | None = None
    revoked_reason: str | None = None

    @property
    def used(self) -> bool:
        return self.used_utc is not None

    @property
    def usable(self) -> bool:
        """Neither used nor revoked."""
        return self.used_utc is None and self.revoked_utc is None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Denial:
    """A recorded denial."""

    plan_id: str
    plan_hash: str
    method: str
    reason: str
    denied_utc: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Ready-made callbacks
# ---------------------------------------------------------------------------


def deny_all(plan: Any) -> bool:
    """Deny every plan (non-interactive default)."""
    return False


deny_all.method = "auto-deny (non-interactive)"  # type: ignore[attr-defined]


def prompt_confirm(
    plan: Any,
    *,
    input_fn: Callable[[str], str] | None = None,
    out: TextIO | None = None,
) -> bool:
    """Show the full plan and approve only if the reviewer types ``yes``.

    Parameters
    ----------
    plan : Plan
        Plan to review.
    input_fn : callable or None
        Reads the answer; it is called with an empty prompt string because
        the question is printed to *out* (default: the built-in
        :func:`input`).
    out : text stream or None
        Where the plan and the question are printed (default ``sys.stderr``,
        so standard output stays free for results and a redirected standard
        output does not hide the question).

    Returns
    -------
    bool
        *True* only for the exact answer ``yes`` (case-insensitive, surrounding
        whitespace ignored).
    """
    stream = out if out is not None else sys.stderr
    print("", file=stream)
    print("=" * 72, file=stream)
    print("The agent asks to run the following plan.", file=stream)
    print("=" * 72, file=stream)
    print(plan.render(), file=stream)
    print("=" * 72, file=stream)
    print("Type 'yes' to run this plan (anything else cancels): ", end="", file=stream)
    stream.flush()
    reader = input_fn if input_fn is not None else input
    try:
        answer = reader("")
    except (EOFError, KeyboardInterrupt):
        return False
    return str(answer).strip().lower() == "yes"


prompt_confirm.method = "typed 'yes' at the terminal prompt"  # type: ignore[attr-defined]


def default_callback() -> ConfirmCallback:
    """Return :func:`prompt_confirm` for an interactive terminal, else :func:`deny_all`."""
    try:
        interactive = sys.stdin is not None and sys.stdin.isatty()
    except (AttributeError, ValueError):
        interactive = False
    return prompt_confirm if interactive else deny_all


def check_plan_current(plan: Any) -> None:
    """Raise :class:`PlanChangedError` unless *plan* still matches its hash.

    Recomputes the hash from the stored configuration and the *current*
    input fingerprints (:meth:`Plan.current_hash`). An input that can no
    longer be fingerprinted (for example a deleted study-area or reference
    file) also raises :class:`PlanChangedError`.
    """
    try:
        current = plan.current_hash()
    except (OSError, ValueError, TypeError) as exc:
        raise PlanChangedError(
            f"Plan {plan.plan_id}: its inputs could not be fingerprinted again "
            f"({type(exc).__name__}: {exc}). Propose a new plan; it needs a new approval."
        ) from exc
    if current != plan.plan_hash:
        raise PlanChangedError(
            f"Plan {plan.plan_id} no longer matches its hash (the configuration, the "
            "study-area geometry or an input file changed since it was proposed). "
            "Propose a new plan; it needs a new approval."
        )


def _callback_method(callback: ConfirmCallback) -> str:
    method = getattr(callback, "method", None)
    if method:
        return str(method)
    name = getattr(callback, "__qualname__", None) or type(callback).__name__
    return f"python callback {name}"


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


class ConfirmationGate:
    """Single-use, hash-bound approvals and a hard execution limit.

    Parameters
    ----------
    callback : callable or None
        ``callback(plan) -> bool`` consulted by every :meth:`request_approval`
        call. *None* uses :func:`default_callback`.
    max_executions : int
        Maximum number of plans this gate lets run (default 1).
    approver : str or None
        Name recorded with callback approvals (default: the OS user name).

    Examples
    --------
    >>> gate = ConfirmationGate(callback=lambda plan: plan.config["year"] == 2024)
    """

    def __init__(
        self,
        callback: ConfirmCallback | None = None,
        *,
        max_executions: int = 1,
        approver: str | None = None,
    ) -> None:
        if int(max_executions) < 0:
            raise ValueError(f"max_executions must be >= 0, got {max_executions}")
        self.callback: ConfirmCallback = callback if callback is not None else default_callback()
        self.max_executions = int(max_executions)
        self.approver = approver
        self.approvals: list[Approval] = []
        self.denials: list[Denial] = []
        self.executions = 0
        # Serialises bookkeeping; MCP runs tool calls in worker threads.
        self._lock = threading.RLock()

    # -- queries ---------------------------------------------------------------

    @property
    def remaining_executions(self) -> int:
        return max(0, self.max_executions - self.executions)

    def _unused_approval(self, plan: Any) -> Approval | None:
        for approval in self.approvals:
            if (
                approval.usable
                and approval.plan_id == plan.plan_id
                and approval.plan_hash == plan.plan_hash
            ):
                return approval
        return None

    def _revoke(self, plan: Any, reason: str) -> int:
        """Revoke every usable approval of *plan*'s hash; return how many (lock held)."""
        n = 0
        for approval in self.approvals:
            if approval.usable and approval.plan_hash == plan.plan_hash:
                approval.revoked_utc = _utc_now()
                approval.revoked_reason = reason
                n += 1
        if n:
            logger.info("Revoked %d unused approval(s) of plan %s: %s", n, plan.plan_id, reason)
        return n

    def check_limit(self) -> None:
        """Raise :class:`ExecutionLimitError` if no executions remain."""
        if self.executions >= self.max_executions:
            raise ExecutionLimitError(
                f"This session already ran {self.executions} plan(s) "
                f"(max_executions={self.max_executions}). Any further run needs a new request "
                "from the human user."
            )

    # -- approvals -------------------------------------------------------------

    def record_approval(self, plan: Any, *, approver: str, method: str) -> Approval:
        """Record an approval obtained outside the callback (e.g. MCP elicitation).

        Parameters
        ----------
        plan : Plan
            The approved plan; the approval is bound to ``plan.plan_hash``.
        approver : str
            Who approved (free text, recorded in the transcript).
        method : str
            How the approval was obtained (recorded in the transcript).

        Returns
        -------
        Approval
        """
        approval = Approval(
            plan_id=plan.plan_id,
            plan_hash=plan.plan_hash,
            approver=str(approver),
            method=str(method),
            approved_utc=_utc_now(),
        )
        with self._lock:
            self.approvals.append(approval)
        logger.info("Plan %s approved by %s (%s)", plan.plan_id, approver, method)
        return approval

    def record_denial(self, plan: Any, *, method: str, reason: str) -> Denial:
        """Record a denial (for the transcript)."""
        denial = Denial(
            plan_id=plan.plan_id,
            plan_hash=plan.plan_hash,
            method=str(method),
            reason=str(reason),
            denied_utc=_utc_now(),
        )
        with self._lock:
            self.denials.append(denial)
        logger.info("Plan %s denied (%s): %s", plan.plan_id, method, reason)
        return denial

    def request_approval(self, plan: Any) -> Approval:
        """Ask the callback to approve *plan* and return the new approval.

        The callback is asked on every call. An earlier approval of the same
        plan hash that was never used (for example because
        :meth:`authorize_execution` refused it) is revoked first, so it can
        never authorise a later run without a new answer from the reviewer.

        Raises
        ------
        ExecutionLimitError
            If no executions remain (the reviewer is not asked).
        ExecutionDeniedError
            If the callback returns anything but *True* or raises.
        """
        with self._lock:
            self.check_limit()
            self._revoke(plan, reason="superseded by a new approval request")
        method = _callback_method(self.callback)
        try:
            decision = self.callback(plan)
        except Exception as exc:
            self.record_denial(plan, method=method, reason=f"callback raised {exc!r}")
            raise ExecutionDeniedError(
                f"Plan {plan.plan_id} was not approved: the confirmation callback failed ({exc})."
            ) from exc
        if decision is True:
            return self.record_approval(
                plan, approver=self.approver or _current_user(), method=method
            )
        self.record_denial(plan, method=method, reason="not approved by the reviewer")
        raise ExecutionDeniedError(
            f"Plan {plan.plan_id} was not approved ({method}). Do not modify the plan to obtain "
            "approval; report the plan and its limitations to the user and stop."
        )

    def authorize_execution(self, plan: Any) -> Approval:
        """Consume the approval for *plan* immediately before it runs.

        Recomputes the plan hash from the stored configuration and the
        current input fingerprints, then marks the matching approval used and
        counts the execution.

        Raises
        ------
        PlanChangedError
            If the recomputed hash differs from ``plan.plan_hash`` or the inputs
            can no longer be fingerprinted (:func:`check_plan_current`). The
            unused approvals of that hash are revoked: the reviewer approved
            what they were shown, which is no longer what would run.
        ExecutionLimitError
            If no executions remain.
        ExecutionNotApprovedError
            If there is no unused approval bound to the plan hash.
        """
        try:
            check_plan_current(plan)
        except PlanChangedError as exc:
            with self._lock:
                self._revoke(plan, reason=f"plan changed before it ran: {exc}")
            raise
        with self._lock:  # check, consume and count atomically
            self.check_limit()
            approval = self._unused_approval(plan)
            if approval is None:
                raise ExecutionNotApprovedError(
                    f"Plan {plan.plan_id} (sha256 {plan.plan_hash[:12]}...) has no unused approval."
                )
            approval.used_utc = _utc_now()
            self.executions += 1
        return approval

    def to_dict(self) -> dict[str, Any]:
        """Transcript view of the gate's state."""
        return {
            "callback": _callback_method(self.callback),
            "max_executions": self.max_executions,
            "executions": self.executions,
            "approvals": [a.to_dict() for a in self.approvals],
            "denials": [d.to_dict() for d in self.denials],
        }


__all__ = [
    "Approval",
    "ConfirmCallback",
    "ConfirmationGate",
    "Denial",
    "check_plan_current",
    "default_callback",
    "deny_all",
    "prompt_confirm",
]
