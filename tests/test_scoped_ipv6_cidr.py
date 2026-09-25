"""Regression tests for IPv6 interface-scope CIDR rejection.

An IPv6 *interface scope* (zone id) is introduced by ``%``, e.g.
``fe80::%eth0/128``.  The policy model has no notion of interface scope, and
the standard library keeps the zone on the parsed network: ``fe80::/128`` and
``fe80::%eth0/128`` hash the same (identical integer network address) yet
compare unequal.  Accepting both corrupted every hash-keyed same-CIDR group —
two distinct "winning" networks for one address range — while the decision map
derived from the stored CIDR strings only reported one action.

Every entry point accepting a user-supplied CIDR must therefore reject the
scoped form up front, in that entry's existing error envelope, and reject the
*whole* request (no persistence, no partial result):

* ``POST /api/analyze``                       -> 400 invalid_ruleset
* ``POST .../simulate`` (add/replace rule)    -> 400 invalid_ruleset
* ``POST .../merge-precheck`` (A or B rule)   -> 400 invalid_ruleset
* ``GET  .../decision-map?cidr=``             -> 400 invalid_cidr
* ``POST .../quarantine-plan`` (target cidr)  -> 400 invalid_cidr
* ``POST .../change-guard`` (authorization)   -> 400 invalid_change_guard

Ordinary unscoped IPv4/IPv6 CIDRs and same-CIDR priority adjudication remain
unchanged.
"""

from __future__ import annotations

import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from app import db
from app.analyzer import ValidationError, analyze_rules, strict_ip_network
from app.change_guard import ChangeGuardError, parse_guard_entries
from app.decision_map import DecisionMapError, parse_query_cidr

#: Several spellings of an interface-scope zone id (numeric ifindex and named,
#: different host/prefix placements) that must all be rejected.
SCOPED_CIDRS = [
    "fe80::%eth0/128",
    "fe80::%1/128",
    "fe80::1%eth0/128",
    "fe80:0:0:0:0:0:0:1%2/64",
]


@pytest.fixture()
def client() -> TestClient:
    fd, path = tempfile.mkstemp(suffix=".db", prefix="ppa-test-scope-")
    os.close(fd)
    os.remove(path)  # let sqlite create it
    db.reset_db(path)
    from app.main import app

    with TestClient(app) as c:
        yield c
    for suffix in ("", "-wal", "-shm"):
        try:
            os.remove(path + suffix)
        except OSError:
            pass


def _save(client: TestClient, rules: list[dict]) -> str:
    r = client.post("/api/analyze", json={"rules": rules})
    assert r.status_code == 200, r.text
    return r.json()["id"]


# ---------------------------------------------------------------------------
# Engine-level unit behavior
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cidr", SCOPED_CIDRS)
def test_strict_parser_rejects_interface_scope(cidr: str) -> None:
    with pytest.raises(ValueError, match="interface scope"):
        strict_ip_network(cidr)


@pytest.mark.parametrize(
    "cidr",
    ["0.0.0.0/0", "10.0.0.0/8", "::/0", "fe80::/64", "fe80::/128", "2001:db8::1/128"],
)
def test_strict_parser_accepts_unscoped_cidrs(cidr: str) -> None:
    assert str(strict_ip_network(cidr)) == cidr


@pytest.mark.parametrize("cidr", SCOPED_CIDRS)
def test_analyzer_rejects_scoped_rule_cidr(cidr: str) -> None:
    with pytest.raises(ValidationError, match="invalid CIDR"):
        analyze_rules([{"rule_id": "r1", "cidr": cidr, "action": "permit"}])


def test_scoped_and_unscoped_pair_cannot_enter_engine_together() -> None:
    # The exact pair from the bug report: previously both became "winners"
    # (distinct_cidrs == 2) while sharing one integer address; now the batch is
    # rejected as a whole instead of producing the distorted coverage.
    with pytest.raises(ValidationError, match="interface scope"):
        analyze_rules(
            [
                {"rule_id": "r1", "cidr": "fe80::/128", "action": "permit"},
                {"rule_id": "r2", "cidr": "fe80::%eth0/128", "action": "deny"},
            ]
        )


def test_unscoped_same_cidr_priority_adjudication_still_works() -> None:
    # Same-CIDR duplicate handling (smaller priority number wins) must keep
    # functioning on the unscoped form that the scoped one imitated.
    result = analyze_rules(
        [
            {"rule_id": "r1", "cidr": "fe80::/128", "action": "deny", "priority": 20},
            {"rule_id": "r2", "cidr": "fe80::/128", "action": "permit", "priority": 10},
        ]
    )
    by_id = {r["rule_id"]: r for r in result["rules"]}
    assert result["summary"]["distinct_cidrs"] == 1
    assert by_id["r1"]["role"] == "duplicate"
    assert by_id["r2"]["role"] == "winner"
    assert by_id["r2"]["status"] == "active"
    assert by_id["r2"]["effective_cidrs"] == ["fe80::/128"]


@pytest.mark.parametrize("cidr", SCOPED_CIDRS)
def test_query_parser_rejects_interface_scope(cidr: str) -> None:
    with pytest.raises(DecisionMapError, match="interface scope"):
        parse_query_cidr(cidr)


def test_guard_entry_parser_rejects_interface_scope() -> None:
    with pytest.raises(ChangeGuardError) as exc_info:
        parse_guard_entries(
            [{"cidr": "fe80::%eth0/128", "from_action": "permit", "to_action": "deny"}]
        )
    assert len(exc_info.value.issues) == 1
    assert "interface scope" in exc_info.value.issues[0]


# ---------------------------------------------------------------------------
# POST /api/analyze — rule submission
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cidr", SCOPED_CIDRS)
def test_analyze_rejects_scoped_rule_cidr_with_400(client: TestClient, cidr: str) -> None:
    r = client.post(
        "/api/analyze",
        json={"rules": [{"rule_id": "r1", "cidr": cidr, "action": "permit"}]},
    )
    assert r.status_code == 400, (cidr, r.status_code)
    body = r.json()
    assert body["error"] == "invalid_ruleset"
    assert "interface scope" in body["detail"]
    # Whole-batch rejection: nothing was saved and no partial result exists.
    assert client.get("/api/analyses").json()["total"] == 0


def test_analyze_with_one_scoped_rule_among_valid_rejects_entire_batch(
    client: TestClient,
) -> None:
    r = client.post(
        "/api/analyze",
        json={
            "rules": [
                {"rule_id": "r1", "cidr": "10.0.0.0/8", "action": "permit"},
                {"rule_id": "r2", "cidr": "fe80::/128", "action": "permit"},
                {"rule_id": "r3", "cidr": "fe80::%eth0/128", "action": "deny"},
            ]
        },
    )
    assert r.status_code == 400
    assert r.json()["error"] == "invalid_ruleset"
    assert "interface scope" in r.json()["detail"]
    # The valid rules in the batch must not have been persisted either.
    assert client.get("/api/analyses").json()["total"] == 0


def test_analyze_save_false_also_rejects_scoped_cidr(client: TestClient) -> None:
    r = client.post(
        "/api/analyze",
        json={
            "save": False,
            "rules": [{"rule_id": "r1", "cidr": "fe80::%eth0/64", "action": "deny"}],
        },
    )
    assert r.status_code == 400
    assert r.json()["error"] == "invalid_ruleset"


# ---------------------------------------------------------------------------
# POST .../simulate — rule change preview
# ---------------------------------------------------------------------------
def test_simulate_add_with_scoped_cidr_rejects_whole_batch(client: TestClient) -> None:
    aid = _save(client, [{"rule_id": "a", "cidr": "10.0.0.0/8", "action": "permit"}])
    r = client.post(
        f"/api/analyses/{aid}/simulate",
        json={"add": [{"rule_id": "x", "cidr": "fe80::%eth0/128", "action": "deny"}]},
    )
    assert r.status_code == 400
    assert r.json()["error"] == "invalid_ruleset"
    assert "interface scope" in r.json()["detail"]
    # The preview must not have mutated the base version.
    assert len(client.get(f"/api/analyses/{aid}").json()["rules"]) == 1


def test_simulate_replace_with_scoped_cidr_rejects_whole_batch(client: TestClient) -> None:
    aid = _save(
        client,
        [
            {"rule_id": "a", "cidr": "2001:db8::/32", "action": "permit"},
            {"rule_id": "b", "cidr": "fe80::/64", "action": "deny"},
        ],
    )
    r = client.post(
        f"/api/analyses/{aid}/simulate",
        json={
            "replace": [
                {"rule_id": "b", "cidr": "fe80::%eth0/64", "action": "permit"}
            ]
        },
    )
    assert r.status_code == 400
    assert r.json()["error"] == "invalid_ruleset"
    # Base rule b is untouched.
    rules = {x["rule_id"]: x for x in client.get(f"/api/analyses/{aid}").json()["rules"]}
    assert rules["b"]["cidr"] == "fe80::/64"
    assert rules["b"]["action"] == "deny"


# ---------------------------------------------------------------------------
# POST .../merge-precheck — two-batch merge precheck
# ---------------------------------------------------------------------------
def test_merge_precheck_rejects_scoped_cidr_in_either_batch(client: TestClient) -> None:
    aid = _save(
        client,
        [
            {"rule_id": "a", "cidr": "10.0.0.0/8", "action": "permit"},
            {"rule_id": "k", "cidr": "10.9.0.0/16", "action": "permit"},
        ],
    )
    # The other batch is independently legal (keeps at least one rule) so the
    # scoped CIDR, not the batch contract, is what must be reported.
    for side, payload in (
        ("a", {"add": [{"rule_id": "x", "cidr": "fe80::%eth0/128", "action": "deny"}]}),
        ("b", {"add": [{"rule_id": "y", "cidr": "fe80::%1/128", "action": "permit"}]}),
    ):
        other = {"remove": [{"rule_id": "k"}]}
        body = {"a": payload, "b": other} if side == "a" else {"a": other, "b": payload}
        r = client.post(f"/api/analyses/{aid}/merge-precheck", json=body)
        assert r.status_code == 400, (side, r.status_code, r.text)
        assert r.json()["error"] == "invalid_ruleset"
        assert "interface scope" in r.json()["detail"]


def test_merge_precheck_replace_with_scoped_cidr_is_rejected(client: TestClient) -> None:
    aid = _save(
        client,
        [
            {"rule_id": "a", "cidr": "10.0.0.0/8", "action": "permit"},
            {"rule_id": "b", "cidr": "fe80::/64", "action": "deny"},
        ],
    )
    r = client.post(
        f"/api/analyses/{aid}/merge-precheck",
        json={
            "a": {"replace": [{"rule_id": "b", "cidr": "fe80::%eth0/64", "action": "permit"}]},
            "b": {"add": [{"rule_id": "y", "cidr": "10.20.0.0/16", "action": "deny"}]},
        },
    )
    assert r.status_code == 400
    assert r.json()["error"] == "invalid_ruleset"


# ---------------------------------------------------------------------------
# GET .../decision-map — CIDR query
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cidr", SCOPED_CIDRS)
def test_decision_map_rejects_scoped_query(client: TestClient, cidr: str) -> None:
    aid = _save(client, [{"rule_id": "r1", "cidr": "fe80::/128", "action": "permit"}])
    r = client.get(f"/api/analyses/{aid}/decision-map", params={"cidr": cidr})
    assert r.status_code == 400, (cidr, r.status_code)
    assert r.json()["error"] == "invalid_cidr"
    assert "interface scope" in r.json()["detail"]


def test_decision_map_validates_scoped_query_before_version_lookup(
    client: TestClient,
) -> None:
    r = client.get(
        "/api/analyses/an_0000000000000000/decision-map",
        params={"cidr": "fe80::%eth0/128"},
    )
    assert r.status_code == 400
    assert r.json()["error"] == "invalid_cidr"


def test_decision_map_unscoped_query_reports_single_action(client: TestClient) -> None:
    # The saved version only ever contains unscoped CIDRs; a normal query must
    # still resolve fe80::/128 to exactly one block/action.
    aid = _save(client, [{"rule_id": "r1", "cidr": "fe80::/128", "action": "permit"}])
    r = client.get(f"/api/analyses/{aid}/decision-map", params={"cidr": "fe80::/128"})
    assert r.status_code == 200, r.text
    blocks = r.json()["blocks"]
    assert len(blocks) == 1
    assert blocks[0]["cidr"] == "fe80::/128"
    assert blocks[0]["action"] == "permit"
    assert blocks[0]["winner_rule_id"] == "r1"


# ---------------------------------------------------------------------------
# POST .../quarantine-plan — temporary ban target
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cidr", SCOPED_CIDRS)
def test_quarantine_plan_rejects_scoped_target(client: TestClient, cidr: str) -> None:
    aid = _save(client, [{"rule_id": "r1", "cidr": "fe80::/64", "action": "permit"}])
    r = client.post(
        f"/api/analyses/{aid}/quarantine-plan",
        json={"cidr": cidr, "rule_id_prefix": "q"},
    )
    assert r.status_code == 400, (cidr, r.status_code)
    assert r.json()["error"] == "invalid_cidr"
    assert "interface scope" in r.json()["detail"]


def test_quarantine_plan_validates_scoped_target_before_version_lookup(
    client: TestClient,
) -> None:
    r = client.post(
        "/api/analyses/an_0000000000000000/quarantine-plan",
        json={"cidr": "fe80::%eth0/128", "rule_id_prefix": "q"},
    )
    assert r.status_code == 400
    assert r.json()["error"] == "invalid_cidr"


# ---------------------------------------------------------------------------
# POST .../change-guard — change authorization
# ---------------------------------------------------------------------------
def test_change_guard_rejects_scoped_authorization(client: TestClient) -> None:
    base_id = _save(client, [{"cidr": "fe80::/64", "action": "permit"}])
    target_id = _save(client, [{"cidr": "fe80::/64", "action": "deny"}])
    r = client.post(
        f"/api/analyses/{base_id}/change-guard/{target_id}",
        json={
            "authorizations": [
                {"cidr": "fe80::%eth0/64", "from_action": "permit", "to_action": "deny"}
            ]
        },
    )
    assert r.status_code == 400
    body = r.json()
    assert body["error"] == "invalid_change_guard"
    assert len(body["issues"]) == 1
    assert "interface scope" in body["issues"][0]


def test_change_guard_rejects_whole_set_when_one_authorization_is_scoped(
    client: TestClient,
) -> None:
    base_id = _save(client, [{"cidr": "10.0.0.0/8", "action": "permit"}])
    target_id = _save(
        client,
        [
            {"cidr": "10.0.0.0/8", "action": "permit"},
            {"cidr": "10.1.0.0/16", "action": "deny"},
        ],
    )
    r = client.post(
        f"/api/analyses/{base_id}/change-guard/{target_id}",
        json={
            "authorizations": [
                {"cidr": "10.1.0.0/16", "from_action": "permit", "to_action": "deny"},
                {"cidr": "fe80::%eth0/128", "from_action": "permit", "to_action": "deny"},
            ]
        },
    )
    assert r.status_code == 400
    body = r.json()
    assert body["error"] == "invalid_change_guard"
    # All-or-nothing: the valid grant must not produce a partial pass/fail result.
    assert any("interface scope" in issue for issue in body["issues"])
    assert "passed" not in body


# ---------------------------------------------------------------------------
# Ordinary IPv4/IPv6 CIDRs keep working end to end
# ---------------------------------------------------------------------------
def test_ordinary_v4_and_v6_rules_still_analyze(client: TestClient) -> None:
    r = client.post(
        "/api/analyze",
        json={
            "rules": [
                {"rule_id": "v4", "cidr": "10.0.0.0/8", "action": "permit"},
                {"rule_id": "v6a", "cidr": "fe80::/64", "action": "deny"},
                {"rule_id": "v6b", "cidr": "fe80::/128", "action": "permit", "priority": 10},
            ]
        },
    )
    assert r.status_code == 200, r.text
    data = r.json()
    by_id = {rule["rule_id"]: rule for rule in data["rules"]}
    assert by_id["v4"]["status"] == "active"
    # fe80::/128 is a strict subnet of fe80::/64 and wins by prefix length.
    assert by_id["v6a"]["status"] == "partial"
    assert by_id["v6b"]["status"] == "active"
