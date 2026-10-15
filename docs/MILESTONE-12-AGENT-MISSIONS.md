# Milestone 12 — NTSA-X bounded plan / observe / verify controller

**Status:** Deterministic research controller that coordinates existing
Milestone 11 skills. It is NOT a trained autonomous thinker, a coding model,
a fully interactive browser operator, or a proven English-fluent assistant.

## Mission types

| Goal | Steps and observable outcomes |
| --- | --- |
| repository.audit | Git status, DevOps inventory, root file listing, optional README contents |
| code.inspect | source read, Git diff, optional exact source search |
| tests.diagnose | Git status, then approved Python unit-test run; nonzero exit code means failure |
| research.inspect | allowlisted public HTTPS text fetch; optional approved read-only browser fallback if fetched text is shorter than 160 characters |

The controller uses tool results to verify **shapes and exit codes**, not
to assert that code is correct or that a website is truthful. An HTTP
result is untrusted data. For browser fallback, the webpage itself cannot
alter the URL, tool names, steps or granted permissions. All tools the
controller *might* run are validated before the first action.

Maximum 7 potential steps; no arbitrary shell, Git writes, deploys, emails,
clicking or typing in browsers. No self-generated recursive plans. Any failure
stops the current mission with a receipt and a suggested operator-led
recovery step. No silent edit, retries, releases or external side effects.
This differs deliberately from an unbounded autonomous agent.

A versioned mission SHA-256 binds goal/inputs, exact workspace, exact
allowlisted hosts, controller version, and every possible step. For
consequential operations (tests, network fetch, browser), execution requires
BOTH a human-supplied exact digest (after reviewing the mission) and
an independent capability flag; requesting read-only actions has no
additional approval requirement. The local process itself is NOT an OS
sandbox and must be run under a non-admin user.

## First local mission — repository audit

Update the checked-out repository, then run:

    Set-Location -LiteralPath (Join-Path $HOME "KindredLabs\NTSA-X-GitHub")
    git pull --ff-only origin main
    py -3.12 -m unittest discover -s tests -p "test_agent_mission.py" -v
    New-Item -ItemType Directory -Force ".local\agent" | Out-Null

Propose a multi-step mission from a restricted English instruction:

    py -3.12 -m ntsa_x.agent_mission_intent --request "audit repository" --output ".local\agent\m12-repo-audit.json"

Inspect the actual mission and the no-action preview:

    Get-Content ".local\agent\m12-repo-audit.json"
    py -3.12 -m ntsa_x.agent_mission --workspace . --mission ".local\agent\m12-repo-audit.json"

Execute and print structured observations:

    py -3.12 -m ntsa_x.agent_mission --workspace . --mission ".local\agent\m12-repo-audit.json" --execute

For repeat invocations, reuse the same mission JSON (the intent generator
refuses to overwrite it). The command does not modify Git or project files.
Only the explicit CLI arguments supply tool capabilities.

## Source investigation

    py -3.12 -m ntsa_x.agent_mission_intent --request "inspect code ntsa_x/cog_fu.py for denoise" --output ".local\agent\m12-code-inspect.json"
    py -3.12 -m ntsa_x.agent_mission --workspace . --mission ".local\agent\m12-code-inspect.json" --execute

Read actual source and Git diff, then search for occurrences. This is
multi-tool retrieval, not a trained code reviewer; no patches are proposed.

## Test diagnosis, with exact operator approval

    py -3.12 -m ntsa_x.agent_mission_intent --request "diagnose tests" --output ".local\agent\m12-tests.json"
    Get-Content ".local\agent\m12-tests.json"
    $ReviewJSON = py -3.12 -m ntsa_x.agent_mission --workspace . --mission ".local\agent\m12-tests.json"
    $Review = $ReviewJSON | ConvertFrom-Json
    $Review.possible_steps | Format-Table

**Review the tests before approving.** Tests execute local Python and are
not sandboxed. If you approve the exact plan:

    py -3.12 -m ntsa_x.agent_mission --workspace . --mission ".local\agent\m12-tests.json" --execute --enable-tests --approve-digest $Review.mission_sha256

A failing test yields mission status "failed", a nonzero exit code, and
recovery guidance instead of a fabricated success. No automatic changes.

## Approved web investigation

    py -3.12 -m ntsa_x.agent_mission_intent --request "research page https://docs.python.org/3/" --output ".local\agent\m12-web.json"
    $ReviewJSON = py -3.12 -m ntsa_x.agent_mission --workspace . --mission ".local\agent\m12-web.json" --allow-host docs.python.org
    $Review = $ReviewJSON | ConvertFrom-Json
    py -3.12 -m ntsa_x.agent_mission --workspace . --mission ".local\agent\m12-web.json" --allow-host docs.python.org --execute --enable-network --approve-digest $Review.mission_sha256

For an optional read-only browser fallback, create a separate JSON mission
with the explicit phrase "research browser https://docs.python.org/3/",
and review that *different* mission. The fallback requires BOTH
--enable-network and --enable-browser, plus its own digest. Playwright and
Chromium are optional installations. There is no unrestricted browsing.

## Remaining general-assistant roadmap

**Language / intelligence:** acquire large, legitimate, deduplicated and
varied training/evaluation corpora; profile the native recurrent/COG-FU
compute path and scale beyond the 29K-parameter smoke experiment; train
instruction following and valid tool-plan selection; measure held-out
English comprehension, coding, long-context recall, reasoning and safe
tool-use completion. A dozen documents or five unit tests cannot prove fluency.

**Agent skills:** add a policy-trained planner only behind the same
typed plan schema and external authorization gate; add code patch proposal
and rollback-safe evaluation, constrained browser click/form interaction,
authenticated provider adapters, real CI evidence and staged DevOps repair.
Require OS sandboxing, operator privileges, consent, provider policies,
negative/adversarial test cases and independent task benchmarks.

The Milestone 12 controller is a measurable foundation for those goals,
NOT evidence that NTSA-X already matches a frontier general assistant.
