# Milestone 11 — NTSA-X local agent workbench

**Research status:** This release installs actual tool adapters and an operator
approval workflow. It does NOT make the tiny native COG-FU checkpoint fluent
or equivalent to a frontier assistant.

## Capability inventory

| Skill | Current support | Restrictions |
| --- | --- | --- |
| Source code | list, read, search, guarded write | exact workspace; bounded text; write approval |
| Version control | Git status and diff | read-only |
| Testing | fixed Python unittest discovery | explicit plan approval and test opt-in |
| DevOps | known deployment configuration inventory | no deployment or credentials |
| Web research | bounded HTTPS text retrieval | exact host allowlist, public DNS pin, network opt-in |
| Browser | Playwright page text snapshot | optional Chromium, host allowlist, browser opt-in |
| English commands | limited explicit phrase router | rule-based; rejects unknown requests |

Not yet supplied: autonomous general reasoning, multi-step repair loops,
interactive browser clicking and forms, actual cloud deployment, email and
calendar integration, image/audio understanding, spreadsheet and document
authoring, and robust English language modeling.

## Safety and authorization

The workbench takes a literal JSON plan with 1-20 steps and validates *all*
steps before effects. A normal run is a no-action preview that prints a
SHA-256 digest bound to the whole plan, workspace, and exact allowed hosts.

For any write, test, web or browser effect, execution requires BOTH:
- the exact preview digest passed by --approve-digest;
- an enabling flag: --enable-write, --enable-tests, --enable-network,
  or --enable-browser.

The model does not approve its own actions. Unexpected tools, undeclared
arguments, traversal, common sensitive file paths and oversized inputs are
rejected. Existing files require a matching expected_sha256 to be replaced.
The workbench stops executing further steps if a tool fails and emits a
machine-readable receipt. Never grant this tool an administrator session.

This is NOT an OS sandbox. Reviewed Python tests can run project code, and
a malicious Chromium site may exploit browser resources. Use a separate OS
user and isolated network for untrusted projects/sites. Remote text is
untrusted data, never an executable workplan. No deployment/write-to-cloud,
payments or credential operations exist in this milestone.

## Setup on Windows PowerShell

    Set-Location -LiteralPath (Join-Path $HOME "KindredLabs\NTSA-X-GitHub")
    git pull --ff-only origin main
    py -3.12 -m unittest discover -s tests -p test_agent_workbench.py -v
    New-Item -ItemType Directory -Force -Path ".local\agent" | Out-Null

### Make a read-only Git status plan using basic English

    py -3.12 -m ntsa_x.agent_intent --request "git status" --output ".local\agent\git-status.json"
    Get-Content ".local\agent\git-status.json"
    py -3.12 -m ntsa_x.agent_workbench --workspace . --plan ".local\agent\git-status.json"
    py -3.12 -m ntsa_x.agent_workbench --workspace . --plan ".local\agent\git-status.json" --execute

Other explicit phrases: list files; read file README.md; search files for
COG-FU; inspect devops; run tests; git diff README.md; fetch page
https://docs.python.org/3/; inspect website https://docs.python.org/3/.

### Run unit tests with deliberate approval

    py -3.12 -m ntsa_x.agent_intent --request "run tests" --output ".local\agent\run-tests.json"
    $ReviewJson = py -3.12 -m ntsa_x.agent_workbench --workspace . --plan ".local\agent\run-tests.json"
    $Review = $ReviewJson | ConvertFrom-Json
    $Review | Format-List
    py -3.12 -m ntsa_x.agent_workbench --workspace . --plan ".local\agent\run-tests.json" --execute --enable-tests --approve-digest $Review.plan_sha256

The tests can execute code in the local project. Inspect them before approval.

### Make a coding change with a reviewed plan

Create .local/agent/new-file.json with contents:

    {
      "version": 1,
      "steps": [
        {"tool": "workspace.write", "args": {
          "path": "hello_agent.py",
          "content": "print('Hello, NTSA-X agent!')\n"
        }}
      ]
    }

    $ReviewJson = py -3.12 -m ntsa_x.agent_workbench --workspace . --plan ".local\agent\new-file.json"
    $Review = $ReviewJson | ConvertFrom-Json
    py -3.12 -m ntsa_x.agent_workbench --workspace . --plan ".local\agent\new-file.json" --execute --enable-write --approve-digest $Review.plan_sha256

No file is overwritten without its correct original SHA-256.

### Fetch a public documentation page

    py -3.12 -m ntsa_x.agent_intent --request "fetch page https://docs.python.org/3/" --output ".local\agent\python-docs.json"
    $ReviewJson = py -3.12 -m ntsa_x.agent_workbench --workspace . --plan ".local\agent\python-docs.json" --allow-host docs.python.org
    $Review = $ReviewJson | ConvertFrom-Json
    py -3.12 -m ntsa_x.agent_workbench --workspace . --plan ".local\agent\python-docs.json" --allow-host docs.python.org --execute --enable-network --approve-digest $Review.plan_sha256

Optional browser.snapshot additionally requires an operator-installed Playwright
Python package and Chromium, explicit hostname and --enable-browser.
Browser snapshots do not click, type, login, or download.

## How we approach English fluency and broad agent capability

The present 29,023-parameter trained smoke model is not English-fluent. Add
licensed, diverse and deduplicated corpora; evaluate source-disjoint holdouts,
then scale native recurrence and COG-FU with profiling and suitable compute.
The twelve starter documents are training examples, NOT sufficient training
for a conversational assistant.

Next, collect reviewed instruction-following and structured tool-use data,
train a model to emit this typed plan protocol, and compare it on separate
coding, web, browser, math, English and DevOps tasks against a recurrent-only
baseline and a strong reference. Report valid plan rate, task completion,
language coherence, refusal calibration, unsafe attempted effects, latency
and cost; never equate having a tool with knowing how to use it.

Only after demonstrated model competence should we consider continuous
agent planning and retries, complex browser interactions, signed plugin
manifests, deployment workflows, voice and multimodal modalities.
