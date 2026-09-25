# molsim-agent

> A transparent scientific agent for atomistic simulation: the LLM chooses *what* to
> do, while trusted Python/ASE tools decide *how* it is executed and validated.

### Quick start

```bash
git clone https://github.com/SamDFr/molconvert.git
cd molconvert
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -e '.[dev]'
ollama pull qwen3:8b
molsim-agent -w ./simulation -m qwen3:8b
```

For a small local model use `--preset local` (the default); for simulation and trajectory
tools use `--preset scientific`; use `--preset debug` when diagnosing tool calls.

The remainder of this document explains the architecture, safety model, and extension
points. Conversion is intentionally documented as one workflow, not as the project's
definition.

`molsim-agent` is an educational, open-source scientific agent for molecular simulation.
It lets an LLM understand a research objective, assess the capabilities available in the
current environment, select safe deterministic tools, validate calculations, and explain
limitations. Structure-file conversion is one workflow among several, not the central
purpose of the project.
The orchestration loop is intentionally implemented here—there is no LangChain,
LangGraph, CrewAI, AutoGen, or other agent framework hiding it.

The current release includes structure conversion plus the foundations for trajectory
analysis, simulation specifications, calculator discovery, provenance, and temporary
scientific-code delegation. It does not pretend that simulation input decks are interchangeable.
ASE supports many additional readers and writers; they are intentionally not exposed until
each one has a project-specific safety policy and validation coverage. A request for one of
those formats is reported as “not implemented here”, rather than silently delegated to an
unvalidated ASE writer.

## Conversion semantics

The agent keeps four outcomes distinct:

1. **Exact for detected properties:** every property present in the source is supported,
   pending an independent numerical validation.
2. **Lossy conversion:** the destination cannot represent observed information; the
   report names what was discarded or changed.
3. **Semantic workflow translation:** settings express intent across different
   simulation methods and require explicit scientific decisions, not file conversion.
4. **No meaningful equivalent:** the agent says so and requests input instead of
   manufacturing a mapping.

For example, extXYZ can preserve a POSCAR's cell and PBC, while plain XYZ cannot. A
LAMMPS data file does not encode the input script's boundary command, so validation
reports its PBC status as `not_encoded` even if ASE's in-memory default happens to match.

### Why there is no second evaluator LLM

For structure conversion, the scientific evaluator is deterministic Python/ASE code,
not another language-model agent. `validate_conversion` compares the source and output
using explicit tolerances and returns a machine-readable `classification`:
`exact`, `lossy`, or `changed`, plus `information_lost_or_changed`. This is reproducible,
fast, and cannot hallucinate an atom count or cell. The orchestrating LLM only interprets
that report for the user. A separate evaluator becomes useful later for semantic tasks
(for example reviewing a proposed VASP-to-LAMMPS workflow), where the criteria are not
fully numeric and a human approval step is required.

## What is an agent?

A chat model produces text. An agent adds a runtime that repeatedly gives the model an
objective, state, and available actions; checks and executes requested actions; returns
observations; and stops only when the model answers or a runtime limit is reached.

In this project, the LLM reasons about intent and selects tools. Normal Python code owns
path safety, parsing, coordinate I/O, and numerical validation. The model never writes
atomic coordinates itself.

## Architecture

```text
User
  |
  v
Scientific Orchestrator (explicit AgentState + bounded runtime)
  |
  +--> capability assessment
  +--> LLM backend (Ollama/OpenAI-compatible/Anthropic)
  +--> trusted tool registry
          +--> simulation (ASE + optional calculators)
          +--> trajectory analysis
          +--> structure conversion (sub-workflow)
          +--> validation and provenance
  +--> ScientificCodeAgent (temporary generated tools)
  |
  v
structured observations -> next LLM call -> scientific report
```

The important modules are:

- `agent/loop.py`: the model → tool → observation loop and maximum-step guard.
- `agent/state.py` and `messages.py`: inspectable task state and backend-neutral messages.
- `llm/base.py`: the narrow interface needed to add a provider.
- `llm/ollama.py`: translation to and from Ollama's local `/api/chat` endpoint.
- `tools/registry.py`: model-visible schemas mapped to Python callables.
- `safety/policies.py`: one workspace boundary shared by every file tool.
- `formats/` and `tools/{inspect,convert,validate}.py`: deterministic scientific code.
- `skills/molecular-conversion/SKILL.md`: domain instructions loaded at startup.

`AgentState` records the objective, messages, tool executions and observations,
created/modified files, warnings, iteration count, and final answer. It is task-local;
v0.1 intentionally has no vector database or long-term memory.

## Alignment with OpenAI's agent guidance

This project follows the core recommendations in OpenAI's *A practical guide to
building agents*: start with a focused use case, keep the model/tool/instructions
contract explicit, run the model in a bounded loop, and add guardrails around every
action. The guide also recommends maximizing one agent before introducing multiple
agents, which is why this repository has one orchestrator and deterministic scientific
tools rather than a second evaluator model. See the [official OpenAI guide](https://openai.com/business/guides-and-resources/a-practical-guide-to-building-ai-agents/).

| OpenAI recommendation | Implementation in this repository | Status |
| --- | --- | --- |
| Use an LLM to manage workflow execution | `Agent.run()` repeatedly asks the backend to choose a tool or finish | Implemented |
| Give the model well-defined tools | `ToolSpec` contains name, description, JSON schema, callable, and safety metadata | Implemented |
| Keep instructions explicit and reusable | Base prompts plus `skills/*/SKILL.md`, with compact/full profiles | Implemented |
| Use a bounded run with clear exit conditions | Maximum iterations, completion checks, repeated-failure stop, and validation gates | Implemented |
| Start with a single agent | One orchestrator; ASE and Python validators are not hidden agents | Implemented |
| Prefer capable models, then optimize latency/cost | Ollama and API backends are interchangeable; compact mode reduces context for small models | Partially implemented |
| Establish an evaluation baseline | 57 deterministic and mock-LLM tests cover tools and orchestration | Partially implemented: no benchmark dashboard yet |
| Layer guardrails and tool safeguards | Workspace sandbox, typed arguments, no shell, overwrite protection, deterministic validation | Implemented for v0.1 scope |
| Human intervention for risky or failed work | Errors and unsupported semantics are returned to the user; no automatic destructive actions | Implemented for file scope; formal approval UI is future work |
| Add multiple agents only when complexity demands it | No specialist/evaluator LLM is used for numeric structure checks | Deliberate design choice |

This is alignment with engineering principles, not a claim of production certification.
The project does not currently provide generic prompt-injection classification,
moderation, identity/access management, a benchmark suite, or a human approval service.
Those controls become necessary when the agent is exposed to untrusted users, remote
systems, or high-impact workflow changes.

## Scientific-agent foundation (incremental)

The repository now has explicit domain boundaries for the next research workflows:

```text
Scientific Orchestrator (Agent runtime)
  | capability assessment
  +--> trusted deterministic tools (ASE/analysis)
  +--> ScientificCodeAgent boundary (temporary generated_tools/)
  +--> SimulationAgent boundary (validated specifications)
                         |
                    ExperimentRecord
```

`CapabilityAssessment` distinguishes understanding a request from being able to execute
it. A request can be `available`, `needs_implementation`, `needs_dependency`,
`needs_external_data`, `needs_compute`, `needs_user_input`, or `unsupported`. Unknown
observables are never silently replaced by a different calculation. `SubAgentResult` and
`SubAgent` provide an isolated delegation contract; the current release keeps generated
code opt-in and never edits trusted `src/` automatically.

`ToolSpec` also exposes category, risk, requirements, compute cost, and determinism so a
future orchestrator can preflight tools without inspecting Python callables. `MDSpec`,
`SinglePointSpec`, `OptimizationSpec`, and `ExperimentRecord` are backend-neutral models
for validated simulations and provenance. MACE and UMA are optional: no large model is
downloaded by installation, and `PotentialRegistry` reports missing dependencies clearly.

The initial analysis helpers (`trajectory_summary`, pair-distance statistics, and MSD)
are deterministic functions. MD execution and fixed-configuration model comparison are
available in the full profile. Dynamic code generation remains intentionally staged.

### Scientific workflows available in this milestone

The full profile exposes trusted ASE tools for `single_point`,
`geometry_optimization`, and bounded `run_md` (NVE or Langevin NVT). Each run writes a
run-specific directory under `runs/`, including results, trajectories where relevant,
logs, and an `experiment.json` provenance record. The built-in EMT calculator is useful
for tests and demonstrations; MACE and UMA are discovered only when their optional
dependencies are installed.

For a request such as “prepare VASP inputs for AIMD at 300 K for 1 ps”, the agent can
write a conservative `INCAR` and Gamma-point `KPOINTS` template. It uses explicit
protocol defaults (`IBRION=0`, `NSW`, `POTIM`, `TEBEG/TEEND`, fixed-cell `ISIF=2`) and
places review comments in `INCAR`. It never fabricates or writes `POTCAR`, `ENCUT`, a
functional, spin settings, or electronic smearing. The final report lists every default
that must be reviewed before production use, and existing inputs are protected unless
overwrite is explicitly authorized.

The same policy applies to LAMMPS MD preparation. The agent can create a deterministic
LAMMPS data file and an `in.molsim` template with standard protocol defaults. It leaves
explicit `__REQUIRED__` placeholders for `pair_style` and `pair_coeff` rather than
inventing a force field or ML potential. The template must be reviewed before execution.

For GROMACS MD preparation, the agent can write an ASE-generated `.gro` geometry, a
generic `md.mdp`, and an explicitly incomplete `topol.top.template`. A POSCAR does not
contain force-field parameters, atom types, or charges, so these are reported as missing
and are never fabricated. Review the files before running `gmx grompp`.

Examples:

```text
Run a single-point EMT calculation on h2.xyz.
Run 100 steps of NVE MD on h2.xyz at 0.5 fs and save the trajectory.
Summarize the trajectory in runs/md-*/trajectory.traj.
```

For a complete orchestrator exercise, see
[`examples/scientific_orchestrator_prompt.txt`](examples/scientific_orchestrator_prompt.txt).
It asks the agent to understand a scientific objective, assess capabilities, choose a
safe workflow, generate only justified files, validate them, and report missing inputs.

Long prompts can be entered without terminal copy/paste using the optional desktop chat:

```bash
molsim-agent --dialog --provider groq --model openai/gpt-oss-20b --profile full
```

The chat keeps a conversation history, displays tool calls and observations, and lets you
send several requests in one session. It uses Tkinter from the standard Python
distribution. On a headless machine it falls back to the terminal prompt. You can also
run a saved prompt directly with `--prompt-file examples/scientific_orchestrator_prompt.txt`.

The compact profile intentionally remains conversion-focused for small local models.
Use `--profile full` for scientific simulation and analysis requests so the orchestrator
can see the broader tool registry.

In `--intent-mode llm`, the scientific planner first receives the complete user request,
the files visible in the workspace, and the registered tool schemas. It returns an
ordered structured plan, which the runtime validates before executing. This allows a
single request to combine inspection, preparation, conversion, validation, and reporting
without adding a new hard-coded branch for every wording or user workflow. If the planner
cannot return a valid plan, the explicit tool-calling loop remains available as a safe
fallback.

### Execution profiles

Profiles change the context and constraints presented to the model, not the ASE
conversion or validation algorithms:

- `full` is the default. It loads the complete skill, exposes all tools, retains the full
  conversation, and lets a capable model choose among them.
- `compact` targets small local models. For explicit conversion objectives it exposes
  one next tool at a time, projects state into a short prompt, omits coordinate arrays
  from model-facing inspection, and constrains explicit source/destination/format values.
  A mismatched call is rejected rather than silently corrected. ASE and validation still
  use the complete coordinates locally.
- `auto` currently selects `compact` for Ollama and `full` for other backends. Explicit
  `--profile` always wins.

Compact mode deliberately trades conversational flexibility and free-form planning for
reliability on resource-constrained hardware. Full mode remains available unchanged for
larger local models and remote backends. Both profiles use the same agent loop, tool
registry, workspace sandbox, conversion functions, and validation guard.

Compact mode also normalizes common everyday wording. If a supported target is named but
the user omits an output filename, it derives a safe name such as `POSCAR.data` for a
LAMMPS data file or `POSCAR.xyz` for XYZ. Scientific choices—units, potentials, boundary
semantics, and workflow mappings—are never inferred.

For ordinary conversation such as `hello` or `bonjour`, the runtime sends no tool schemas
to the provider. This keeps providers/models that do not accept function tools usable for
simple chat while preserving the full tool set for file and structure requests.

### Intent normalization modes

The default `--intent-mode deterministic` uses local rules and adds no model request. It
is the recommended mode for small or slow local models. `--intent-mode llm` performs one
additional, tightly constrained LLM call before the agent loop and asks for a JSON intent
(`source`, `destination`, `target_format`, `validate`). The result is accepted only when
it contains a supported format and concrete source; otherwise the normal deterministic
interpretation remains in control. This optional call improves tolerance of natural
phrasing, but costs latency and does not authorize scientific mappings or bypass safety.

## Agent loop

The readable implementation is in
[`src/molsim_agent/agent/loop.py`](src/molsim_agent/agent/loop.py). In simplified form:

```python
for iteration in range(max_iterations):
    response = backend.chat(state.messages, registry.schemas())
    state.messages.append(response)
    if response.tool_calls:
        for call in response.tool_calls:
            observation = registry.execute(call.name, call.arguments)
            state.messages.append(observation)
    else:
        return response.content
```

The real loop turns invalid or failed calls into observations so the model can recover,
records every execution, emits debug events, and terminates safely at the configured
limit. A final answer is the model's explicit completion signal.

## Installation

Python 3.11 or newer is required.

```bash
git clone https://github.com/SamDFr/molconvert.git molsim-agent
cd molsim-agent
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
```

ASE is the only runtime Python dependency. Pytest is optional development tooling.

## Simple command-line usage

Most users only need a workspace and a model. The default `local` preset selects the
compact deterministic workflow:

```bash
molsim-agent -w ./simulation -m qwen3:8b
```

This command uses **Ollama locally by default** (`--provider ollama`). No API key is
required. The startup banner prints the active provider, model, workspace, and preset so
there is no ambiguity. Use `--provider groq`, `--provider openai`, `--provider mistral`,
or `--provider anthropic` when you intentionally want a hosted backend.

Then type requests at the `>` prompt, or provide one directly:

```bash
molsim-agent -w ./simulation -m qwen3:8b \
  "Convert POSCAR to extended XYZ and validate it"
```

Use a preset instead of repeating several flags:

```bash
# Full scientific tool registry and LLM intent normalization
molsim-agent -w ./simulation -m qwen3:8b --preset scientific

# Detailed model/tool diagnostics
molsim-agent -w ./simulation -m qwen3:8b --preset debug
```

The equivalent environment variables are `MOLSIM_AGENT_MODEL`,
`MOLSIM_AGENT_PROVIDER` (default `ollama`), and `MOLSIM_AGENT_WORKSPACE`. Advanced flags such as
`--profile`, `--intent-mode`, `--progress-level`, `--timeout`, and `--verbose` remain
available when you need precise control.

## Ollama setup

Install [Ollama](https://ollama.com/), start it, and obtain a model that reliably emits
native tool calls:

```bash
ollama serve
ollama pull qwen3:8b
molsim-agent --workspace ./simulation --model qwen3:8b
```

The model name is never hardcoded. Pass `--model` or set `MOLSIM_AGENT_MODEL`.
Instruction-following models trained for function/tool calling work best. A small model
may select nonexistent tools, omit required arguments, or stop too early; the registry
and scientific validators remain authoritative even when model planning is imperfect.

The compact intent router is intentionally English-only, not a general multilingual
parser. It can select an explicitly named supported structure file (POSCAR/CONTCAR,
`.xyz`, `.extxyz`, `.traj`, `.data`, or `.cif`) and can use the unique supported structure in a
workspace when the request says “the file”. If multiple candidates exist, provide the
filename explicitly. Other languages or unusual phrasing may require the full profile.

## Hosted API providers

The same explicit loop can use hosted providers without changing the tools. Set the
provider-specific key in the environment and select `--provider`:

```bash
export OPENAI_API_KEY="..."
molsim-agent --provider openai --model gpt-5-mini --workspace ./simulation

export MISTRAL_API_KEY="..."
molsim-agent --provider mistral --model mistral-small-latest --workspace ./simulation

export GROQ_API_KEY="..."
molsim-agent --provider groq --model YOUR_GROQ_MODEL_ID --workspace ./simulation

export ANTHROPIC_API_KEY="..."
molsim-agent --provider anthropic --model claude-3-5-haiku-latest --workspace ./simulation
```

OpenAI, Mistral, and Groq use the OpenAI-compatible Chat Completions adapter. Claude uses
the Anthropic Messages adapter. `--api-key` and `--base-url` are available for testing,
but environment variables are safer because secrets do not appear in shell history or
process arguments. For Groq, the adapter uses `https://api.groq.com/openai/v1` and reads
`GROQ_API_KEY`. Groq supports local function calling, so the agent still executes the
filesystem and ASE tools on your computer rather than delegating them to the provider.
API providers are optional: Ollama remains the default and no cloud dependency or key is
required for local use.

### API-key safety

Never commit a key, put it in a prompt, or paste it into source code. Prefer a short-lived
shell environment variable, a password manager, or the macOS Keychain. If a key is ever
printed, committed, or shared, revoke it immediately in the provider console and create a
replacement. The application sends the key only as an HTTP authorization header; it is
not included in `AgentState`, tool observations, progress messages, or debug payloads.

### Groq on macOS with Keychain

For local macOS use, store the key in **Keychain Access** instead of putting it in the
repository or `.zshrc`:

1. Open **Keychain Access** with `Cmd+Space`.
2. Create a **New Generic Password** in the `login` keychain.
3. Use `molsim-agent-groq` as the item name, your macOS account as the account, and paste
   the Groq key into the password field.
4. Load it only for the current terminal session:

```bash
export GROQ_API_KEY="$(security find-generic-password \
  -a "$USER" -s "molsim-agent-groq" -w)"
test -n "$GROQ_API_KEY" && echo "Groq key loaded"
```

Run the agent with a model ID shown in [Groq's supported-model list](https://console.groq.com/docs/models):

```bash
molsim-agent \
  --provider groq \
  --model YOUR_GROQ_MODEL_ID \
  --workspace ./simulation \
  --profile compact
```

To change the model, keep the same key and replace only the value after `--model`, for
example `--model llama-3.3-70b-versatile` or `--model openai/gpt-oss-20b`. You can also
set `MOLSIM_AGENT_MODEL` instead of passing `--model`. Groq model IDs and availability
can change, so use its model list rather than assuming an old ID remains active. Clear
the session variable when finished with `unset GROQ_API_KEY`.

## First example

Place a `POSCAR` in a workspace, launch the CLI, and ask:

```text
Convert POSCAR to extended XYZ as structure.xyz.
Then verify that atom count, species, positions, cell and PBC were preserved.
```

The expected agent-selected sequence is:

```text
detect_file_format -> inspect_structure -> convert_structure -> validate_conversion -> final
```

This completion requirement applies to both `compact` and `full`: a planning paragraph
alone is not accepted as the final answer for a conversion request.

It is not hard-wired to that sentence. The sequence arises from model tool calls plus the
loaded molecular-conversion skill. For a one-shot command:

```bash
molsim-agent --workspace . --model qwen3:8b \
  "Convert POSCAR to extended XYZ as structure.xyz, then validate it."
```

Use `--verbose` to see model requests, normalized tool decisions, observations, and step
numbers. Debug output shows messages and actions, not hidden chain-of-thought.

Each `Agent loop iteration N` is one complete loop turn: the runtime sends the current
state to the LLM, receives a tool call or answer, and (for a tool call) executes it and
appends the observation before starting the next iteration. It is not an ASE operation
or a shell command count.

In normal CLI mode, internal runtime guards are hidden. While waiting for an LLM response,
the CLI shows a small animated `Thinking...` indicator; this is only a waiting indicator,
not the model's private reasoning. The label follows the current runtime action (for
example `Checking the file format...` or `Validating the conversion...`) and its timer is
cumulative for the current task. Use `--progress-level detailed` to show factual text
that the LLM itself returned alongside a tool call. `--verbose` exposes the iteration
and normalized request/response diagnostics. In normal mode, tool observations are shown
as concise summaries while `--verbose` prints the complete structured result. Detailed
`Progress:` lines are the model's own factual recap of the observations available at that
point.

Compact inspection deliberately sends species counts and structural capabilities to the
LLM, rather than the full coordinate and symbol arrays. The deterministic tool and
validator still read the complete ASE structure; this is only a context-size reduction.

Use `--dry-run` to let the agent inspect and plan a conversion without writing the output
file. In interactive mode, `help` lists available CLI commands and `status` prints the
workspace, model, and profile configuration.

Use `--progress` (or `--progress-level brief`) to display short progress sentences
actually generated by the model before its tool calls. For richer factual updates, use
`--progress-level detailed`; the model is asked for one or two concise scientific
sentences (maximum 45 words) using observed file format, atom count, species, cell, or
validation status. The runtime does not synthesize these messages;
models that return empty content will produce no progress line.

When progress is enabled, the runtime requires a model-generated announcement before a
native tool call is executed. If the model returns only a tool call with no text, it is
asked again to say what it found and what it will do next. This may add an LLM turn, but
it prevents the runtime from inventing a canned announcement. If a small model still
returns an empty announcement on the retry, the original tool call is allowed so the
workflow cannot deadlock.

On a small Intel Mac, use:

```bash
molsim-agent --workspace . --model granite3.3:2b \
  --profile compact --no-think --timeout 120
```

The same runtime is available from Python:

```python
from molsim_agent import Agent

agent = Agent(model="qwen3:8b", workspace="./simulation")
state = agent.run("Convert POSCAR to XYZ and validate the result.")
print(state.final_answer)
print(state.tool_executions)
```

Tests inject a scripted `LLMBackend`, so neither Ollama nor a probabilistic model is
needed to verify orchestration.

## How tool calling works

A `ToolSpec` contains a name, description, JSON-compatible parameter schema, callable,
and safety metadata. `ToolRegistry.schemas()` exposes only the schema to the model. When
the model requests, for example:

```json
{
  "name": "convert_structure",
  "arguments": {
    "source": "POSCAR",
    "destination": "structure.xyz",
    "target_format": "extxyz"
  }
}
```

the registry rejects missing, unknown, wrongly typed, or invalid enum arguments before
calling Python. The returned dictionary is serialized as a `tool` message and becomes
the model's next observation. Conversion is deterministic ASE code; tool descriptions
do not grant the model direct filesystem or shell access.

## Skills

A skill is a folder whose `SKILL.md` supplies focused instructions. At startup,
`agent/planner.py` loads `skills/molecular-conversion/SKILL.md` into the system context.
It defines supported work, the no-guessing policy, and the preferred inspect → convert →
validate flow. Pass `skill_paths=[...]` to `Agent` to experiment with other instruction
files. Skills guide planning; they do not bypass tool or safety enforcement.

The repository also includes an explanatory slide deck:
[`docs/molsim-agent-overview.pdf`](docs/molsim-agent-overview.pdf) and its editable
source [`docs/molsim-agent-overview.tex`](docs/molsim-agent-overview.tex). Update the
source when adding a major workflow, tool category, safety rule, or capability status,
then rebuild it with `pdflatex`.

## Safety model

- Every tool resolves paths against one canonical workspace; `..`, absolute paths, and
  symlinks cannot escape it.
- Existing outputs are never overwritten by default. If the requested destination already
  exists, the runtime chooses a numbered sibling (`file_1.ext`, then `file_2.ext`, etc.)
  and reports both the requested and actual destination. `overwrite=true` is reserved for
  an explicit user request.
- There is no arbitrary shell tool.
- ASE, not the LLM, parses and writes atomic data.
- Every conversion should be re-read and validated rather than trusted after writing.
- Validation compares atom count, ordered symbols, positions, cell, PBC, velocities,
  constraints, charges, forces, and energy where present.
- Plain XYZ is treated as lossy for cell/PBC. Format risks and actual observed losses are
  reported separately.
- Missing potential parameters, units, type mappings, or scientific semantics require
  user input. They are never invented.
- Structure conversion is distinct from semantic workflow translation. VASP settings
  such as `ENCUT`, `GGA`, and `ISMEAR` do not directly map to classical LAMMPS settings.

This sandbox limits the agent's tools, not Ollama itself or unrelated processes on the
machine. Run untrusted models and files with the usual operating-system isolation.

## Adding a conversion format

1. Add a conservative signature and normalized name in `formats/detection.py`.
2. Confirm ASE's explicit reader/writer format and any required writer options in
   `formats/structures.py`.
3. Document unsupported properties in `tools/convert.py`; never infer silent mappings.
4. Add a small fixture and round-trip tests that assert both preservation and expected
   loss.
5. Update the skill and format list only after the deterministic tests pass.

## Adding a tool

Write a normal typed function returning a JSON-compatible dictionary. Wrap it in a
`ToolSpec`, use a closed JSON schema (`additionalProperties: false`), bind the workspace
instead of accepting unrestricted paths, and register it in `Agent._default_registry()`.
Test the function independently, then add a mock-backend loop test proving the
observation returns to the model.

## Adding another LLM backend

Subclass `LLMBackend` and implement:

```python
def chat(messages: Sequence[Message], tools: Sequence[dict]) -> LLMResponse:
    ...
```

The adapter owns provider-specific message and tool-call syntax. It must normalize calls
to `ToolCall`; the runtime, registry, state, and scientific tools remain unchanged.

## Testing

```bash
pytest -q
```

For a small live test on a resource-constrained computer, use the compact one-tool
smoke test. It creates and removes its own temporary workspace:

```bash
python examples/ollama_smoke_test.py --model granite3.3:2b
```

A pass requires a real native Ollama tool call, a Python tool observation, and a final
model answer. It is intentionally smaller than the full molecular-agent prompt.

To run a real temporary POSCAR → extXYZ conversion through Ollama and print the output:

```bash
python examples/live_conversion_demo.py --model granite3.3:2b --timeout 120
```

To keep the generated files for inspection, provide an empty directory:

```bash
python examples/live_conversion_demo.py --model granite3.3:2b \
  --keep-workspace ./demo-output
```

Deterministic tests cover the registry/loop, workflow dispatch, Ollama request
normalization, filesystem sandbox, overwrite policy, supported conversion paths, expected
plain-XYZ loss, simulation specifications/tools, and the complete five-step milestone
with a mock LLM. No test requires an Ollama server.

## Roadmap

Next priorities: dynamic ScientificCodeAgent generation with stronger isolation; RDF and
other trajectory observables; MACE/UMA execution integrations; comparative trajectory
experiments; VASP INCAR/XDATCAR/OUTCAR; LAMMPS input assistance; GROMACS, CP2K, and
Quantum ESPRESSO; SLURM; documentation RAG; MCP exposure; and specialist subagents.
Semantic translations will use explicit equivalence taxonomies and user-confirmed
assumptions rather than pretending to be file conversions.

## License

MIT. See [LICENSE](LICENSE).
