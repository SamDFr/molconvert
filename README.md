# molsim-agent

`molsim-agent` is an educational, open-source agent for molecular-simulation file
conversion. It lets a local LLM decide which safe, deterministic tools to use, observes
their structured results, validates scientific preservation, and reports limitations.
The orchestration loop is intentionally implemented here—there is no LangChain,
LangGraph, CrewAI, AutoGen, or other agent framework hiding it.

Version 0.1 focuses on structures in VASP POSCAR/CONTCAR, XYZ/extXYZ, ASE `.traj`, and
LAMMPS data formats. It does not pretend that simulation input decks are interchangeable.

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
Agent Runtime (loop.py + explicit AgentState)
  |
  v
LLMBackend ----------------------------------+
  |                                          |
  +--> OllamaBackend                         |
  +--> test MockBackend                      |
  |                                          |
  | tool call                                | final answer
  v                                          |
Tool Registry                                |
  |                                          |
  +--> constrained filesystem                |
  +--> format detection / inspection         |
  +--> deterministic ASE conversion          |
  +--> independent validation                |
  |                                          |
  v                                          |
structured Observation ----------------> next LLM call
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

The compact intent router is not a general multilingual parser. It currently recognizes
common English wording and explicit French structure requests such as `Trouve le POSCAR
et convertie le en POSCAR.xyz`. Other languages or unusual phrasing may require an
explicit filename and format, or the full profile.

## Hosted API providers

The same explicit loop can use hosted providers without changing the tools. Set the
provider-specific key in the environment and select `--provider`:

```bash
export OPENAI_API_KEY="..."
molsim-agent --provider openai --model gpt-5-mini --workspace ./simulation

export MISTRAL_API_KEY="..."
molsim-agent --provider mistral --model mistral-small-latest --workspace ./simulation

export ANTHROPIC_API_KEY="..."
molsim-agent --provider anthropic --model claude-3-5-haiku-latest --workspace ./simulation
```

OpenAI and Mistral use the OpenAI-compatible Chat Completions adapter. Claude uses the
Anthropic Messages adapter. `--api-key` and `--base-url` are available for testing, but
environment variables are safer. API providers are optional: Ollama remains the default
and no cloud dependency or key is required for local use.

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

It is not hard-wired to that sentence. The sequence arises from model tool calls plus the
loaded molecular-conversion skill. For a one-shot command:

```bash
molsim-agent --workspace . --model qwen3:8b \
  "Convert POSCAR to extended XYZ as structure.xyz, then validate it."
```

Use `--verbose` to see model requests, normalized tool decisions, observations, and step
numbers. Debug output shows messages and actions, not hidden chain-of-thought.

Use `--progress` (or `--progress-level brief`) to display short progress sentences
actually generated by the model before its tool calls. For richer factual updates, use
`--progress-level detailed`; the model may mention observed file format, atom count,
species, cell, and validation status. The runtime does not synthesize these messages;
models that return empty content will produce no progress line.

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

## Safety model

- Every tool resolves paths against one canonical workspace; `..`, absolute paths, and
  symlinks cannot escape it.
- Existing outputs are not overwritten unless `overwrite=true`, which the skill permits
  only after an explicit user request.
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

Deterministic tests cover the registry/loop, Ollama request normalization, filesystem
sandbox, overwrite policy, supported conversion paths, expected plain-XYZ loss, and the
complete five-step milestone with a mock LLM. No test requires an Ollama server.

## Roadmap

After the architecture is stable: VASP INCAR/XDATCAR/OUTCAR; LAMMPS input assistance;
GROMACS, CP2K, and Quantum ESPRESSO; MACE/MLIP setup; workflow validation; SLURM;
trajectory analysis; documentation RAG; OpenAI-compatible backends; MCP exposure; and
specialist VASP/LAMMPS subagents. Semantic translations will use explicit equivalence
taxonomies and user-confirmed assumptions rather than pretending to be file conversions.

## License

MIT. See [LICENSE](LICENSE).
