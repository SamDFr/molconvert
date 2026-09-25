import json

import pytest

from molsim_agent.science import (
    CapabilityAssessment,
    CapabilityStatus,
    ExperimentRecord,
    MDSpec,
    ScientificSpecError,
)


def test_capability_assessment_is_structured_and_serializable() -> None:
    assessment = CapabilityAssessment(
        status="needs_implementation",
        task="orientational autocorrelation",
        missing_capability="orientational_autocorrelation",
        proposed_solution="Implement with NumPy and MDAnalysis.",
        delegate_to="scientific_code_agent",
    )

    assert assessment.status is CapabilityStatus.NEEDS_IMPLEMENTATION
    assert assessment.to_dict()["delegate_to"] == "scientific_code_agent"


def test_capability_assessment_rejects_unknown_status() -> None:
    with pytest.raises(ValueError, match="Unknown capability status"):
        CapabilityAssessment(status="maybe", task="test")


def test_md_spec_validates_nvt_and_normalizes_ensemble() -> None:
    spec = MDSpec(
        structure="input.xyz",
        calculator="emt",
        ensemble="nvt",
        temperature_K=300,
        timestep_fs=0.5,
        steps=100,
        seed=7,
    )

    assert spec.ensemble == "NVT"
    assert spec.to_dict()["steps"] == 100


def test_md_spec_rejects_incomplete_or_unsafe_values() -> None:
    with pytest.raises(ScientificSpecError, match="NVT requires"):
        MDSpec(
            structure="input.xyz",
            calculator="emt",
            ensemble="NVT",
            timestep_fs=0.5,
            steps=1,
        )
    with pytest.raises(ScientificSpecError, match="positive"):
        MDSpec(
            structure="input.xyz",
            calculator="emt",
            ensemble="NVE",
            timestep_fs=0,
            steps=1,
        )


def test_experiment_record_hashes_input_and_serializes(tmp_path) -> None:
    source = tmp_path / "structure.xyz"
    source.write_text("2\ncomment\nH 0 0 0\nH 0 0 1\n")
    record = ExperimentRecord.for_input(
        id="run-1",
        kind="single_point",
        status="success",
        input_structure=source,
        parameters={"calculator": "emt"},
        outputs=["results.json"],
    )

    assert record.input_hash == ExperimentRecord.hash_file(source)
    payload = json.loads(record.to_json())
    assert payload["id"] == "run-1"
    assert payload["outputs"] == ["results.json"]
