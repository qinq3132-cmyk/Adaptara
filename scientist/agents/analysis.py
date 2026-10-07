from scientist.schemas import Analysis, Experiment, ExperimentResult


class MockAnalysisAgent:
    def analyze(self, experiment: Experiment, result: ExperimentResult) -> Analysis:
        difference = result.metrics["excess_loss"]
        count = result.metrics["false_adaptation_events"]
        return Analysis(
            id="A001", result_ids=(result.id,),
            observations=(f"Mean gated minus frozen squared loss: {difference:.6f}.",
                          f"Recorded unnecessary adaptation events: {count:g}."),
            statistical_comparisons={"excess_loss": difference, "false_adaptation_events": count},
            unexpected_results=(("Residual-gated updates increased behavioral loss.",) if difference > 0 else ()),
            possible_interpretations=("The result is descriptive and specific to the executed toy rule.",),
        )
