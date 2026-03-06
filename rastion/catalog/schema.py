from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


RoutingFamily = Literal["routing"]
ProblemVariant = Literal["tsp", "cvrp", "vrptw", "pickup-delivery", "other"]
MethodClass = Literal["heuristic", "metaheuristic", "exact", "hybrid"]
ListingTier = Literal["official", "experimental"]
AdapterKind = Literal["local_plugin", "entrypoint"]


class InstallationSpec(BaseModel):
    command: str
    package: str | None = None
    extras: list[str] = Field(default_factory=list)
    python: str | None = None
    optional_dependencies: list[str] = Field(default_factory=list)


class AdapterSpec(BaseModel):
    kind: AdapterKind
    solver_name: str
    path: str
    entrypoint: str | None = None


class CapabilitySpec(BaseModel):
    streaming: bool
    deterministic_mode: bool
    warm_start: bool
    local_search: bool
    multi_route_ready: bool
    notes: str | None = None


class LimitSpec(BaseModel):
    tested_max_nodes: int = Field(gt=0)
    unsupported_constraints: list[str] = Field(default_factory=list)
    depot_assumptions: str
    notes: str | None = None


class ArtifactPaths(BaseModel):
    benchmark_json: str
    arena_json: str
    suite_results_dir: str
    detail_markdown: str


class SolverCard(BaseModel):
    id: str
    name: str
    routing_family: RoutingFamily
    problem_variants: list[ProblemVariant] = Field(min_length=1)
    summary: str
    description: str
    repository: HttpUrl
    homepage: HttpUrl
    license: str
    authors: list[str] = Field(min_length=1)
    maintainers: list[str] = Field(min_length=1)
    installation: InstallationSpec
    adapter: AdapterSpec
    capabilities: CapabilitySpec
    limits: LimitSpec
    hardware: list[str] = Field(default_factory=list)
    determinism: str
    method_class: MethodClass
    version: str
    citations: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
    known_failure_modes: list[str] = Field(default_factory=list)
    artifact_paths: ArtifactPaths
    tags: list[str] = Field(default_factory=list)
    listing_tier: ListingTier = "official"


class EvalInstanceSpec(BaseModel):
    id: str
    label: str
    path: str
    size: str


class BudgetPolicy(BaseModel):
    iters: int | None = Field(default=None, ge=1)
    time_budget_ms: int | None = Field(default=None, ge=1)
    repeat: int = Field(default=1, ge=1)


class SeedPolicy(BaseModel):
    seeds: list[int] = Field(min_length=1)


class MetricPolicy(BaseModel):
    primary_metric: str
    objective: Literal["minimize", "maximize"]
    runtime_unit: Literal["ms", "s"] = "ms"


class EvalSuiteSpec(BaseModel):
    id: str
    title: str
    description: str
    problem_variant: ProblemVariant
    instances: list[EvalInstanceSpec] = Field(min_length=1)
    budget_policy: BudgetPolicy
    seed_policy: SeedPolicy
    metric_policy: MetricPolicy
    result_schema_version: str
    optional_dependencies: list[str] = Field(default_factory=list)
    listing_tiers: list[ListingTier] = Field(default_factory=lambda: ["official", "experimental"])


class EvalRunRecord(BaseModel):
    suite_id: str
    solver_id: str
    solver_version: str
    problem_variant: ProblemVariant
    instance_id: str
    score: float | None = None
    runtime: float | None = None
    status: str
    seed_policy: dict[str, object]
    budget_policy: dict[str, object]
    generated_at: str
    artifact_path: str
    environment: dict[str, object]
