"""Load and validate config.yaml and products.yaml (fail fast, readable errors)."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

ANCHORS = (
    "center", "top", "bottom", "left", "right",
    "top_left", "top_right", "bottom_left", "bottom_right",
)
Anchor = Literal[
    "center", "top", "bottom", "left", "right",
    "top_left", "top_right", "bottom_left", "bottom_right",
]
_PRODUCT_KEY_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")


class ConfigError(Exception):
    """Raised when config.yaml or products.yaml is missing or invalid."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------- config.yaml
class RunCfg(_Strict):
    designs_per_batch: int = Field(10, ge=1)
    briefs_per_niche: int = Field(5, ge=1, le=20)
    max_qa_retries: int = Field(3, ge=0, le=20)
    max_designs_per_run: int = Field(30, ge=1)
    min_free_disk_gb: float = Field(3.0, ge=0)


class RuntimeCfg(_Strict):
    gpu_layout: Literal["auto", "dual", "single"] = "auto"
    work_dir: str = "/kaggle/working"


class LLMCfg(_Strict):
    backend: Literal["ollama"] = "ollama"
    host: str = "http://127.0.0.1:11434"
    text_model: str = "qwen2.5:14b"
    fallback_model: str | None = "qwen2.5:7b"
    temperature: float = Field(0.8, ge=0, le=2)
    num_ctx: int = Field(4096, ge=512)
    keep_alive: str = "10m"
    max_json_retries: int = Field(2, ge=0, le=10)
    request_timeout_s: float = Field(600, gt=0)
    allow_text_on_design: bool = False

    @field_validator("keep_alive", mode="before")
    @classmethod
    def _ka_to_str(cls, v: Any) -> str:
        return str(v)


class ImageGenCfg(_Strict):
    backend: Literal["flux_schnell", "sdxl", "dry_run"] = "flux_schnell"
    width: int = Field(1024, ge=256, le=2048)
    height: int = Field(1024, ge=256, le=2048)
    steps: int = Field(4, ge=1, le=100)
    guidance: float = Field(0.0, ge=0)
    max_sequence_length: int = Field(256, ge=16, le=512)
    quantization: Literal["nf4", "none"] = "nf4"
    compute_dtype: Literal["float16", "bfloat16", "float32"] = "float16"
    flux_repo: str = "black-forest-labs/FLUX.1-schnell"
    sdxl_repo: str = "stabilityai/stable-diffusion-xl-base-1.0"
    sdxl_steps: int = Field(25, ge=1)
    sdxl_guidance: float = Field(6.0, ge=0)
    cpu_offload: bool = False

    @model_validator(mode="after")
    def _square_multiple_of_16(self) -> ImageGenCfg:
        if self.width != self.height:
            raise ValueError("v1 generates square images only: width must equal height")
        if self.width % 16:
            raise ValueError("width/height must be a multiple of 16")
        return self


class CutoutCfg(_Strict):
    model: str = "u2net"
    alpha_matting: bool = False
    erode_px: int = Field(1, ge=0, le=10)
    feather_px: int = Field(1, ge=0, le=10)
    min_speck_area: int = Field(64, ge=0)


class UpscaleCfg(_Strict):
    model: str = "realesrgan_x4plus_anime_6B"       # flat / illustration art
    general_model: str = "realesrgan_x4plus"        # photographic / painterly styles
    tile: int = Field(400, ge=0)                     # 0 = no tiling
    tile_pad: int = Field(10, ge=0)
    device: Literal["auto", "cuda", "cpu"] = "auto"


class VisionQACfg(_Strict):
    enabled: bool = False
    model: str = "qwen2.5vl:7b"
    ip_risk_threshold: float = Field(0.35, ge=0, le=1)


class QACfg(_Strict):
    clip_dedupe: bool = True
    similarity_threshold: float = Field(0.92, gt=0, le=1)
    reference_dir: str | None = None
    vision_qa: VisionQACfg = VisionQACfg()
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"
    min_resolution: int = Field(1024, ge=64)
    bg_tolerance: int = Field(30, ge=1, le=255)
    min_subject_fraction: float = Field(0.02, ge=0, le=1)
    max_subject_fraction: float = Field(0.92, ge=0, le=1)
    border_px: int = Field(6, ge=1)
    max_border_fraction: float = Field(0.02, ge=0, le=1)
    max_noise: float = Field(12.0, gt=0)


class ReviewCfg(_Strict):
    mode: Literal["manual", "auto"] = "manual"


class ExportCfg(_Strict):
    output_dir: str = "output"
    layout: Literal["by_category", "flat"] = "by_category"
    dpi: int = Field(300, ge=72, le=1200)
    jpeg_quality: int = Field(95, ge=1, le=100)
    max_scale_factor: float = Field(2.5, gt=0)
    max_dim_px: int = Field(13500, ge=1)
    max_file_mb: float = Field(300, gt=0)
    only_groups: list[str] | None = None
    alpha_bbox_threshold: int = Field(8, ge=0, le=254)


class MetadataCfg(_Strict):
    enabled: bool = True
    title_max: int = Field(100, ge=10)
    tags_max: int = Field(15, ge=1)
    tag_max_len: int = Field(30, ge=3)
    desc_max: int = Field(1000, ge=20)
    ai_disclosure: str = "Created with AI assistance."


class SafetyCfg(_Strict):
    blocklist_dir: str = "assets/blocklist"


class Settings(_Strict):
    run: RunCfg = RunCfg()
    runtime: RuntimeCfg = RuntimeCfg()
    llm: LLMCfg = LLMCfg()
    imagegen: ImageGenCfg = ImageGenCfg()
    cutout: CutoutCfg = CutoutCfg()
    upscale: UpscaleCfg = UpscaleCfg()
    qa: QACfg = QACfg()
    review: ReviewCfg = ReviewCfg()
    export: ExportCfg = ExportCfg()
    metadata: MetadataCfg = MetadataCfg()
    safety: SafetyCfg = SafetyCfg()


# -------------------------------------------------------------- products.yaml
class ProductSpec(_Strict):
    key: str
    category: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    transparent: bool
    format: Literal["png", "jpg"]
    margin_pct: float = Field(ge=0, lt=50)
    anchor: Anchor
    scale: float = Field(gt=0, le=10)
    enabled: bool = True

    @model_validator(mode="after")
    def _check(self) -> ProductSpec:
        if not _PRODUCT_KEY_RE.match(self.key):
            raise ValueError(f"product key {self.key!r} must be snake_case [a-z0-9_]")
        if self.transparent and self.format != "png":
            raise ValueError(f"{self.key}: transparent products must use format png")
        return self

    @property
    def canvas_w(self) -> int:
        return round(self.width * self.scale)

    @property
    def canvas_h(self) -> int:
        return round(self.height * self.scale)

    @property
    def file_key(self) -> str:
        """Product key as used in file names (hyphens inside fields)."""
        return self.key.replace("_", "-")


_PRODUCT_FIELDS = {"transparent", "format", "margin_pct", "anchor", "scale", "enabled"}


def parse_products(raw: Any, source: str = "products.yaml") -> list[ProductSpec]:
    """Parse the products.yaml structure into a flat, validated product list."""
    if not isinstance(raw, dict):
        raise ConfigError(f"{source}: top level must be a mapping with 'defaults' and 'groups'")
    unknown_top = set(raw) - {"defaults", "groups"}
    if unknown_top:
        raise ConfigError(f"{source}: unknown top-level keys {sorted(unknown_top)}")
    defaults = raw.get("defaults") or {}
    groups = raw.get("groups")
    if not isinstance(defaults, dict):
        raise ConfigError(f"{source}: 'defaults' must be a mapping")
    _check_keys(defaults, f"{source}: defaults")
    if not isinstance(groups, dict) or not groups:
        raise ConfigError(f"{source}: 'groups' must be a non-empty mapping")

    products: list[ProductSpec] = []
    seen: set[str] = set()
    for group, body in groups.items():
        where = f"{source}: groups.{group}"
        if not isinstance(body, dict):
            raise ConfigError(f"{where} must be a mapping of products")
        group_defaults = body.get("_") or {}
        if not isinstance(group_defaults, dict):
            raise ConfigError(f"{where}._ must be a mapping")
        _check_keys(group_defaults, f"{where}._")
        group_enabled = body.get("enabled", True)
        for key, value in body.items():
            if key in ("_", "enabled"):
                continue
            pwhere = f"{where}.{key}"
            fields: dict[str, Any] = {**defaults, **group_defaults}
            if isinstance(value, list):
                size = value
            elif isinstance(value, dict):
                if "size" not in value:
                    raise ConfigError(f"{pwhere}: mapping form needs 'size: [w, h]'")
                overrides = {k: v for k, v in value.items() if k != "size"}
                _check_keys(overrides, pwhere)
                fields.update(overrides)
                size = value["size"]
            else:
                raise ConfigError(f"{pwhere}: expected [w, h] or {{size: [w, h], ...}}")
            if (
                not isinstance(size, list) or len(size) != 2
                or not all(isinstance(s, int) and not isinstance(s, bool) and s > 0 for s in size)
            ):
                raise ConfigError(f"{pwhere}: size must be [width, height] positive integers, got {size!r}")
            if key in seen:
                raise ConfigError(f"{pwhere}: duplicate product key")
            seen.add(key)
            fields["enabled"] = bool(fields.get("enabled", True)) and bool(group_enabled)
            try:
                products.append(ProductSpec(key=key, category=group, width=size[0], height=size[1], **fields))
            except ValidationError as exc:
                raise ConfigError(f"{pwhere}: {_fmt_validation(exc)}") from exc
    return products


def _check_keys(d: dict[str, Any], where: str) -> None:
    bad = set(d) - _PRODUCT_FIELDS
    if bad:
        raise ConfigError(f"{where}: unknown keys {sorted(bad)} (allowed: {sorted(_PRODUCT_FIELDS)})")


def _fmt_validation(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = ".".join(str(x) for x in err["loc"]) or "(root)"
        parts.append(f"{loc}: {err['msg']}")
    return "; ".join(parts)


# ------------------------------------------------------------------ AppConfig
class AppConfig:
    """Validated settings plus the product catalogue and resolved paths."""

    def __init__(self, settings: Settings, products: list[ProductSpec], config_dir: Path) -> None:
        self.settings = settings
        self.all_products = products
        self.config_dir = config_dir

    # convenience accessors ------------------------------------------------
    @property
    def s(self) -> Settings:
        return self.settings

    @property
    def products(self) -> list[ProductSpec]:
        """Enabled products, honouring export.only_groups."""
        only = self.settings.export.only_groups
        return [p for p in self.all_products if p.enabled and (only is None or p.category in only)]

    @property
    def work_dir(self) -> Path:
        return Path(self.settings.runtime.work_dir).expanduser().resolve()

    @property
    def data_dir(self) -> Path:
        return self.work_dir / "data"

    @property
    def output_dir(self) -> Path:
        out = Path(self.settings.export.output_dir).expanduser()
        return out if out.is_absolute() else (self.work_dir / out)

    @property
    def models_dir(self) -> Path:
        return self.work_dir / "models"

    @property
    def db_path(self) -> Path:
        return self.work_dir / "pipeline.db"

    @property
    def blocklist_dir(self) -> Path:
        p = Path(self.settings.safety.blocklist_dir).expanduser()
        return p if p.is_absolute() else (self.config_dir / p)

    def dir(self, name: Literal["raw", "cutout", "master", "rejected"]) -> Path:
        return self.data_dir / name

    def ensure_dirs(self) -> None:
        for d in (self.work_dir, self.output_dir, self.models_dir,
                  *(self.dir(n) for n in ("raw", "cutout", "master", "rejected"))):
            d.mkdir(parents=True, exist_ok=True)

    def with_overrides(self, overrides: dict[str, Any]) -> AppConfig:
        data = self.settings.model_dump()
        for dotted, value in overrides.items():
            _set_dotted(data, dotted, value)
        return AppConfig(_validate_settings(data, "overrides"), self.all_products, self.config_dir)


def _set_dotted(data: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cur = data
    for k in keys[:-1]:
        nxt = cur.get(k)
        if not isinstance(nxt, dict):
            raise ConfigError(f"override {dotted!r}: {k!r} is not a config section")
        cur = nxt
    if keys[-1] not in cur:
        raise ConfigError(f"override {dotted!r}: unknown key {keys[-1]!r}")
    cur[keys[-1]] = value


def _validate_settings(raw: Any, source: str) -> Settings:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{source}: top level must be a mapping")
    try:
        return Settings.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"{source}: {_fmt_validation(exc)}") from exc


def _read_yaml(path: Path) -> Any:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path.name}: invalid YAML: {exc}") from exc


def parse_override(text: str) -> tuple[str, Any]:
    """Parse ``key.path=value`` (value is YAML, so 'true', '3', '[a, b]' work)."""
    if "=" not in text:
        raise ConfigError(f"override {text!r} must look like key.path=value")
    key, _, value = text.partition("=")
    return key.strip(), yaml.safe_load(value)


def load_config(
    config_path: str | Path | None = None,
    products_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> AppConfig:
    """Load both YAML files. Env vars: PIPELINE_CONFIG, PIPELINE_PRODUCTS, PIPELINE_WORK_DIR."""
    cfg_path = Path(config_path or os.environ.get("PIPELINE_CONFIG", "config.yaml")).expanduser().resolve()
    prod_src: str | Path = products_path or os.environ.get("PIPELINE_PRODUCTS") or cfg_path.parent / "products.yaml"
    prod_path = Path(prod_src).expanduser().resolve()
    raw = _read_yaml(cfg_path)
    if isinstance(raw, dict):
        for dotted, value in (overrides or {}).items():
            _set_dotted_lenient(raw, dotted, value)
        env_wd = os.environ.get("PIPELINE_WORK_DIR")
        if env_wd and "runtime.work_dir" not in (overrides or {}):
            raw.setdefault("runtime", {})["work_dir"] = env_wd
    settings = _validate_settings(raw, cfg_path.name)
    products = parse_products(_read_yaml(prod_path), prod_path.name)
    cfg = AppConfig(settings, products, cfg_path.parent)
    if settings.export.only_groups:
        known = {p.category for p in products}
        unknown = set(settings.export.only_groups) - known
        if unknown:
            raise ConfigError(f"export.only_groups: unknown groups {sorted(unknown)}")
    if not cfg.products:
        raise ConfigError("no products enabled: check products.yaml and export.only_groups")
    return cfg


def _set_dotted_lenient(data: dict[str, Any], dotted: str, value: Any) -> None:
    """Like _set_dotted but creates missing sections (validation catches typos)."""
    keys = dotted.split(".")
    cur = data
    for k in keys[:-1]:
        nxt = cur.get(k)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[k] = nxt
        cur = nxt
    cur[keys[-1]] = value
