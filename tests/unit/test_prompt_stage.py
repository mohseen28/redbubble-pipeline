from __future__ import annotations

import pytest

from pipeline.deps import Deps
from pipeline.llm.dry_run import DryRunLLM
from pipeline.schemas import BriefBatch, BriefDraft
from pipeline.stages import image, prompt
from pipeline.stages.prompt import PromptRequest, kebab, make_slug, trigram_vector


def test_slug_rules():
    assert kebab("Cozy Mushroom Cottage!") == "cozy-mushroom-cottage"
    long = kebab("a very long niche name that keeps going and going forever and ever")
    assert len(long) <= 40 and not long.endswith("-")
    assert make_slug("cozy mushroom cottage", set()) == "cozy-mushroom-cottage-01"
    assert make_slug("cozy mushroom cottage", {"cozy-mushroom-cottage-01"}) == "cozy-mushroom-cottage-02"


def test_generate_creates_new_states_with_listing(cfg):
    deps = Deps(cfg, dry_run=True)
    states = prompt.generate(PromptRequest(niche="cozy mushroom cottage", n=7), cfg, deps)
    assert len(states) == 7
    assert [s.slug for s in states][:2] == ["cozy-mushroom-cottage-01", "cozy-mushroom-cottage-02"]
    assert len({s.slug for s in states}) == 7
    for s in states:
        assert s.status == "new" and s.brief is not None
        assert s.brief.text_on_design is None
        assert "plain solid" in s.brief.prompt.lower()
        assert s.listing and s.listing["description"].endswith("Created with AI assistance.")
    # briefs_per_niche=5 -> two brief calls + one listing call per design
    assert deps.llm.calls == 2 + 7


def test_cap_by_max_designs_per_run(tmp_path):
    from tests.conftest import make_cfg
    cfg = make_cfg(tmp_path, **{"run.max_designs_per_run": 2, "metadata.enabled": False})
    states = prompt.generate(PromptRequest(niche="axolotl barista", n=10), cfg, Deps(cfg, dry_run=True))
    assert len(states) == 2 and states[0].listing is None


def test_auto_niche(cfg):
    states = prompt.generate(PromptRequest(auto_niche=True, n=4), cfg, Deps(cfg, dry_run=True))
    assert len(states) == 4
    assert len({s.brief.niche for s in states}) >= 1


def test_blocked_niche_refused(cfg):
    with pytest.raises(ValueError, match="safety"):
        prompt.generate(PromptRequest(niche="pikachu fan art", n=1), cfg, Deps(cfg, dry_run=True))


class _BadLLM(DryRunLLM):
    def complete_json(self, system, user, schema, **kw):
        if schema is BriefBatch:
            self.calls += 1
            mk = lambda c, p: BriefDraft(concept=c, prompt=p, style="flat", palette=["#112233"],  # noqa: E731
                                        bg_color="#ffffff", text_on_design="HELLO")
            return BriefBatch(briefs=[
                mk("A Mickey Mouse balloon.", "Mickey Mouse balloon on a plain solid white background."),
                mk("A frog in the style of a famous painter.", "frog painted in the style of Monet"),
                mk("A sleepy frog on a lily pad at dawn.", "A sleepy frog on a lily pad, flat vector."),
                mk("A sleepy frog on a lily pad at dawn!", "Near duplicate frog."),
            ])
        return super().complete_json(system, user, schema, **kw)


def test_post_process_filters_unsafe_duplicates_and_text(cfg):
    deps = Deps(cfg, dry_run=True, llm=_BadLLM())
    states = prompt.generate(PromptRequest(niche="pond frogs", n=1), cfg, deps)
    assert len(states) == 1
    b = states[0].brief
    assert b.concept == "A sleepy frog on a lily pad at dawn."
    assert b.text_on_design is None                       # text disabled in v1
    assert b.prompt.endswith("Isolated on a plain solid white background.")


def test_trigram_similarity():
    a, b = trigram_vector("A sleepy frog on a lily pad"), trigram_vector("A sleepy frog on a lily pad!")
    c = trigram_vector("A rocket launching over a desert")
    assert float(a @ b) > 0.95 and float(a @ c) < 0.5


def test_image_stage_dry_run_new_seed_each_time(cfg):
    cfg.ensure_dirs()
    deps = Deps(cfg, dry_run=True)
    st = prompt.generate(PromptRequest(niche="cozy mushroom cottage", n=1), cfg, deps)[0]
    g1 = image.run(st, cfg, deps)
    g2 = image.run(g1.patch(status="new"), cfg, deps)
    assert g1.status == "generated" and g1.raw_image.endswith("v1.png") and g2.raw_image.endswith("v2.png")
    assert g1.seed != g2.seed
