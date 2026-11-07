from __future__ import annotations

import pytest

from pipeline.utils import text_safety
from pipeline.utils.text_safety import Blocklist, levenshtein, normalize
from tests.conftest import REPO


@pytest.fixture(scope="module")
def bl() -> Blocklist:
    return text_safety.load_blocklist(REPO / "assets/blocklist")


@pytest.mark.parametrize("text", ["PIKACHU in a hat", "Nike shoes", "a Coca-Cola can", "cocacola",
                                  "Pokémon fan art", "MAY THE FORCE BE WITH YOU", "Darth Vader helmet"])
def test_exact_case_insensitive(bl, text):
    assert not bl.check(text).ok


@pytest.mark.parametrize("text", ["Mickey Mous ears", "Starbuks cup", "darth vadr", "Taylor Swiftt",
                                  "Lamborgini car", "Spongebobb"])
def test_fuzzy(bl, text):
    res = bl.check(text)
    assert not res.ok
    assert any("fuzzy" in h for h in res.hits)


@pytest.mark.parametrize("text", ["painted in the style of a famous artist", "In The Manner Of old masters",
                                  "inspired by Monet", "a ghibliesque valley", "artwork by someone",
                                  "trending on artstation"])
def test_artist_style(bl, text):
    res = bl.check(text)
    assert not res.ok
    assert any(h.startswith("artist-style") for h in res.hits)


@pytest.mark.parametrize("text", [
    "A cozy mushroom cottage with a glowing window, flat vector sticker, plain solid cream background.",
    "Cute axolotl with metallic gold accents, people smiling, on a plain solid light blue background.",
    "A caterpillar transforms into a butterfly on a picturesque river bank, grotesque gargoyle, mobius strip.",
    "A jaguar in a camper van, retro sunset, kafkaesque mood, channel the vibes.",
])
def test_no_false_positives_on_ordinary_prompts(bl, text):
    assert bl.check(text).ok, bl.check(text).hits


def test_module_check_and_empty(bl):
    assert text_safety.check(None, bl).ok
    assert text_safety.check("", REPO / "assets/blocklist").ok
    assert not text_safety.check("lego bricks", REPO / "assets/blocklist").ok


def test_helpers():
    assert normalize("Café  CRÈME, l'été!") == "cafe creme lete"
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("abc", "abcdef", max_dist=1) == 2


def test_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        Blocklist.from_dir(tmp_path / "none")
    (tmp_path / "brands.txt").write_text("acme\n")
    with pytest.raises(FileNotFoundError, match=r"characters\.txt"):
        Blocklist.from_dir(tmp_path)
