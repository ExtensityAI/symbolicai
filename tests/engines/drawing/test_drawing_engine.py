import os
from pathlib import Path

import pytest

from symai.backend.settings import SYMAI_CONFIG
from symai.extended import Interface

DRAWING_ENGINE = SYMAI_CONFIG.get("DRAWING_ENGINE_MODEL")

OAI_MODELS = ["dall-e-2", "dall-e-3", "gpt-image-1"]
BFL_MODELS = ["flux-dev", "flux-pro", "flux-pro-1.1", "flux-pro-1.1-ultra"]
GEMINI_IMAGE_MODELS = ["gemini-2.5-flash-image", "gemini-3-pro-image-preview"]


@pytest.mark.skipif(not DRAWING_ENGINE.startswith("dall-e-") and not DRAWING_ENGINE.startswith("gpt-image-"), reason="Not an OpenAI model")
@pytest.mark.parametrize("model", OAI_MODELS)
def test_gpt_image_create(model):
    gpt_image = Interface('gpt_image')
    paths = gpt_image('a fluffy cat with a cowboy hat', model=model, size=1024, response_format='url')
    assert os.path.exists(paths[0]) and os.path.getsize(paths[0]) > 0


@pytest.mark.skipif(not DRAWING_ENGINE.startswith("dall-e-") and not DRAWING_ENGINE.startswith("gpt-image-"), reason="Not an OpenAI model")
@pytest.mark.parametrize("model", ['dall-e-2'])
def test_gpt_image_variation(model):
    img_path = Path(__file__).parent.parent.parent.parent / 'assets' / 'images' / 'cat.png'
    gpt_image = Interface('gpt_image')
    paths = gpt_image(
        operation='variation',
        model=model,
        image_path=img_path,
        n=1,
        size=1024,
        response_format='url',
    )
    assert os.path.exists(paths[0]) and os.path.getsize(paths[0]) > 0


@pytest.mark.skipif(not DRAWING_ENGINE.startswith("dall-e-") and not DRAWING_ENGINE.startswith("gpt-image-"), reason="Not an OpenAI model")
@pytest.mark.skipif(DRAWING_ENGINE.startswith("dall-e-"), reason="For whatever reason, this test doesn't work for dall-e-2 even though I followed the documentation. Not sure what's wrong. Maybe it's a bug in the OpenAI API (testing version: openai==1.76.0")
@pytest.mark.parametrize("model,quality", [('gpt-image-1', 'medium')])
def test_gpt_image_edit(model, quality):
    img_path = Path(__file__).parent.parent.parent.parent / 'assets' / 'images' / 'cat.png'
    gpt_image = Interface('gpt_image')

    prompt = (
        "Convert the cat in this image into a character from The Elder Scrolls IV: Oblivion, "
        "complete with medieval armor, a nocturnal-fantasy style helmet, zoomed out, and ancient scrolls in the background."
    )

    paths = gpt_image(
        prompt=prompt,
        operation='edit',
        model=model,
        image_path=img_path,
        n=1,
        size=1024,
        quality=quality,
    )

    assert os.path.exists(paths[0]) and os.path.getsize(paths[0]) > 0


@pytest.mark.skipif(not DRAWING_ENGINE.startswith("flux"), reason="Not a Black Forest Labs model")
@pytest.mark.parametrize("model", BFL_MODELS)
def test_flux_image_create(model):
    flux = Interface('flux')
    paths = flux(
        "a fluffy cat with a cowboy hat",
        operation='create',
        model=model,
        width=1024,
        height=768,
        steps=50,
        guidance=7.5,
        seed=42,
        safety_tolerance=2
    )
    assert os.path.exists(paths[0]) and os.path.getsize(paths[0]) > 0


@pytest.mark.skipif(
    not DRAWING_ENGINE.startswith(("gemini-2.5-flash-image", "gemini-3-pro-image-preview")),
    reason="Not a Gemini image model",
)
@pytest.mark.parametrize("model", GEMINI_IMAGE_MODELS)
def test_gemini_image_create(model):
    nanobanana = Interface("nanobanana")
    paths = nanobanana(
        "a fluffy cat with a cowboy hat",
        operation="create",
        model=model,
    )
    assert os.path.exists(paths[0]) and os.path.getsize(paths[0]) > 0


if __name__ == '__main__':
    pytest.main()
