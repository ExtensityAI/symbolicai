from ... import core
from ...symbol import Expression


class ocr(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = self.__class__.__name__

    def __call__(self, image_url: str | None = None, *, document_url: str | None = None, per_page: bool = False, **kwargs):
        url = document_url or image_url
        assert url is not None, "Provide image_url or document_url."

        # normalize local paths to file:// URIs
        if not url.startswith(("http://", "https://", "file://", "data:")):
            url = f"file://{url}"

        if document_url:
            kwargs["document_url"] = document_url
        if image_url:
            kwargs["image_url"] = image_url
        kwargs["per_page"] = per_page

        @core.ocr(image=url, **kwargs)
        def _func(_):
            pass

        return _func(self)
