from typing import TYPE_CHECKING, List, Tuple


if TYPE_CHECKING:
    from spacy.tokens import Doc


class AspectExtractor:
    def __init__(self, spacy_model: str) -> None:
        super().__init__()
        import spacy

        self.nlp = spacy.load(spacy_model)

    def find_groups(self, aspect_mask: List[bool]):
        start = None
        for idx, flag in enumerate(aspect_mask):
            if flag:
                if start is None:
                    start = idx
            else:
                if start is not None:
                    yield slice(start, idx)
                    start = None
        if start is not None:
            yield slice(start, idx + 1)

    def __call__(self, texts: List[str]) -> Tuple[List["Doc"], List[slice]]:
        aspects_list = []
        docs = list(self.nlp.pipe(texts))
        for doc in docs:
            aspect_mask = [token.pos_ in ("NOUN", "PROPN") for token in doc]
            aspects_list.append(list(self.find_groups(aspect_mask)))
        return docs, aspects_list
