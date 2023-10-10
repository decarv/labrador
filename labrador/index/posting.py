

class Posting:
    def __init__(self, doc_id: int, positions: list[int]):
        self.doc_id: int = doc_id
        self.positions: list[int] = positions

    def add(self, position):
        self.positions.append(position)

    @property
    def term_frequency(self) -> int:
        return len(self.positions)

    def __repr__(self):
        return f"{self.doc_id} [{self.term_frequency}]: {self.positions}"


class PostingList:
    def __init__(self):
        self._postings: list[Posting] = []
        # self._skip_pointers: dict[str, list[int]] = {}
        self._skip_pointers_space: dict[str, int] = {}  # Contains the space between skip-pointers for each term

    def __getitem__(self, term: str) -> list[int]:
        return self._posting_list[term]

    def add(self, term: str, doc_id: int):
        if term not in self._posting_list:
            self._posting_list[term] = []
        self._posting_list[term].append(doc_id)
        doc_id_index: int = len(self._posting_list[term]) - 1

        self._update_skip_pointers(term)

    def _update_skip_pointers(self, term: str) -> None:
        if term not in self._skip_pointers_space:
            self._skip_pointers_space[term] = 0
            return

        # The space is the square root of posting list size, here we check if the list has doubled in size since
        # previous restructuring.
        space: int = self._skip_pointers_space[term]
        if len(self._posting_list[term]) <= 2 * (space ** 2):
            return

        self._skip_pointers_space[term] = int(len(self._posting_list[term]) ** 0.5)

    def union(self, other):
        pass

    def intersect(self, other, term: str) -> list[int]:
        if not isinstance(other, PostingList):
            raise TypeError("Can only intersect with another PostingList")

        intersection: list[int] = []
        pla: list[int] = self._posting_list[term]
        pa: int = self._skip_pointers_space[term]
        plb: list[int] = other._posting_list[term]
        pb: int = self._skip_pointers_space[term]

        i, j = 0, 0
        while i < len(pla) and j < len(plb):
            if pla[i] == plb[j]:
                intersection.append(pla[i])
                i += 1
                j += 1
            elif pla[i] < plb[j]:
                while i < len(pla) and pla[i] < plb[j]:
                    if i % pa == 0 and pla[i] < plb[j]:
                        i += pa
                    else:
                        i += 1
            else:
                while j < len(plb) and pla[i] > plb[j]:
                    if j % pb == 0 and pla[i] > plb[j]:
                        j += pb
                    else:
                        j += 1

        return intersection

    def __repr__(self):
        representation = {}
        for term, posting_list in self._posting_list.items():
            representation[term] = []
            for i, doc_id in enumerate(posting_list):
                if i % self._skip_pointers_space[term] == 0:
                    representation[term].append(f"*{doc_id}")
                else:
                    representation[term].append(f"{doc_id}")
            representation[term] = str(representation[term])
        return str(representation)

