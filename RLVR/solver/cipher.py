"""
Solver for text encryption (substitution cipher) problems.

The problem gives examples of encrypted text → plain text (word by word, char by char).
Strategy:
  1. Parse (encrypted_text, plaintext) pairs.
  2. Build a character-level mapping: cipher_char → plain_char.
     Match characters positionally within aligned words.
  3. For unmapped characters in the query, use vocabulary-constrained pattern matching
     to recover the missing plain chars (only 77 unique words used across the dataset).
  4. Apply the mapping to decrypt.

Notes:
  - Words are aligned: each word in the cipher maps to the corresponding word in plaintext.
  - Spaces map to spaces (preserved).
  - The cipher uses a fixed vocabulary of 77 words.
"""

from .base_solver import BaseSolver, SolverResult

# Fixed vocabulary extracted from training data (all cipher answers use exactly these words)
CIPHER_VOCABULARY: list[str] = [
    'above', 'alice', 'ancient', 'around', 'beyond', 'bird', 'book', 'bright',
    'castle', 'cat', 'cave', 'chases', 'clever', 'colorful', 'creates', 'crystal',
    'curious', 'dark', 'discovers', 'door', 'dragon', 'draws', 'dreams', 'explores',
    'follows', 'forest', 'found', 'garden', 'golden', 'hatter', 'hidden', 'imagines',
    'in', 'inside', 'island', 'key', 'king', 'knight', 'library', 'magical', 'map',
    'message', 'mirror', 'mountain', 'mouse', 'mysterious', 'near', 'ocean', 'palace',
    'potion', 'princess', 'puzzle', 'queen', 'rabbit', 'reads', 'school', 'secret',
    'sees', 'silver', 'story', 'strange', 'student', 'studies', 'teacher', 'the',
    'through', 'tower', 'treasure', 'turtle', 'under', 'valley', 'village', 'watches',
    'wise', 'wizard', 'wonderland', 'writes',
]

_VOCAB_BY_LEN: dict[int, list[str]] = {}
for _w in CIPHER_VOCABULARY:
    _VOCAB_BY_LEN.setdefault(len(_w), []).append(_w)


def _word_matches_pattern(word: str, pattern: str, char_map: dict[str, str]) -> bool:
    """
    Check if a vocabulary word matches a partially-decoded pattern.
    pattern: e.g. 'c?t' where '?' is an unmapped cipher char.
    char_map: current cipher→plain map.
    Returns True only if the word is consistent with all mapped chars and
    the unmapped chars have no conflicts.
    """
    if len(word) != len(pattern):
        return False
    unmapped_assignments: dict[str, str] = {}  # cipher_char → tentative plain_char
    for w_ch, p_ch in zip(word, pattern):
        if p_ch != "?":
            if p_ch != w_ch:
                return False
        else:
            # p_ch is '?' — check the corresponding cipher char
            # We need to track which cipher char this '?' came from
            # This function doesn't have access to cipher chars, handled in caller
            pass
    return True


def _try_recover_with_vocabulary(
    query_words: list[str],
    partial_plain_words: list[str],  # each word has '?' for unmapped chars
    cipher_query_words: list[str],   # original cipher chars per word
    char_map: dict[str, str],
) -> tuple[list[str], dict[str, str]]:
    """
    For each partially-decoded word with '?' chars, try to find the unique
    vocabulary word that matches. If found, fill in the missing char_map entries.
    Returns (resolved_words, updated_char_map).
    """
    resolved: list[str] = list(partial_plain_words)
    updated_map = dict(char_map)
    # Also track inverse map (plain → cipher) to detect conflicts
    inverse_map: dict[str, str] = {v: k for k, v in updated_map.items()}

    for i, (cipher_word, partial_plain) in enumerate(zip(cipher_query_words, partial_plain_words)):
        if "?" not in partial_plain:
            continue  # already fully decoded

        word_len = len(partial_plain)
        candidates = _VOCAB_BY_LEN.get(word_len, [])

        matching = []
        for vocab_word in candidates:
            ok = True
            tentative: dict[str, str] = {}  # cipher_ch → plain_ch for this word
            for c_ch, p_ch, v_ch in zip(cipher_word, partial_plain, vocab_word):
                if p_ch != "?":
                    if p_ch != v_ch:
                        ok = False
                        break
                else:
                    # c_ch is unmapped; v_ch is the candidate plain char
                    if c_ch in tentative and tentative[c_ch] != v_ch:
                        ok = False
                        break
                    if v_ch in inverse_map and inverse_map[v_ch] != c_ch:
                        # plain char already assigned to a different cipher char
                        ok = False
                        break
                    tentative[c_ch] = v_ch
            if ok:
                matching.append((vocab_word, tentative))

        if len(matching) == 1:
            vocab_word, tentative = matching[0]
            resolved[i] = vocab_word
            updated_map.update(tentative)
            inverse_map.update({v: k for k, v in tentative.items()})

    return resolved, updated_map


class CipherSolver(BaseSolver):

    def solve(self, prompt: str) -> SolverResult:
        lines = prompt.strip().split("\n")

        examples: list[tuple[str, str]] = []
        query: str | None = None

        for line in lines:
            line = line.strip()
            if " -> " in line:
                parts = line.split(" -> ", 1)
                if len(parts) == 2:
                    examples.append((parts[0].strip(), parts[1].strip()))
            elif line.lower().startswith("now, decrypt"):
                colon = line.find(":")
                if colon != -1:
                    query = line[colon + 1:].strip()

        if not examples:
            return SolverResult(
                answer=None, category="cipher", confidence=0.0,
                reasoning="No examples found."
            )
        if query is None:
            return SolverResult(
                answer=None, category="cipher", confidence=0.0,
                reasoning="No query found."
            )

        # Build char→char mapping from positionally-aligned word pairs
        char_map: dict[str, str] = {}
        conflicts: list[str] = []

        for enc_text, plain_text in examples:
            enc_words = enc_text.split()
            plain_words = plain_text.split()
            if len(enc_words) != len(plain_words):
                continue
            for enc_word, plain_word in zip(enc_words, plain_words):
                if len(enc_word) != len(plain_word):
                    continue
                for enc_ch, plain_ch in zip(enc_word, plain_word):
                    if enc_ch in char_map:
                        if char_map[enc_ch] != plain_ch:
                            conflicts.append(f"'{enc_ch}'→'{char_map[enc_ch]}' vs '{plain_ch}'")
                    else:
                        char_map[enc_ch] = plain_ch

        if not char_map:
            return SolverResult(
                answer=None, category="cipher", confidence=0.0,
                reasoning="Could not build any character mapping."
            )

        # First pass: apply known mapping
        query_words = query.split()
        partial_plain_words: list[str] = []
        unmapped: set[str] = set()
        for word in query_words:
            chars = []
            for ch in word:
                if ch in char_map:
                    chars.append(char_map[ch])
                else:
                    chars.append("?")
                    unmapped.add(ch)
            partial_plain_words.append("".join(chars))

        # Second pass: vocabulary-constrained recovery for any '?' chars
        if unmapped:
            resolved_words, char_map = _try_recover_with_vocabulary(
                query_words, partial_plain_words, query_words, char_map
            )
        else:
            resolved_words = partial_plain_words

        # Re-apply updated map to get final answer
        final_words: list[str] = []
        still_unmapped: set[str] = set()
        for word in query_words:
            chars = []
            for ch in word:
                if ch in char_map:
                    chars.append(char_map[ch])
                else:
                    chars.append("?")
                    still_unmapped.add(ch)
            final_words.append("".join(chars))

        answer = " ".join(final_words)
        confidence = 1.0 if not still_unmapped and not conflicts else 0.6

        reasoning = (
            f"Built char map from {len(examples)} examples ({len(char_map)} unique mappings).\n"
            + (f"Conflicts: {', '.join(conflicts)}\n" if conflicts else "")
            + (f"Still unmapped chars: {still_unmapped}\n" if still_unmapped else "")
            + f"Query: '{query}' → '{answer}'"
        )

        return SolverResult(
            answer=answer,
            category="cipher",
            confidence=confidence,
            reasoning=reasoning,
            verified=confidence == 1.0,
            extra={"char_map": char_map},
        )
