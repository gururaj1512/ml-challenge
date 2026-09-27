from __future__ import annotations

from collections import Counter
from functools import lru_cache
import math
import re
import unicodedata

LEGAL = frozenset('inc incorporated corp corporation co company llc llp ltd limited private pvt plc'.split())
FIELDS = ('name', 'stripped', 'accent', 'address_char', 'address_word', 'combined', 'numeric')
WORD = re.compile(r'(?u)\b[\w/-]+\b')
NUMBER = re.compile(r'(?u)\b[\w/-]*\d[\w/-]*\b')
STRUCT = re.compile(r'(?u)\b[^\W_]+(?:[-/][^\W_]+)*\b')


def safe(value) -> str:
    if value is None:
        return ''
    if isinstance(value, float) and math.isnan(value):
        return ''
    return str(value)


def normalize_text(value) -> str:
    text = unicodedata.normalize('NFKC', safe(value)).casefold().replace('&', ' and ')
    return ' '.join(''.join(ch if unicodedata.category(ch)[0] in {'L', 'N', 'M'} else ' '
                            for ch in text).split())


def strip_name(value) -> str:
    # Preserves the pinned feature's behavior: removes listed tokens anywhere.
    return ' '.join(t for t in normalize_text(value).split() if t not in LEGAL)


def accent_fold(value) -> str:
    text = unicodedata.normalize('NFKC', safe(value)).casefold()
    out = []
    for ch in text:
        if 'LATIN' in unicodedata.name(ch, ''):
            out.extend(c for c in unicodedata.normalize('NFD', ch) if unicodedata.category(c) != 'Mn')
        else:
            out.append(ch)
    return normalize_text(''.join(out))


def numeric_tokens(value) -> set[str]:
    # Intentionally preserves the original V4 numeric view for route/feature compatibility.
    return {t for t in NUMBER.findall(normalize_text(value)) if len(t) >= 2}


def structured_tokens(value) -> set[str]:
    """Secondary view; fixes the old regex missing 5A1 and G-1 / spaced 14 / 15."""
    text = unicodedata.normalize('NFKC', safe(value)).casefold()
    text = re.sub(r'\s*([/-])\s*', r'\1', text)
    return {t for t in STRUCT.findall(text) if any(c.isdigit() for c in t)}


def char_terms(text: str) -> list[str]:
    # Equivalent to sklearn's char_wb (3,5), including short padded words.
    result = []
    for word in text.split():
        padded = ' ' + word + ' '
        for n in (3, 4, 5):
            if len(padded) < n:
                break
            result.extend(padded[i:i+n] for i in range(len(padded)-n+1))
        if len(padded) < 3:
            result.append(padded)
    return result


def view(record: dict, field: str) -> str:
    if field == 'name':
        return record['normalized_name']
    if field == 'stripped':
        return record['stripped_name']
    if field == 'accent':
        return record['accent_name']
    if field in {'address_char', 'address_word', 'numeric'}:
        return record['normalized_address']
    if field == 'combined':
        return (record['normalized_name'] + ' ' + record['normalized_address']).strip()
    raise KeyError(field)


def terms(field: str, value: str) -> list[str]:
    if field == 'address_word':
        w = WORD.findall(value)
        return w + [a + ' ' + b for a, b in zip(w, w[1:])]
    if field == 'numeric':
        return sorted(numeric_tokens(value))
    return char_terms(value)


def encode_term(term: str) -> str:
    # Lossless encoding, not small-bucket hashing. Unicode/whitespace n-grams remain distinct.
    return 'x' + term.encode('utf-8').hex()


def encoded_counts(field: str, value: str) -> Counter:
    return Counter(encode_term(t) for t in terms(field, value))


def record_from_row(row: dict) -> dict:
    name = safe(row.get('business_name'))
    address = safe(row.get('business_address'))
    country = safe(row.get('country')).strip()
    # Reuse already-produced views, but preserve original fields separately.
    result = {
        'entity_id': safe(row['entity_id']), 'business_name': name,
        'business_address': address, 'country': country,
        'normalized_name': safe(row['normalized_name']) if 'normalized_name' in row else normalize_text(name),
        'stripped_name': safe(row['stripped_name']) if 'stripped_name' in row else strip_name(name),
        'accent_name': safe(row['accent_name']) if 'accent_name' in row else accent_fold(name),
        'normalized_address': safe(row['normalized_address']) if 'normalized_address' in row else normalize_text(address),
    }
    return result
