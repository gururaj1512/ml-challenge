from __future__ import annotations

import math
from rapidfuzz import fuzz

from .text import numeric_tokens, safe, structured_tokens

FEATURES = [
    'name_exact', 'name_stripped_exact', 'name_ratio', 'name_token_sort', 'name_token_set',
    'name_jaccard', 'name_length_ratio', 'name_stripped_ratio', 'name_containment',
    'name_stripped_containment', 'name_accent_ratio', 'name_accent_exact',
    'address_ratio', 'address_token_sort', 'address_token_set', 'address_jaccard',
    'address_length_ratio', 'address_exact', 'address_containment', 'left_address_missing',
    'right_address_missing', 'either_address_missing', 'both_addresses_present',
    'numeric_shared_count', 'numeric_jaccard', 'structured_shared_count', 'structured_jaccard',
    'structured_exact_any', 'name_char_tfidf_cosine', 'address_char_tfidf_cosine',
    'address_word_tfidf_cosine', 'embedding_cosine', 'exact_retrieval', 'name_reciprocal_rank',
    'address_char_reciprocal_rank', 'address_word_reciprocal_rank', 'numeric_reciprocal_rank',
    'stripped_reciprocal_rank', 'accent_reciprocal_rank', 'combined_reciprocal_rank',
    'embedding_reciprocal_rank', 'retriever_count', 'candidate_name_frequency', 'source_is_s3',
    'name_address_product', 'name_address_min', 'missing_address_x_name', 'exact_name_x_address'
]
assert len(FEATURES) == 48


def equal(a, b):
    return int(bool(a) and a == b)


def contain(a, b):
    return int(len(a) >= 4 and len(b) >= 4 and (a in b or b in a))


def jac(a: set, b: set):
    return len(a & b)/len(a | b) if (a or b) else 0.0


def ratio(a, b):
    return fuzz.ratio(a, b)/100 if a and b else 0.0


def string_features(a: str, b: str):
    if not a or not b:
        return (0.0,) * 5
    return (ratio(a, b), fuzz.token_sort_ratio(a, b)/100, fuzz.token_set_ratio(a, b)/100,
            jac(set(a.split()), set(b.split())), min(len(a),len(b))/max(len(a),len(b)))


def make_features(a: dict, b: dict, ranks: dict[str, int], lexical, embedding_cosine: float) -> dict[str, float]:
    na, nb = a['normalized_name'], b['normalized_name']
    aa, ab = a['normalized_address'], b['normalized_address']
    sa, sb = a['stripped_name'], b['stripped_name']
    nr, nts, nset, nj, nl = string_features(na, nb)
    ar, ats, aset, aj, al = string_features(aa, ab)
    nums_a, nums_b = numeric_tokens(a['business_address']), numeric_tokens(b['business_address'])
    struct_a, struct_b = structured_tokens(a['business_address']), structured_tokens(b['business_address'])
    left_missing = int(not safe(a['business_address']).strip())
    right_missing = int(not safe(b['business_address']).strip())
    sr = ratio(sa, sb)
    rr = lambda route: 1.0/ranks[route] if route in ranks else 0.0
    vals = {
        'name_exact': equal(na,nb), 'name_stripped_exact': equal(sa,sb), 'name_ratio': nr,
        'name_token_sort': nts, 'name_token_set': nset, 'name_jaccard': nj, 'name_length_ratio': nl,
        'name_stripped_ratio': sr, 'name_containment': contain(na,nb), 'name_stripped_containment': contain(sa,sb),
        'name_accent_ratio': ratio(a['accent_name'],b['accent_name']), 'name_accent_exact': equal(a['accent_name'],b['accent_name']),
        'address_ratio': ar, 'address_token_sort': ats, 'address_token_set': aset, 'address_jaccard': aj,
        'address_length_ratio': al, 'address_exact': equal(aa,ab), 'address_containment': contain(aa,ab),
        'left_address_missing': left_missing, 'right_address_missing': right_missing,
        'either_address_missing': int(left_missing or right_missing),
        'both_addresses_present': int(not left_missing and not right_missing),
        'numeric_shared_count': len(nums_a & nums_b), 'numeric_jaccard': jac(nums_a,nums_b),
        'structured_shared_count': len(struct_a & struct_b), 'structured_jaccard': jac(struct_a,struct_b),
        'structured_exact_any': int(bool(struct_a & struct_b)),
        'name_char_tfidf_cosine': lexical.cosine(a,b,'name'),
        'address_char_tfidf_cosine': lexical.cosine(a,b,'address_char'),
        'address_word_tfidf_cosine': lexical.cosine(a,b,'address_word'),
        'embedding_cosine': float(embedding_cosine), 'exact_retrieval': int('exact' in ranks),
        'name_reciprocal_rank': rr('name'), 'address_char_reciprocal_rank': rr('address_char'),
        'address_word_reciprocal_rank': rr('address_word'), 'numeric_reciprocal_rank': rr('numeric'),
        'stripped_reciprocal_rank': rr('stripped'), 'accent_reciprocal_rank': rr('accent'),
        'combined_reciprocal_rank': rr('combined'), 'embedding_reciprocal_rank': rr('embedding'),
        'retriever_count': len(ranks), 'candidate_name_frequency': math.log1p(lexical.name_frequency(nb)),
        'source_is_s3': int(b['entity_id'].startswith('S3-')),
        'name_address_product': sr*aset, 'name_address_min': min(sr,aset),
        'missing_address_x_name': right_missing*sr, 'exact_name_x_address': equal(na,nb)*ar,
    }
    if set(vals) != set(FEATURES) or not all(math.isfinite(v) for v in vals.values()):
        raise RuntimeError('Invalid feature schema or non-finite feature.')
    return {name: float(vals[name]) for name in FEATURES}
