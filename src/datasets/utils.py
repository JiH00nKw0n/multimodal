from spacy.tokens import Doc, Token, Span
from typing import Union, List, Dict, Optional, Any, Callable

def swap_spans(tokens: List[Token], span1: Span, span2: Span) -> List[Token]:
    """
    두 명사구의 위치를 교환합니다.

    Returns:
    - List[Token]: 위치가 교환된 새로운 Token 리스트
    """

    start1, end1 = span1.start, span1.end
    start2, end2 = span2.start, span2.end

    if start1 < start2:
        return tokens[:start1] + tokens[start2:end2] + tokens[end1:start2] + tokens[start1:end1] + tokens[end2:]
    else:
        return tokens[:start2] + tokens[start1:end1] + tokens[end2:start1] + tokens[start2:end2] + tokens[end1:]
