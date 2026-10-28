"""Offline application-event contract; no classifier, prompt, model or activation."""
from dataclasses import dataclass, replace


class BoundaryError(ValueError):
    pass


DETAILS = ('brief', 'normal', 'deep')


@dataclass(frozen=True)
class PriorTurn:
    role: str
    text: str


@dataclass(frozen=True)
class ConversationState:
    source_scope: tuple = ()
    detail: str = 'normal'
    prior_turns: tuple = ()


@dataclass(frozen=True)
class QuestionRequest:
    question: str
    source_scope: tuple
    detail: str
    prior_turns: tuple


def source_scope(values, available):
    if (not isinstance(values, (tuple, list)) or not 1 <= len(values) <= len(available)
            or any(not isinstance(v, str) or v not in available for v in values)
            or len(values) != len(set(values))):
        raise BoundaryError('Require a nonempty subset of distinct configured source IDs')
    return tuple(values)


def apply_event(state, event, available_source_ids):
    """Caller supplies event intent; free-form text is never reclassified here.

    Structured UI controls may update scope/detail without asking a question.
    This contract cannot authenticate a caller or infer natural-language intent.
    Prior turns remain opaque context; they never supply the current question.
    """
    if not isinstance(available_source_ids, (list, tuple)):
        raise BoundaryError('Configured source IDs require an explicit sequence')
    available = tuple(available_source_ids)
    if (not available or any(not isinstance(s, str) or not s for s in available)
            or len(available) != len(set(available))):
        raise BoundaryError('Configured source inventory differs')
    if (type(state) is not ConversationState or type(state.source_scope) is not tuple or type(state.prior_turns) is not tuple
            or state.detail not in DETAILS
            or any(type(t) is not PriorTurn or t.role not in ('user', 'assistant')
                   or not isinstance(t.text, str) for t in state.prior_turns)):
        raise BoundaryError('Invalid application conversation state')
    if state.source_scope:
        source_scope(state.source_scope, available)
    if not isinstance(event, dict) or 'event' not in event:
        raise BoundaryError('Explicit application event required')
    kind = event['event']
    if kind == 'set_source_scope':
        if set(event) != {'event', 'source_ids'}:
            raise BoundaryError('Source control accepts source IDs only')
        return replace(state, source_scope=source_scope(event['source_ids'], available)), None
    if kind == 'set_detail':
        if set(event) != {'event', 'detail'} or event['detail'] not in DETAILS:
            raise BoundaryError('Unknown detail control')
        return replace(state, detail=event['detail']), None
    if kind == 'ask':
        if set(event) != {'event', 'question'} or not isinstance(event['question'], str) or not event['question'].strip():
            raise BoundaryError('Question event requires the explicit current question only')
        scope = source_scope(state.source_scope, available)
        request = QuestionRequest(event['question'], scope, state.detail, state.prior_turns)
        # Compiling a request does not record an answer, accept a factual claim
        # or add a turn to trusted memory; completion is outside this module.
        return state, request
    raise BoundaryError('Unknown application event')
