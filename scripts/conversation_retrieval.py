"""Opt-in lexical conversation retrieval; prior human questions are hints, not facts."""
from collections import Counter
import json
import math
import re
from grounded_chat import Retriever, terms

POLICY = 'human_context_decay_v1'
RESET = re.compile(r'\b(?:new topic|switch topics|switching topics|different topic|unrelated question|start over)\b', re.I)


class ConversationRetriever(Retriever):
    """Always score the current ask, plus at most four decaying human questions.

    Explicit topic-reset phrases clear the context window. Otherwise a new topic
    must compete on current evidence: there is no claim of semantic intent
    detection. Scores/weights and both contributing queries remain inspectable.
    """
    def retrieve(self, question, history=(), *, top_k=2, max_chars=26000):
        if not isinstance(question, str) or not question.strip() or top_k < 1 or max_chars < 1:
            raise ValueError('Nonempty question and positive retrieval limits required')
        prior = [m['content'] for m in history if m['role'] == 'user']
        # A reset is an application-level search hint, not a factual instruction.
        if RESET.search(question):
            prior = []
        else:
            for i in range(len(prior)-1, -1, -1):
                if RESET.search(prior[i]):
                    prior = prior[i:]
                    break
        query = Counter(terms(question)); current = dict(query); contributions = []
        for distance, text in enumerate(reversed(prior[-4:]), start=1):
            weight = .5**distance
            words = sorted(set(terms(text)))
            contributions.append({'distance': distance, 'weight': weight, 'terms': words})
            for word in words:
                query[word] += weight
        ranked = []
        for packet, counts, length in zip(self.packets, self.counts, self.lengths):
            matched = sorted(set(query) & set(counts)); score = 0.
            for word in matched:
                tf = counts[word]
                idf = math.log(1+(len(self.packets)-self.df[word]+.5)/(self.df[word]+.5))
                score += query[word]*idf*tf*2.2/(tf+1.2*(.25+.75*length/self.average))
            if score >= 1.:
                ranked.append((score, packet['id'], matched, packet))
        ranked.sort(key=lambda row: (-row[0], row[1]))
        selected = []; omitted = []; used = 0
        for score, pid, matched, packet in ranked[:top_k]:
            size = len(json.dumps(packet, ensure_ascii=False))
            if used+size > max_chars:
                omitted.append(pid)
                continue
            selected.append({'packet': packet, 'score': round(score, 6), 'matched_terms': matched})
            used += size
        return {'query_terms': dict(sorted(query.items())), 'current_query_terms': current,
                'context_contributions': contributions, 'explicit_topic_reset': bool(RESET.search(question)),
                'method': POLICY, 'matches': selected, 'budget_omitted': omitted, 'top_k': top_k,
                'status': 'candidates_require_interpretation' if selected else 'insufficient_evidence',
                'collection_sha256': self.collection['packets_sha256']}
