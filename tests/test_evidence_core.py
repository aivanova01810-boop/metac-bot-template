"""Ledger, citation checks and shrinkage with canned model outputs; no network, no SDK."""
import json
import math
import unittest
from datetime import datetime, timezone

from evidence_bot import ledger
from evidence_bot.analyst import (ContractError, ShrinkPolicy, apply_verdict, check_proposal,
                                  decide, parse_json, render_reasoning, verifier_prompt)

AS_OF = datetime(2026, 10, 15, 12, tzinfo=timezone.utc)
ARTICLES = [
    {'title': 'Parliament schedules vote', 'summary': 'Vote set for Nov 3.', 'url': 'https://www.reuters.com/a',
     'published_at': '2026-10-14T09:00:00Z'},
    {'title': 'Coalition partner wavers', 'summary': 'Second party undecided.', 'url': 'https://apnews.com/b',
     'published_at': datetime(2026, 10, 13, tzinfo=timezone.utc)},
    {'title': 'Undated blog post', 'summary': 'No timestamp.', 'url': 'https://blog.example/c', 'published_at': None},
    {'title': 'Leaked tomorrow story', 'summary': 'Published after as_of.', 'url': 'https://x.example/d',
     'published_at': '2026-10-16T00:00:00+00:00'},
    {'title': 'Parliament schedules vote', 'summary': 'duplicate', 'url': 'https://www.reuters.com/a'},
]

def proposal(**changes):
    base = {'outside_view': {'reference_class': 'scheduled confidence votes', 'probability': 0.3},
            'status_quo': 'Government survives', 'key_uncertainties': ['turnout'],
            'probability_yes': 0.6, 'summary': 'Vote is scheduled; partner wavering.',
            'claims': [{'claim': 'Vote scheduled for Nov 3', 'direction': 'yes', 'evidence_ids': ['E1']},
                       {'claim': 'Partner undecided', 'direction': 'yes', 'evidence_ids': ['[E2]', 'E9']},
                       {'claim': 'Blog says it will fail', 'direction': 'no', 'evidence_ids': ['E3', 'E4']},
                       {'claim': 'Parliaments rarely topple governments', 'direction': 'neutral', 'evidence_ids': []}]}
    return base | changes

def logit(p):
    return math.log(p/(1-p))


class LedgerTests(unittest.TestCase):
    def test_round_trip_dedups_and_keeps_dates(self):
        items = ledger.build(ARTICLES)
        self.assertEqual([e.id for e in items], ['E1', 'E2', 'E3', 'E4'])
        self.assertEqual(items[0].source, 'reuters.com')
        text = 'Some preamble\n' + ledger.render(items)
        self.assertEqual(ledger.parse(text), items)
        self.assertIsNone(items[2].published_at)

    def test_naive_or_garbage_dates_are_unknown(self):
        self.assertIsNone(ledger.parse_time('2026-10-14T09:00:00'))
        self.assertIsNone(ledger.parse_time('yesterday'))
        self.assertEqual(ledger.parse(''), [])
        self.assertEqual(ledger.render([]), '')


class AnalystTests(unittest.TestCase):
    def setUp(self):
        self.items = ledger.build(ARTICLES)

    def test_citations_need_existing_dated_past_items(self):
        checked = check_proposal(proposal(), self.items, AS_OF)
        claims = checked['claims']
        self.assertEqual([c['cited'] for c in claims], [True, True, False, False])
        self.assertEqual(claims[1]['evidence_ids'], ['E2'])
        self.assertEqual(claims[1]['dropped_ids'], ['E9'])
        # Undated and future items never count as support.
        self.assertEqual(claims[2]['dropped_ids'], ['E3', 'E4'])

    def test_shrink_moves_only_toward_outside_view(self):
        checked = check_proposal(proposal(), self.items, AS_OF)
        decision = decide(checked, None, [])
        # Two of three directional claims are backed: weight 0.5 + 0.5*2/3.
        self.assertAlmostEqual(decision.support, 2/3)
        self.assertAlmostEqual(decision.weight, 0.5 + 0.5*2/3)
        expected = logit(0.3) + decision.weight*(logit(0.6)-logit(0.3))
        self.assertAlmostEqual(logit(decision.probability), expected)
        self.assertTrue(0.3 < decision.probability < 0.6)

    def test_verifier_can_only_pull_back(self):
        checked = check_proposal(proposal(), self.items, AS_OF)
        verdict = {'claim_checks': [{'index': 0, 'supported': False, 'note': 'wrong date'},
                                    {'index': 7, 'supported': False, 'note': 'ignored'}],
                   'critical_issues': ['misread criteria'], 'accept': True}
        accepted, issues = apply_verdict(checked, verdict)
        self.assertFalse(accepted)
        rejected = decide(checked, accepted, issues)
        self.assertAlmostEqual(rejected.support, 1/3)
        self.assertAlmostEqual(rejected.weight, (0.5 + 0.5/3)*0.5)
        clean = decide(check_proposal(proposal(), self.items, AS_OF), True, [])
        self.assertLess(abs(rejected.probability-0.3), abs(clean.probability-0.3))

    def test_empty_ledger_uses_analyst_estimate_unchanged(self):
        checked = check_proposal(proposal(), [], AS_OF)
        # Even a verifier rejection does not shrink when nothing could be cited.
        decision = decide(checked, False, ['misread criteria'], ShrinkPolicy(), evidence_available=False)
        self.assertEqual((decision.probability, decision.weight, decision.support), (0.6, 1.0, 0.0))
        self.assertFalse(decision.evidence_available)
        self.assertIn('used unchanged', render_reasoning(checked, decision, []))
        # With a ledger but no backed claims the old floor still applies.
        self.assertEqual(decide(checked, None, []).weight, 0.5)

    def test_contract_errors_and_percent_answers(self):
        self.assertEqual(check_proposal(proposal(probability_yes=60), self.items, AS_OF)['probability_yes'], 0.6)
        for bad in (proposal(probability_yes='high'), proposal(probability_yes=150),
                    proposal(claims=[{'claim': 'x', 'direction': 'maybe', 'evidence_ids': []}]),
                    {'claims': []}):
            with self.subTest(bad=bad), self.assertRaises(ContractError):
                check_proposal(bad, self.items, AS_OF)
        with self.assertRaises(ContractError):
            apply_verdict(check_proposal(proposal(), self.items, AS_OF), {'accept': 'yes'})

    def test_parse_json_tolerates_fences(self):
        self.assertEqual(parse_json('```json\n{"a": 1}\n```'), {'a': 1})
        for bad in ('no json here', '{broken', '[1, 2]'):
            with self.subTest(bad=bad), self.assertRaises(ContractError):
                parse_json(bad)

    def test_reasoning_and_verifier_prompt_mention_citations(self):
        checked = check_proposal(proposal(), self.items, AS_OF)
        text = render_reasoning(checked, decide(checked, True, []), self.items)
        self.assertIn('E1 (reuters.com)', text)
        self.assertIn('not counted', text)
        prompt = verifier_prompt('Q', ledger.render(self.items), proposal())
        self.assertIn('"index": 3', prompt)
        self.assertIn(ledger.HEADER, prompt)


if __name__ == '__main__':
    unittest.main()
