"""Bot wiring with stub LLMs and a fake news fetcher: no network, no Metaculus calls."""
import asyncio
import json
import os
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import AsyncMock, patch

from forecasting_tools import BinaryQuestion, GeneralLlm, ReasonedPrediction

import main
from evidence_bot.analyst import ContractError
from evidence_bot.bot import EvidenceAnalystBot, nordrouter_llm
from evidence_bot.news import asknews_fetcher

ANALYST = json.dumps({
    'outside_view': {'reference_class': 'announced votes that happen on schedule', 'probability': 0.7},
    'status_quo': 'Vote proceeds', 'key_uncertainties': ['delay'], 'probability_yes': 0.85,
    'summary': 'Vote scheduled and whipped.',
    'claims': [{'claim': 'Vote set for Nov 3', 'direction': 'yes', 'evidence_ids': ['E1']},
               {'claim': 'Speaker confirmed agenda', 'direction': 'yes', 'evidence_ids': ['E2']}]})
VERDICT = json.dumps({'claim_checks': [{'index': 0, 'supported': True, 'note': 'ok'},
                                       {'index': 1, 'supported': True, 'note': 'ok'}],
                      'critical_issues': [], 'accept': True})


class StubLlm:
    def __init__(self, *responses):
        self.responses, self.prompts = list(responses), []

    async def invoke(self, prompt):
        self.prompts.append(prompt)
        # The last response repeats so concurrent samples all get an answer.
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]


async def fake_news(question):
    return [{'title': 'Parliament sets vote for Nov 3', 'summary': 'Scheduled.', 'url': 'https://www.reuters.com/x',
             'published_at': '2026-10-14T09:00:00Z'},
            {'title': 'Speaker confirms agenda', 'summary': 'Confirmed.', 'url': 'https://apnews.com/y',
             'published_at': '2026-10-14T10:00:00Z'}]


def question():
    return BinaryQuestion(question_text='Will parliament hold the vote by Nov 30?', id_of_question=101, id_of_post=202,
                          page_url='https://www.metaculus.com/questions/202', background_info='Context.',
                          resolution_criteria='Resolves YES if the vote happens.', fine_print='',
                          community_prediction_at_access_time=0.8)


class BotTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)

    def bot(self, analyst=None, verifier=None, news=fake_news, shadows=('no_research',)):
        stub = GeneralLlm(model='openai/stub-model')
        return EvidenceAnalystBot(
            research_reports_per_question=1, predictions_per_research_report=5,
            publish_reports_to_metaculus=False, enable_summarize_research=False,
            llms={'default': stub, 'parser': stub, 'summarizer': stub, 'researcher': 'no_research'},
            analyst_llm=analyst or StubLlm(ANALYST), verifier_llm=verifier or StubLlm(VERDICT),
            news_fetcher=news, journal_folder=self.folder.name, shadow_variants=shadows)

    def journal(self):
        return [json.loads(line) for f in Path(self.folder.name).glob('*.jsonl') for line in f.read_text(encoding='utf-8').splitlines()]

    def test_end_to_end_median_of_five_with_one_shadow(self):
        analyst = StubLlm(ANALYST)
        report = asyncio.run(self.bot(analyst=analyst).forecast_question(question()))
        # Both claims cited and verified: full weight, so the analyst's 0.85 stands.
        self.assertAlmostEqual(report.prediction, 0.85)
        kinds = Counter((r['kind'], r.get('variant')) for r in self.journal())
        self.assertEqual(kinds[('research', None)], 1)
        self.assertEqual(kinds[('forecast', 'main')], 5)
        self.assertEqual(kinds[('forecast', 'no_research')], 1)
        self.assertEqual(kinds[('shadow', 'no_research')], 1)
        shadow = next(r for r in self.journal() if r['kind'] == 'forecast' and r['variant'] == 'no_research')
        # Without a ledger nothing is shrunk: the analyst's 0.85 is used as is.
        self.assertEqual(shadow['decision']['weight'], 1.0)
        self.assertAlmostEqual(shadow['decision']['probability'], 0.85)
        self.assertEqual(shadow['community_prediction'], 0.8)
        self.assertIn('E1 (reuters.com)', report.explanation)
        self.assertNotIn('0.8', analyst.prompts[0].split('Resolution criteria')[0].replace('Nov 30', ''))

    def test_template_shadow_and_failing_shadow_never_block_main(self):
        template = AsyncMock(return_value=ReasonedPrediction(prediction_value=0.42, reasoning='template'))
        with patch.object(main.FallTemplateBot2026, '_run_forecast_on_binary', template):
            bot = self.bot(shadows=('template', 'bogus'))
            prediction = asyncio.run(bot._run_forecast_on_binary(question(), asyncio.run(bot.run_research(question()))))
        self.assertAlmostEqual(prediction.prediction_value, 0.85)
        records = self.journal()
        self.assertIn(0.42, [r.get('probability') for r in records if r['kind'] == 'shadow'])
        self.assertEqual([r['variant'] for r in records if r['kind'] == 'shadow_failed'], ['bogus'])

    def test_contract_retry_then_failure(self):
        bot = self.bot(analyst=StubLlm('not json', ANALYST))
        research = asyncio.run(bot.run_research(question()))
        decision, _ = asyncio.run(bot.evidence_forecast(question(), research, 'main'))
        self.assertAlmostEqual(decision.probability, 0.85)
        bot = self.bot(analyst=StubLlm('still not json'))
        with self.assertRaises(ContractError):
            asyncio.run(bot.evidence_forecast(question(), research, 'main'))
        self.assertEqual([r['kind'] for r in self.journal()][-1], 'forecast_failed')

    def test_verifier_rejection_shrinks_and_broken_verifier_is_no_verdict(self):
        reject = json.dumps({'claim_checks': [{'index': 0, 'supported': False, 'note': 'date wrong'}],
                             'critical_issues': ['misread criteria'], 'accept': False})
        bot = self.bot(verifier=StubLlm(reject))
        research = asyncio.run(bot.run_research(question()))
        decision, reasoning = asyncio.run(bot.evidence_forecast(question(), research, 'main'))
        self.assertTrue(0.7 < decision.probability < 0.85)
        self.assertIn('Verifier issues: misread criteria', reasoning)
        decision, _ = asyncio.run(self.bot(verifier=StubLlm('garbage')).evidence_forecast(question(), research, 'main'))
        self.assertIsNone(decision.verifier_accepted)
        self.assertIn('verifier_error', self.journal()[-1])

    def test_research_without_fetcher_uses_template_path(self):
        self.assertEqual(asyncio.run(self.bot(news=None).run_research(question())), '')

    def test_throttle_caps_calls_in_flight(self):
        state = {'now': 0, 'peak': 0}

        async def direct_call(self, prompt):
            state['now'] += 1
            state['peak'] = max(state['peak'], state['now'])
            await asyncio.sleep(0.01)
            state['now'] -= 1
            return 'ok'

        async def burst():
            with patch.dict(os.environ, {'NORDROUTER_API_KEY': 'test-key', 'NORDROUTER_MAX_IN_FLIGHT': '2'}):
                llm = nordrouter_llm()
            return await asyncio.gather(*(llm._mockable_direct_call_to_model('p') for _ in range(10)))

        with patch.object(GeneralLlm, '_mockable_direct_call_to_model', direct_call):
            self.assertEqual(asyncio.run(burst()), ['ok']*10)
        self.assertEqual(state['peak'], 2)

    def test_nordrouter_wiring_and_missing_asknews(self):
        with patch.dict(os.environ, {'NORDROUTER_API_KEY': 'test-key', 'NORDROUTER_MODEL': ''}, clear=False):
            llm = nordrouter_llm({'type': 'object'}, 'analyst', 0)
        self.assertEqual(llm.model, 'openai/qwen/qwen3.8-max')
        self.assertEqual(llm.litellm_kwargs['base_url'], 'https://europe-api.nordrouter.com/v1')
        self.assertEqual(llm.litellm_kwargs['response_format']['json_schema']['name'], 'analyst')
        self.assertEqual(llm.litellm_kwargs['extra_body'], {'reasoning_effort': 'none'})
        with patch.dict(os.environ, {'ASKNEWS_API_KEY': 'REPLACE_ME', 'ASKNEWS_CLIENT_ID': '', 'ASKNEWS_SECRET': ''}):
            self.assertIsNone(asknews_fetcher())


if __name__ == '__main__':
    unittest.main()
