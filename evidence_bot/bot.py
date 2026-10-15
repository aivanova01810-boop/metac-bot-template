"""Template subclass: evidence-ledger research, cited analyst, verifier, journal and shadow variants.

Only binary questions use the evidence pipeline; other types keep the template prompts.
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Awaitable, Callable

from forecasting_tools import BinaryQuestion, GeneralLlm, MetaculusQuestion, ReasonedPrediction

from main import FallTemplateBot2026

from . import ledger
from .analyst import (ANALYST_SCHEMA, VERIFIER_SCHEMA, ContractError, ShrinkPolicy, analyst_prompt,
                      apply_verdict, check_proposal, decide, parse_json, question_block,
                      render_reasoning, verifier_prompt)
from .journal import Journal

logger = logging.getLogger(__name__)

NORDROUTER_BASE_URL = 'https://europe-api.nordrouter.com/v1'
DEFAULT_MODEL = 'qwen/qwen3.8-max'
NewsFetcher = Callable[[MetaculusQuestion], Awaitable[list[dict]]]


class ThrottledLlm(GeneralLlm):
    """NordRouter rejects calls beyond a small per-account in-flight limit (HTTP 429).

    The template runs every question and all five samples concurrently, so each direct
    call takes a slot; the retry wait sits outside the slot and does not hold it.
    """
    _limits: dict = {}

    async def _mockable_direct_call_to_model(self, prompt):
        # One semaphore per event loop: evidence_main runs asyncio.run once per tournament.
        loop = asyncio.get_running_loop()
        limit = int(os.environ.get('NORDROUTER_MAX_IN_FLIGHT') or 2)
        semaphore = ThrottledLlm._limits.setdefault(loop, asyncio.Semaphore(limit))
        async with semaphore:
            return await super()._mockable_direct_call_to_model(prompt)


def nordrouter_llm(schema: dict | None = None, name: str = 'forecast', temperature: float = 0.3,
                   model: str | None = None) -> GeneralLlm:
    """A model behind NordRouter's OpenAI-compatible endpoint; a schema forces strict JSON output."""
    # Qwen reasons for 2-3k hidden tokens by default (60-160 s per call, past the timeout);
    # NordRouter honours reasoning_effort=none but ignores reasoning.enabled=false.
    extra = {'extra_body': {'reasoning_effort': os.environ.get('NORDROUTER_REASONING_EFFORT') or 'none'}}
    if schema is not None:
        extra['response_format'] = {'type': 'json_schema',
                                    'json_schema': {'name': name, 'strict': True, 'schema': schema}}
    # Empty CI variables arrive as '' rather than unset, so fall back on falsy values too.
    model = model or os.environ.get('NORDROUTER_MODEL') or DEFAULT_MODEL
    return ThrottledLlm(model='openai/' + model,
                        base_url=os.environ.get('NORDROUTER_BASE_URL') or NORDROUTER_BASE_URL,
                        api_key=os.environ['NORDROUTER_API_KEY'],
                        temperature=temperature, timeout=120, allowed_tries=4, **extra)


def identity(question: MetaculusQuestion) -> dict:
    return {'question_id': getattr(question, 'id_of_question', None), 'post_id': getattr(question, 'id_of_post', None),
            'url': getattr(question, 'page_url', None), 'text': getattr(question, 'question_text', None)}


class EvidenceAnalystBot(FallTemplateBot2026):
    def __init__(self, *args, analyst_llm=None, verifier_llm=None, news_fetcher: NewsFetcher | None = None,
                 journal_folder: str | None = 'journal', shadow_variants: tuple[str, ...] = ('template', 'no_research'),
                 shrink_policy: ShrinkPolicy = ShrinkPolicy(), analyst_attempts: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        # Injected rather than read from llms= so tests can pass stubs with an async invoke().
        self.analyst_llm = analyst_llm
        self.verifier_llm = verifier_llm
        self.news_fetcher = news_fetcher
        self.journal = Journal(journal_folder)
        self.shadow_variants = tuple(shadow_variants)
        self.shrink_policy = shrink_policy
        self.analyst_attempts = analyst_attempts
        self._shadowed: set = set()

    ##################################### RESEARCH #####################################

    async def run_research(self, question: MetaculusQuestion) -> str:
        if self.news_fetcher is None:
            research = await super().run_research(question)
            self.journal.write('research', **identity(question), source='template', research=research)
            return research
        try:
            async with self._concurrency_limiter:
                articles = await self.news_fetcher(question)
        except Exception as exc:
            # A news outage must not cost the question: forecast on an empty ledger instead.
            logger.warning('News fetch failed for %s: %s', getattr(question, 'page_url', None), exc)
            self.journal.write('research_failed', **identity(question), error=repr(exc))
            return ''
        items = ledger.build(articles)
        self.journal.write('research', **identity(question), source='news_fetcher',
                           ledger=[asdict(e) for e in items], fetched=len(articles))
        return ledger.render(items)

    ##################################### BINARY QUESTIONS #####################################

    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        key = (getattr(question, 'id_of_question', None), getattr(question, 'page_url', None))
        # predictions_per_research_report calls this several times; shadows run once per question.
        # The check-and-add has no await in between, so concurrent calls cannot both claim it.
        if self.shadow_variants and key not in self._shadowed:
            self._shadowed.add(key)
            await self._run_shadows(question, research)
        decision, reasoning = await self.evidence_forecast(question, research, 'main')
        return ReasonedPrediction(prediction_value=decision.probability, reasoning=reasoning)

    async def evidence_forecast(self, question: BinaryQuestion, research: str, variant: str):
        if self.analyst_llm is None:
            raise RuntimeError('EvidenceAnalystBot needs analyst_llm')
        as_of = datetime.now(timezone.utc)
        items = ledger.parse(research)
        block = question_block(question.question_text, question.background_info, question.resolution_criteria,
                               question.fine_print, as_of.strftime('%Y-%m-%d'))
        prompt = analyst_prompt(block, research)
        # The community prediction is logged for evaluation only; the analyst never sees it.
        record = {**identity(question), 'variant': variant, 'as_of': as_of, 'analyst_prompt': prompt, 'analyst_raw': [],
                  'community_prediction': getattr(question, 'community_prediction_at_access_time', None)}
        checked = None
        for attempt in range(self.analyst_attempts):
            raw = await self.analyst_llm.invoke(prompt)
            record['analyst_raw'].append(raw)
            try:
                checked = check_proposal(parse_json(raw), items, as_of)
                break
            except ContractError as exc:
                record.setdefault('analyst_errors', []).append(str(exc))
        if checked is None:
            self.journal.write('forecast_failed', **record)
            raise ContractError(f"Analyst output broke the contract {self.analyst_attempts} times")
        accepted, issues = None, []
        if self.verifier_llm is not None:
            claims = [{k: c[k] for k in ('claim', 'direction', 'evidence_ids')} for c in checked['claims']]
            vprompt = verifier_prompt(block, ledger.render(items), {'claims': claims})
            vraw = await self.verifier_llm.invoke(vprompt)
            record |= {'verifier_prompt': vprompt, 'verifier_raw': vraw}
            try:
                accepted, issues = apply_verdict(checked, parse_json(vraw))
            except ContractError as exc:
                # A broken verifier means no verdict, not a rejection.
                record['verifier_error'] = str(exc)
        decision = decide(checked, accepted, issues, self.shrink_policy, evidence_available=bool(items))
        self.journal.write('forecast', **record, checked=checked, decision=decision.as_dict())
        return decision, render_reasoning(checked, decision, items)

    async def _run_shadows(self, question: BinaryQuestion, research: str) -> None:
        """Logged, never submitted: they measure what research and verification add."""
        for variant in self.shadow_variants:
            try:
                if variant == 'template':
                    result = await FallTemplateBot2026._run_forecast_on_binary(self, question, research)
                    self.journal.write('shadow', **identity(question), variant=variant,
                                       probability=result.prediction_value, reasoning=result.reasoning)
                elif variant == 'no_research':
                    decision, _ = await self.evidence_forecast(question, '', variant)
                    self.journal.write('shadow', **identity(question), variant=variant,
                                       probability=decision.probability)
                else:
                    raise ValueError(f'Unknown shadow variant {variant}')
            except Exception as exc:  # A failing shadow must never block the submitted forecast.
                logger.warning('Shadow %s failed for %s: %s', variant, getattr(question, 'page_url', None), exc)
                self.journal.write('shadow_failed', **identity(question), variant=variant, error=repr(exc))


def build_llms() -> dict:
    """Default wiring: free-text Qwen for template prompts and parsing, JSON Qwen for the pipeline.

    NORDROUTER_VERIFIER_MODEL puts the verifier on another model family, so it does not
    share the analyst's blind spots; until it is set the verifier is Qwen at temperature 0.
    """
    text = nordrouter_llm()
    # The template parses each answer twice and drops the sample when the parses differ,
    # so the parser runs at temperature 0.
    parser = nordrouter_llm(temperature=0)
    return {'llms': {'default': text, 'parser': parser, 'summarizer': text, 'researcher': 'no_research'},
            'analyst_llm': nordrouter_llm(ANALYST_SCHEMA, 'analyst'),
            'verifier_llm': nordrouter_llm(VERIFIER_SCHEMA, 'verifier', temperature=0,
                                           model=os.environ.get('NORDROUTER_VERIFIER_MODEL'))}
