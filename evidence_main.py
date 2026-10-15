"""Run the evidence analyst bot. Dry run by default; --publish submits forecasts and comments."""
import argparse
import asyncio
import logging
import os
import sys

import dotenv

from bot_helpers import print_run_summary_banner, silence_noisy_dependencies

silence_noisy_dependencies()


from evidence_bot.bot import EvidenceAnalystBot, build_llms  # noqa: E402
from evidence_bot.news import asknews_fetcher  # noqa: E402

dotenv.load_dotenv()

# Explicit slugs, not MetaculusClient.CURRENT_*: the template's poetry.lock pins
# forecasting-tools 0.2.92, whose "current" tournament is the closed Summer one (33022).
# Update the slug when the season changes (Fall 2026 closes 2027-03-05).
SEASON = os.getenv('METACULUS_SEASON_SLUG') or 'fall-futureeval-2026'
TARGETS = {'tournament': [SEASON, 'minibench'], 'minibench': ['minibench'], 'test_questions': ['bot-testing-area']}

TOURNAMENT_URLS = {
    'tournament': 'https://www.metaculus.com/tournament/fall-futureeval-2026/',
    'minibench': 'https://www.metaculus.com/aib/minibench/',
    'test_questions': 'https://www.metaculus.com/tournament/bot-testing-area/',
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=list(TOURNAMENT_URLS), default='test_questions')
    parser.add_argument('--publish', action='store_true', help='Submit forecasts; without it nothing is posted')
    args = parser.parse_args()
    missing = [k for k in ('METACULUS_TOKEN', 'NORDROUTER_API_KEY') if not os.getenv(k) or os.getenv(k) == 'REPLACE_ME']
    if missing:
        print('Missing environment variables: ' + ', '.join(missing))
        return 1

    news = asknews_fetcher()
    if news is None:
        print('AskNews credentials not set: forecasting without retrieved evidence.')
    bot = EvidenceAnalystBot(
        research_reports_per_question=1,
        predictions_per_research_report=5,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=args.publish,
        folder_to_save_reports_to='reports',
        skip_previously_forecasted_questions=args.mode != 'test_questions',
        extra_metadata_in_explanation=True,
        news_fetcher=news,
        journal_folder='journal',
        **build_llms(),
    )
    targets = TARGETS[args.mode]
    reports = []
    for target in targets:
        reports += asyncio.run(bot.forecast_on_tournament(target, return_exceptions=True))
    bot.log_report_summary(reports)
    print_run_summary_banner(reports, will_publish=args.publish, tournament_url=TOURNAMENT_URLS[args.mode])
    return 0


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    sys.exit(main())
