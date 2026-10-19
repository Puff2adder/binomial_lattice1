# Binomial Option Learning Studio v01

An ungraded Streamlit laboratory for Executive MBA, MBA, and senior finance students. No corporate-finance applications. The current release is a working teaching draft; textbook and assessed homework are not yet written.

## Pages

- Start here: objectives, route, notation, assumptions, homework use.
- One period: calls left / puts right, continuous dividends, stock and borrowing replication, state-by-state checks, pricing weights, European parity.
- Two periods: European/American trees, backward reveal, node equations and substitutions, optional replication.
- Three periods: European calls/puts followed by American calls/puts; exercise decisions and indifference are distinct.
- Optional eight-period probabilities/returns: distributions, full return derivation, selectable node, expected one-step and annualized growth measures, leverage and moneyness exploration.
- Convergence: both exercise styles, calls/puts, European Black-Scholes benchmark, American fine-tree stability check.
- Bermudan: selectable finite dates, date timeline, comparison with European/American contracts on the same tree.
- Optional barrier: discretely monitored European up-and-out call, no rebate, path tracking and monitoring limitations.
- Practice: 14 tagged conceptual questions with hints, distractor explanations, and retry.

Every core page permits exact numeric entry, reset, staged hints, requested solutions, node inspection, and export. Dividends and volatility use sliders. Formula panels remain accessible without correctness gates. No student identities, grades, or instructor-only assessed solutions are included.

## Local launch

Python 3.11 or newer is recommended. In this folder:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

The launcher uses the local .venv if present, otherwise the default Python. On this workstation the existing Anaconda environment has the pinned dependencies: `C:\ProgramData\anaconda3\python.exe -m streamlit run app.py`.

Run independent financial checks with `python -m unittest discover -s tests -v`.

## Deployment preparation

Only publish this allowlist: app.py, finance.py, calculator.py, questions.py, requirements.txt, and .streamlit/config.toml. Tests and this README may be included in the code repository. Never publish the parent textbook workspace or homework/instructor_solutions. No public deployment has been performed.

For Streamlit Community Cloud, put the allowlisted files in a dedicated repository, choose app.py as the entry point, and use Python 3.11+ with requirements.txt. Confirm available Python/package versions during deployment. Validate the live app and direct routes before giving students the URL. In Canvas, create an external URL item to the verified app, opening a new tab if embedding is restricted. A route such as `?page=one` opens the one-period page in a fresh session. Hosting sleep and classroom concurrency still require live validation.

## Financial conventions

- Continuous compounding; maturity in years; prices per share; quantities and multipliers applied externally.
- H shares and B dollars borrowed: V=HS-B. Dividends reinvest over a step, so next holdings are H exp(q dt).
- p denotes pricing probability. Actual probability is assumed through expected price growth mu, not estimated by this model.
- mu_C = log(E[C_next]/C)/dt is annualized log expected growth; it is not expected log return. Percentage returns at zero-value nodes are undefined.
- American/Bermudan exercise compares immediate payoff and continuation. Equality is indifference.
- Barrier monitoring is at every tree date including time 0 and expiry. After knockout, the contract remains dead; displayed continuation values are conditional on prior survival.

## Source and limitations

Inspired by the instructor's Binomial.tex and PR_Binomial_Models.xlsm worksheet examples. The VBA has not been ported or audited; calculations are independently implemented and reconciled to the lecture's numerical examples. Exact contract definitions and hypotheses are shown on each page. Financial engine is separate from interface code. The calculator uses a bounded AST parser, with no unrestricted eval and no automatic dimensional analysis. Changing any symbol invalidates its saved result.

The lecture originals and homework originals are unchanged. All files in this package are development copies under working/studios. The studio is not connected to external market data. It is educational and does not offer individualized trading advice.
