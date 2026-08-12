> **⚠️ This is a plan, not a report.** The study it describes was never completed and produced no
> causal estimate. Every method, case study and specification below is proposed work. See
> [README.md](README.md) for what was actually built.

Below is a concrete, research‑ready blueprint for studying the impact of **Slack adoption** on public‑company performance without true A/B tests. I’ve grounded the plan in current data sources and modern causal‑inference methods, and I call out limitations and robustness checks along the way.

---

## 1) Core idea & unit of analysis

* **Unit:** public companies (e.g., Russell 3000), observed quarterly or annually, 2016–2025.
* **Treatment:** adoption of Slack (date + intensity).
* **Outcomes:** financial operating performance, stock returns, innovation output, employee sentiment, hiring dynamics.

Why now: we can triangulate adoption dates from job‑postings “skills” data, SEC full‑text mentions, press releases/case studies, and community tech‑stack listings, and then apply staggered difference‑in‑differences (DiD) estimators that are robust to staggered timing. ([docs.lightcast.dev][1], [kb.lightcast.io][2], [SEC][3], [The Verge][4], [Okta][5])

---

## 2) Measuring “Slack adoption” (multi‑signal)

Relying on a single technographics source (e.g., website scanners) is risky for Slack because it’s not primarily a website technology. We’ll combine multiple, complementary signals:

**A. Job‑postings skill signal (primary)**

* From a labor‑market data provider (e.g., Lightcast), count firm‑month (or firm‑quarter) postings that **mention “Slack”** in skills/requirements. Define adoption when Slack appears **persistently** (e.g., ≥3 unique postings and ≥1% of that firm’s postings over a rolling 6‑month window). Track **intensity** as the share of postings mentioning Slack. (Also collect “Microsoft Teams” mentions to study substitution.) ([docs.lightcast.dev][1], [kb.lightcast.io][2])

**B. SEC full‑text mentions (validation)**

* Use the SEC’s **Full‑Text Search** to detect first mentions of “Slack” (or “Slack Technologies,” “Slack channel,” “Slack integration”) in 10‑K/10‑Q/8‑K (e.g., IT/security sections, risk factors, MD\&A). Time‑stamp the earliest credible mention for cross‑validation. ([SEC][3])

**C. Press/case studies (event anchors)**

* For large, public adoptions with dated announcements (e.g., **IBM rolling out Slack to \~350k employees in Feb 2020**), use these as clean “event dates” for case‑level analyses. ([The Verge][4])

**D. Community tech‑stack listings (auxiliary)**

* **StackShare** has company tech‑stack pages that often list Slack; use only as a supporting signal (coverage is community‑contributed and biased to tech firms). ([StackShare][6])

**E. Caution on website trackers**

* Tools like **BuiltWith/Wappalyzer** detect technologies on **websites**. Their “Slack” detections primarily reflect widgets/buttons (e.g., “Add to Slack”) and won’t reliably indicate internal, firm‑wide Slack usage; use only to enrich, not to define adoption. ([BuiltWith][7])

**Adoption date rule (ensemble):** set the adoption date when **(A)** triggers and is corroborated by **(B)** or **(C)** within ±6 months. Where corroboration is missing, require higher persistence thresholds in (A).

---

## 3) Outcomes (dependent variables)

**Financial & operating (Compustat):**

* Productivity: **Revenue per employee** (SALE/EMP), operating margin (OIADP/SALE), gross margin (GP/SALE), SG\&A efficiency (1 − SG\&A/SALE), revenue growth (ΔSALE). ([S\&P Global][8], [Center for Research in Security Prices][9])

**Market response (CRSP):**

* Short‑window **event‑study CARs** around high‑confidence adoption announcements; longer‑horizon abnormal returns for DiD complements. ([Center for Research in Security Prices][10])

**Innovation output (USPTO/PatentsView):**

* **Patents granted** per year; **patents per R\&D dollar** (XRD). Roll to firm‑year and normalize by firm size. ([USPTO][11])

**Employee sentiment (optional):**

* **Glassdoor/Comparably** ratings (availability/license permitting) to proxy culture/satisfaction changes pre/post adoption. Use cautiously and document coverage. ([Glassdoor][12], [Comparably][13])

**Hiring/mix (Lightcast):**

* Total postings, **remote/hybrid share**, and functional mix changes (e.g., more roles with “platform integrations,” “automation,” “DevOps”). ([docs.lightcast.dev][14], [kb.lightcast.io][2])

**Context controls:**

* **Teams adoption pressure:** use “Teams” mentions in postings; exploit **2023–2024 EU investigation and 2024 global unbundling** as exogenous shifts in Teams’ distribution. ([European Commission][15], [Microsoft][16])
* **Remote‑work intensity:** include Dingel–Neiman teleworkability indexes and contemporaneous remote‑posting shares by industry/region. ([NBER][17])

---

## 4) Identification strategy

**A) Staggered DiD with event‑time (baseline)**

* Model firm outcomes $Y_{it}$ on **relative time to adoption** (leads/lags), with firm fixed effects and calendar‑time (or industry×time) fixed effects. Use **modern estimators** robust to staggered timing and heterogeneity (e.g., **Sun–Abraham**, **Callaway–Sant’Anna**, **Borusyak–Jaravel–Spiess imputation**). Report event‑time plots and **pre‑trend** tests. ([ScienceDirect][18], [Oxford Academic][19])

**B) Synthetic control (targeted cases)**

* For mega‑adopters with crisp dates (e.g., IBM), build a synthetic control from non‑adopters (matched on pre‑trends and covariates) to estimate post‑adoption gaps in productivity and margins. ([MIT Economics][20])

**C) IV / quasi‑experiments (optional, for endogeneity)**

* **Salesforce intensity × post‑acquisition:** Slack was acquired by Salesforce on **July 21, 2021**; firms with high pre‑2021 Salesforce reliance (measured in postings/SEC text) plausibly faced higher Slack adoption propensity thereafter. Use this as an instrument or a difference‑in‑difference‑in‑differences (DDD) moderator. ([Salesforce][21])
* **Teams bundling/unbundling shock:** Microsoft **unbundled Teams globally on Apr 1, 2024** after EU scrutiny. Firms/industries more tied to Office 365 pre‑2024 faced exogenous changes in Teams economics, which may shift relative Slack adoption—use in a DDD comparing high vs. low Office exposure pre/post 2024. ([European Commission][15], [Microsoft][16])

**D) Matching / weighting**

* Build a **matched control** set (industry, size, growth, margins, leverage, R\&D, remote‑workability), or apply **entropy balancing** to achieve covariate balance before DiD. ([Massachusetts Institute of Technology][22])

**E) Placebos & falsification**

* Placebo adoption dates for never‑adopters; “fake” outcomes (e.g., lagged pre‑treatment). Run **Teams‑only** analyses to verify that results are Slack‑specific, not generic to collaboration tooling.

---

## 5) Covariates & controls (pre‑specified)

* Firm FE; **calendar time FE** (or industry×time).
* Industry, size (log assets), cash, leverage, capex, R\&D intensity, prior growth and margins.
* **Remote‑workability** by occupation mix; **remote posting share**. ([NBER][17], [docs.lightcast.dev][14])
* **Teams intensity** (to capture substitution effects).

---

## 6) Data engineering plan (reproducible)

**Linking & keys**

* Use **CRSP/Compustat Merged (CCM)** to link tickers/PERMNOs to **gvkey**; standardize employer names for job‑postings links. ([Center for Research in Security Prices][9])

**Pipelines**

1. **Firm universe:** Russell 3000 tickers → CCM link → Compustat quarterly panel. ([S\&P Global][8])
2. **Adoption signals:**

   * Lightcast postings (firm‑normalized) → monthly Slack/Teams counts; remote/hybrid flags. ([docs.lightcast.dev][14])
   * **SEC full‑text** “Slack” mentions (company‑date). ([SEC][3])
   * Press/case events (e.g., IBM 2020). ([The Verge][4])
3. **Outcomes:** Compustat (SALE, EMP, GP, OIADP, SG\&A, XRD), CRSP returns, PatentsView patents. ([S\&P Global][8], [Center for Research in Security Prices][10], [USPTO][11])
4. **Join & QC:** roll to quarterly/annual; winsorize outliers; validate adoption date consistency across signals.

**Data dictionary (illustrative)**

* `firm_id (gvkey)`, `permno`, `fyearq/fqtr`, `adopt_date`, `slack_postings_share`, `teams_postings_share`, `remote_postings_share`, `SALE`, `EMP`, `SGA`, `GP`, `OIADP`, `XRD`, `patents_cnt`, `industry`, `assets`, `leverage`, `cash`, `capex`.

---

## 7) Estimation details

* **Baseline:** Event‑study DiD with Sun–Abraham or Callaway–Sant’Anna estimators; report event‑time coefficients $\beta_k$ for $k=-8..+12$ quarters relative to adoption; cluster SEs **two‑way** (firm and time). ([ScienceDirect][18])
* **Heterogeneity:** Interact post‑adoption with pre‑adoption Salesforce intensity, remote‑workability, or industry digitization to assess where Slack has the largest effects. ([NBER][17])
* **Synthetic control:** Pre‑adoption window ≥8–12 quarters; validate fit (RMSPE) and run permutation tests. ([MIT Economics][20])
* **Event study (stocks):** ±3/±5/±10‑day windows around dated announcements (only where (C) provides clean dates). ([Center for Research in Security Prices][10])

---

## 8) Threats to validity & how we address them

* **Measurement error in adoption:** mitigated via multi‑signal rule and persistence thresholds; sensitivity to alternate thresholds. (A/B/E above.) ([docs.lightcast.dev][1], [SEC][3], [BuiltWith][23])
* **Concurrent shocks (e.g., pandemic, Teams bundling):** include time fixed effects, explicit Teams intensity measures, and leverage 2024 unbundling as a quasi‑experimental moderator. ([Microsoft][16])
* **Selection bias:** matching/entropy balancing on rich pre‑trends and covariates; demonstrate parallel pre‑trends graphically and via statistical tests. ([Massachusetts Institute of Technology][22])
* **Overstated effects from vanilla TWFE:** use modern staggered‑DiD estimators and Goodman‑Bacon decomposition diagnostics to understand weights. ([ScienceDirect][24])

---

## 9) Pilot analyses to de‑risk

1. **IBM case study (synthetic control):** quantify post‑Feb‑2020 shifts in revenue/employee, SG\&A ratio, and margins vs. a synthetic peer basket. (Event date from coverage of IBM’s rollout.) ([The Verge][4])
2. **Panel prototype (50–100 firms):** pick sectors with high collaborative‑tool penetration; build initial adoption dates from job‑postings + SEC mentions; run **Sun–Abraham** event‑study for revenue/employee and SG\&A%. ([docs.lightcast.dev][1], [SEC][3])

---

## 10) Optional extensions

* **Industry‑time shift‑share:** use **Okta Businesses at Work** industry‑level Slack vs. Teams trends to instrument industry‑time adoption pressure (aggregate measure only, not firm‑level). ([Okta][5])
* **Innovation lag structure:** link patent grants to application years to allow for multi‑year lags post adoption (e.g., +2 to +4 years). ([USPTO][11])
* **Employee sentiment:** incorporate Glassdoor/Comparably ratings (licensing permitting) for turnover/satisfaction proxies. ([Glassdoor][12], [Comparably][13])

---

## 11) What this study can credibly answer

* **Average causal effect** of Slack adoption on (i) revenue per employee, (ii) SG\&A efficiency, (iii) operating margins, and (iv) near‑term stock reactions—conditional on parallel‑trends tests and robustness checks passing.
* **Where Slack seems to help most,** via heterogeneity (e.g., high‑remote, Salesforce‑heavy, or high‑knowledge‑work industries).

---

## 12) Practical notes & constraints

* **Data access:** Compustat/CRSP via WRDS (or S\&P/LSEG channels). Lightcast for postings/skills. Patents via **PatentsView** (free). SEC full‑text search is public. ([S\&P Global][8], [Center for Research in Security Prices][10], [USPTO][11], [SEC][3])
* **Compliance:** If scraping any ratings/job boards, ensure terms of service are respected; prefer official APIs/datasets.

---

### References (selected)

Methods: Sun & Abraham (2021); Callaway & Sant’Anna (2021); Borusyak, Jaravel & Spiess (2024/2021); Goodman‑Bacon (2021). ([ScienceDirect][18], [Oxford Academic][19], [arXiv][25])
Context: Okta **Businesses at Work 2024**; IBM–Slack rollout (2020); Salesforce–Slack close (2021); Microsoft **Teams unbundling** (2024). ([Okta][5], [The Verge][4], [Salesforce][21], [Microsoft][16])
Data: Compustat/CRSP/CCM; Lightcast postings; PatentsView; SEC full‑text. ([S\&P Global][8], [Center for Research in Security Prices][10], [docs.lightcast.dev][14], [USPTO][11], [SEC][3])

---

## Recommended next move

If you’d like, I can draft the **variable schema and code scaffolding** (Python + SQL) to: (a) ingest postings counts for “Slack/Teams,” (b) extract SEC “Slack” mentions, (c) build the adoption date/intensity features, and (d) run a Sun–Abraham event‑study on Compustat outcomes—so you have an immediately runnable pilot.

[1]: https://docs.lightcast.dev/datasets/us-job-postings?utm_source=chatgpt.com "US Job Postings - docs.lightcast.dev"
[2]: https://kb.lightcast.io/en/articles/6957446-job-posting-analytics-jpa-methodology?utm_source=chatgpt.com "Job Posting Analytics (JPA) Methodology - Lightcast"
[3]: https://www.sec.gov/search-filings?utm_source=chatgpt.com "Search Filings - SEC.gov"
[4]: https://www.theverge.com/2020/2/10/21132060/ibm-slack-chat-employee-rollout-microsoft-teams-competition?utm_source=chatgpt.com "IBM picks Slack over Microsoft Teams for its 350,000 employees"
[5]: https://www.okta.com/sites/default/files/2024-04/Okta-2024_Businesses_at_Work.pdf?utm_source=chatgpt.com "2024 Businesses at Work - Okta"
[6]: https://stackshare.io/slack?utm_source=chatgpt.com "Slack - Reviews, Pros & Cons | Companies using Slack - StackShare"
[7]: https://trends.builtwith.com/widgets/Slack?utm_source=chatgpt.com "Slack Usage Statistics - BuiltWith"
[8]: https://www.spglobal.com/market-intelligence/en/solutions/products/fundamental-data?utm_source=chatgpt.com "Fundamental Financial Data - S&P Global"
[9]: https://www.crsp.org/wp-content/uploads/guides/CRSP_Compustat_Merged_Database_Guide.pdf?utm_source=chatgpt.com "CRSP/COMPUSTAT MERGED DATABASE GUIDE"
[10]: https://www.crsp.org/research/crsp-us-stock-databases/?utm_source=chatgpt.com "CRSP US Stock Databases - Center for Research in Security Prices"
[11]: https://www.uspto.gov/ip-policy/economic-research/patentsview?utm_source=chatgpt.com "PatentsView | USPTO"
[12]: https://www.glassdoor.com/developer/companiesApiActions.htm?utm_source=chatgpt.com "Glassdoor Company API Documentation"
[13]: https://www.comparably.com/api?utm_source=chatgpt.com "Culture API Partnership Program | Comparably"
[14]: https://docs.lightcast.dev/apis/job-postings?utm_source=chatgpt.com "Job Postings - docs.lightcast.dev"
[15]: https://ec.europa.eu/commission/presscorner/api/files/document/print/en/ip_23_3991/IP_23_3991_EN.pdf?utm_source=chatgpt.com "European Commission - Press release - Die Europäische Kommission"
[16]: https://www.microsoft.com/en-us/licensing/news/Microsoft365-Teams-WW?utm_source=chatgpt.com "Realigning global licensing for Microsoft 365"
[17]: https://www.nber.org/papers/w26948?utm_source=chatgpt.com "How Many Jobs Can be Done at Home? | NBER"
[18]: https://www.sciencedirect.com/science/article/pii/S0304405X22000204?utm_source=chatgpt.com "How much should we trust staggered difference-in-differences estimates ..."
[19]: https://academic.oup.com/restud/article/91/6/3253/7601390?utm_source=chatgpt.com "Revisiting Event-Study Designs: Robust and Efficient Estimation"
[20]: https://economics.mit.edu/sites/default/files/publications/Synthetic%20Control%20Methods.pdf?utm_source=chatgpt.com "Synthetic Control Methods for Comparative Case Studies: Estimating the ..."
[21]: https://www.salesforce.com/news/press-releases/2021/07/21/salesforce-slack-deal-close/?utm_source=chatgpt.com "Salesforce Completes Acquisition of Slack"
[22]: https://www.mit.edu/~jhainm/Paper/eb.pdf?utm_source=chatgpt.com "Entropy Balancing for Causal Effects: A Multivariate Reweighting ... - MIT"
[23]: https://trends.builtwith.com/websitelist/Slack-App?utm_source=chatgpt.com "Websites using Slack App - BuiltWith"
[24]: https://www.sciencedirect.com/science/article/pii/S0304407621001445?utm_source=chatgpt.com "Difference-in-differences with variation in treatment timing"
[25]: https://arxiv.org/pdf/2108.12419?utm_source=chatgpt.com "Revisiting Event Study Designs: Robust and Eficient Estimation"
