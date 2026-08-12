# 1 - Initial analysis of existing questions

### The Systemic Issue: All 6 Use Comparative Framing

Every question says "more focused", "more connected", "faster than before", "higher quality", "easier." For a **quarterly** survey, this is the single biggest problem. The research is clear: **absolute framing** ("I feel focused at work") is preferred over comparative ("I feel MORE focused") for repeated measurement because:

- People are bad at remembering how they felt 3 months ago (recall bias)
- Comparative wording implies improvement is expected (leading)
- "More than last quarter" in Q2 means something different than "more than last quarter" in Q4 - scores aren't comparable over time
- Every validated workplace instrument (Gallup Q12, UWES, Edmondson) uses absolute framing

You track change by comparing Q1 scores to Q2 scores statistically, not by asking people to estimate their own change.

---

### Question-by-Question

**Q1: "I feel more focused (using Convictional over Slack)"** - Strong construct, weak execution. Focus is real and measurable. But the parenthetical Slack comparison makes it partially double-barreled (measuring focus AND tool preference). What about users who never used Slack? Drop the comparison, reframe to absolute.

**Q2: "I feel more connected with my team"** - Good construct, vague wording. "Connected" has at least 4 distinct dimensions (membership, influence, need fulfillment, emotional connection). Decide's value is likely about being *informed* and *aligned*, not social bonding. Needs more specificity.

**Q3: "We are more likely to achieve our goals"** - **Weakest question.** Shifts from "I" to "We" (different measurement entirely). Asks for a prediction about future outcomes. Goal achievement depends on dozens of factors beyond any tool. Better to measure the *mechanism* (clarity, alignment, visibility into progress) than the *outcome*.

**Q4: "I'm able to make decisions faster than before"** - Decent. Decision speed is concrete. But speed isn't always good - fast decisions can be reckless. "Than before" requires recall. Could be reframed around having the right context to decide efficiently.

**Q5: "My decisions are higher quality"** - **Weakest self-report validity.** Dunning-Kruger research shows people are systematically poor judges of their own decision quality. "Quality" has no anchor - different people interpret it differently. Consider measuring the decision *process* (right info, right people) instead.

**Q6: "It's easier to find what I need when I need it"** - **Strongest question.** Concrete, behavioral, directly attributable to the tool, and users can genuinely assess it. Echoes Gallup Q12's validated item about having what you need. Just needs the comparative framing fixed.

---

### Other Cross-Cutting Issues

- **No reverse-coded items**: All 6 are positive. A yea-sayer who agrees with everything looks like a power user. Need 1-2 negatively framed items.
- **Inconsistent subjects**: "I feel" / "We are" / "It's" - the "We" in Q3 changes what's being measured entirely.
- **High face validity**: The survey's purpose is obvious, making it easy for respondents to give favorable answers.
- **Missing constructs**: No coverage of time savings, meeting effectiveness, async communication, transparency, or reduced context-switching.

# 2 - Reworking of existing questions

### Proposed Revisions (same order as originals)

**Q1 - Focus**
- Original: *"I feel more focused (using Convictional over Slack)"*
- Proposed: **"I am able to focus on my most important work without unnecessary distractions"**

Rationale: Absolute framing. Drops the Slack comparison entirely. "My most important work" ties to the clarity Decide provides via Goals. "Without unnecessary distractions" captures the async-first value (Decide vs. Slack's constant pings) without naming Slack. Users can genuinely self-assess this.

---

**Q2 - Connection**
- Original: *"I feel more connected with my team"*
- Proposed: **"I feel well-informed about my team's work, priorities, and decisions"**

Rationale: Sharpens "connected" to what Decide actually provides - *being informed*. Decide's connection value isn't social bonding (that's what happy hours are for), it's having visibility into what your team is doing and why. "Work, priorities, and decisions" maps directly to Decide's core content types. If scores rise over time, it means Decide is delivering on the "connected" promise in a way that's actually attributable to the tool.

---

**Q3 - Goal Achievement**
- Original: *"We are more likely to achieve our goals"*
- Proposed: **"I have clear visibility into our goals and how we are progressing toward them"**

Rationale: Biggest change. Switches from "We" to "I". Measures the *mechanism* (visibility, clarity) rather than the *outcome* (achievement), because goal achievement is influenced by dozens of factors outside any tool. This is directly tied to what Decide provides - the Goals feature with tracking, alignment scoring, and progress updates. A rising score here means Decide is doing its job. A rising score on the original could mean anything.

Trade-off worth noting: This shifts the original brief's intent from "do you believe you'll achieve goals" to "can you see where goals stand." I think that's a stronger question, but it's a meaningful departure from the original spirit.

---

**Q4 - Decision Speed**
- Original: *"I'm able to make decisions faster than before"*
- Proposed: **"When a decision needs to be made, I have the context I need to move forward efficiently"**

Rationale: Drops "than before." Reframes from pure speed (which can reward recklessness) to *having the context to be efficient* - which is what Decide actually provides via Posts, gathered discussions, and structured decision-making. "Efficiently" preserves the speed intent without making speed the sole virtue.

---

**Q5 - Decision Quality**
- Original: *"My decisions are higher quality"*
- Proposed: **"I feel confident that important decisions are made with sufficient information and input"**

Rationale: Most significant rework. The original asks people to judge their own decision quality, which is the self-report construct with the weakest validity (Dunning-Kruger). The revision sidesteps this by asking about the decision *process* - whether the right information and input are present. "I feel confident" keeps it as subjective experience (valid to self-report). This measures what Decide actually enables: gathering context, involving the right people, and making decisions transparently.

---

**Q6 - Information Access**
- Original: *"It's easier to find what I need when I need it"*
- Proposed: **"I can quickly find the information I need when I need it"**

Rationale: Lightest touch. Switches "It's easier" (comparative) to "I can quickly" (absolute). Switches from "It's" to "I" for consistent subject. Preserves everything that made this the strongest question in the original set.

---

### The Set as a Whole

All 6 now share:
- **Consistent "I" subject** (no more "We" or "It's")
- **Absolute framing** (trackable quarter-over-quarter)
- **Agreement scale**: All work with Strongly disagree → Strongly agree
- **Tied to what Decide actually provides** rather than broad outcomes

# 3 - Brainstorming gaps between existing questions and product offerings

The 6 revised questions collectively cover: focus, team awareness, goal visibility, decision efficiency, decision process quality, and information retrieval. In aggregate, they also implicitly address productivity ("get more done in less time") since that is the natural outcome when focus, goal clarity, decision speed, and information access all improve.

Three gaps remain where the existing questions do not capture something genuinely distinct.

---

### Gap 1: Knowledge Persistence

**What it is:** The confidence that important context from past discussions, decisions, and work is preserved and accessible over time - that organizational knowledge accumulates rather than disappearing.

**Why it's distinct from Q1-Q6:** Q6 measures whether you can *retrieve* information in the moment. Knowledge persistence is about whether that information *exists to be found* in the first place. You could have excellent search (Q6 high) over a shallow pool of recent content, or you could have a deep, growing organizational memory where decisions from 6 months ago are still documented with full rationale. These are different constructs - retrieval mechanics vs. durability of knowledge.

**Why it matters for Decide:** This is one of Decide's most concrete structural advantages. Decisions are formally recorded in Posts with the deciding comment marked. Events have audit trails with who, what, when. Content is indexed in persistent workspaces. In contrast, Slack conversations scroll away and critical context is effectively lost within weeks. Knowledge persistence also compounds over time - the value increases the longer an organization uses the tool, which makes it an especially valuable signal for a quarterly survey where you'd expect scores to rise as the knowledge base grows.

---

### Gap 2: Autonomy / Control Over Time and Attention

**What it is:** The feeling that you control when and how you engage with work communication, rather than being at the mercy of real-time demands on your attention.

**Why it's distinct from Q1-Q6:** Q1 measures focus *during* work - whether you can concentrate without distractions. Autonomy is about controlling *when* you shift between deep work and communication. You could be focused when working (Q1 high) and well-informed (Q2 high) but still feel like communication demands dictate your schedule - checking messages first thing, feeling pressure to respond quickly, having your day structured around others' timelines rather than your own. Published workplace-wellbeing survey instruments explicitly measure "schedule control" and "flexibility" as variables separate from productivity or wellbeing, confirming this is a recognized, distinct construct in organizational research.

**Why it matters for Decide:** This is the core promise of async-first communication. Decide's entire design philosophy is that people should engage with discussions, decisions, and updates on their own schedule rather than in real-time. Posts, email-as-workspace, and structured updates all enable this. If users don't feel more in control of their time, the fundamental thesis isn't landing.

---

### Gap 3: Collaboration Quality

**What it is:** The experience of actively working together with colleagues on shared work - whether joint effort is productive and produces better outcomes than working alone.

**Why it's distinct from Q1-Q6:** Q2 measures being *informed* about team work - this is relatively passive (receiving information, being in the loop). Q5 measures whether decisions have the right *inputs* - this is structural (the right information and people are present). Collaboration quality is about the *active experience* of co-creation: building on each other's ideas, working through problems together, the sense that collaboration makes the work better. You could be well-informed (Q2 high) and confident in decision inputs (Q5 high) but still find that actually working together is clunky, slow, or unproductive.

**Why it matters for Decide:** Decide provides real-time collaborative document editing, structured threaded discussions on Posts, collaborative email workspaces where teams can comment and assign together, and shared goal tracking with comments and updates. The quality of these collaborative interactions is central to whether the tool delivers value beyond just being an information hub.

# 4 - Proposed new questions and rationale

Three new questions to complement the 6 revised existing questions, each addressing a gap where the existing set does not capture something genuinely distinct.

---

**Q7 - Knowledge Persistence**

*"I can understand the context behind past work and discussions when I need to revisit them"*

Rationale: "I can understand" follows the capability framing used in Q6. "Context behind past work and discussions" is deliberately broad - covering decision rationale, meeting outcomes, project history, goal evolution, email threads, and the thinking behind how work progressed. "Past" introduces the temporal dimension that makes this distinct from Q6: Q6 measures whether you can find information *now*, this measures whether organizational knowledge *endures* over time. You could score Q6 high (great search) but Q7 low (you find the item but the surrounding context - the why, the who, the discussion that led there - is gone). This is one of Decide's most concrete structural advantages: Posts record decisions with context, events create audit trails, workspaces persist. In Slack, that context scrolls away within weeks.

---

**Q8 - Autonomy / Control Over Time**

*"I am able to engage with team discussions and updates on my own schedule"*

Rationale: "I am able to" matches Q1's capability framing. "Team discussions and updates" maps directly to Decide's core async content - Posts, Updates, email threads, goal comments. "On my own schedule" captures the autonomy construct concretely: do you choose when to engage, or does communication dictate your day? This is distinct from Q1: Q1 measures focus *during* work (not being distracted while doing something). Q8 measures control over *when* you shift attention to communication. You could be focused when working (Q1 high) but still feel compelled to check and respond constantly throughout the day. Published workplace-wellbeing survey instruments explicitly measure "schedule control" and "flexibility" as variables separate from productivity or focus, confirming this is a recognized, distinct construct. This question tests whether Decide's fundamental async-first thesis is landing.

---

**Q9 - Collaboration Quality**

*"I am able to collaborate effectively with my colleagues on shared work"*

Rationale: "I am able to" keeps the capability framing consistent across Q1, Q8, and Q9. "Collaborate effectively" captures the active co-creation experience - whether working together is smooth, productive, and not wasted effort. "With my colleagues" scopes to team interactions. "On shared work" ties it to actual work product (documents, decisions, goals, email threads) rather than just communication. This is distinct from Q2 (which measures passively *receiving* information about team work) and Q5 (which measures whether decisions have the right *structural inputs*). You could be well-informed (Q2 high) and confident decisions have the right inputs (Q5 high) but still find the actual process of working together - co-editing, discussing, building on each other's contributions - to be clunky or unproductive. Decide enables collaboration through real-time document editing, structured threaded discussions, collaborative email workspaces, and shared goal tracking.

# 5 - Likert scale analysis and assignment

**Scale types used:**

- **Frequency** (Never | Rarely | Sometimes | Often | Always): For capabilities that recur and vary in consistency. Frequency captures the success rate and reliability of the experience across instances. "Sometimes" is a concretely meaningful midpoint - it tells you "roughly half the time this is true." This is more informative than "Neither agree nor disagree" for questions about fluctuating capabilities.

- **Agreement** (Strongly disagree | Disagree | Neither agree nor disagree | Agree | Strongly agree): For persistent states and general assessments where the respondent is evaluating an overall condition, not reporting consistency across instances. This is the foundational Likert scale type, designed for statements about attitudes and beliefs.

---

**Q1 - Focus**: **Frequency**
*"I am able to focus on my most important work without unnecessary distractions"*
Focus fluctuates day to day. Some days you get deep work done, other days interruptions take over. Frequency captures this consistency: "Often" means most days you can focus, "Rarely" means most days you can't. A respondent who can focus 3 out of 5 days would select "Sometimes" - that's concrete, actionable data. With Agreement, that same respondent might select "Neither agree nor disagree," which is ambiguous.

**Q2 - Informed**: **Agreement**
*"I feel well-informed about my team's work, priorities, and decisions"*
Being informed is a persistent state - a general sense of whether you're in the loop - not a recurring event you succeed or fail at instance by instance. "I agree I feel well-informed" captures the overall assessment naturally. You don't have discrete "feeling informed" moments the way you have discrete decision moments or search attempts.

**Q3 - Goal visibility**: **Agreement**
*"I have clear visibility into our goals and how we are progressing toward them"*
Goal visibility is a persistent assessment of clarity. You either have a clear picture of where goals stand or you don't. This doesn't fluctuate instance by instance - it's an ongoing state shaped by the tools and processes in place. Agreement captures the strength of that assessment.

**Q4 - Decision efficiency**: **Frequency**
*"When a decision needs to be made, I have the context I need to move forward efficiently"*
Decision moments are discrete, recurring events. Each time a decision comes up, you either have the context or you don't. Frequency captures the hit rate: "Often" means most decision moments have sufficient context, "Rarely" means most don't. The question itself signals recurring instances with "When a decision needs to be made."

**Q5 - Decision process**: **Agreement**
*"I feel confident that important decisions are made with sufficient information and input"*
This is a belief about a process - an overall assessment of how decision-making works in your organization. It's not about specific decision instances (that's Q4) but about general confidence in the system. Agreement captures the strength of that belief: "I strongly agree" means deep confidence, "I disagree" means genuine doubt.

**Q6 - Information access**: **Frequency**
*"I can quickly find the information I need when I need it"*
Information retrieval is a repeated behavior that happens daily. Each search attempt succeeds or fails. Frequency tells you the reliability of the search experience: "Always" means every time you look for something, you find it. "Sometimes" means it's a coin flip. This is the success rate of a recurring action, which Frequency is designed for.

**Q7 - Knowledge persistence**: **Frequency**
*"I can understand the context behind past work and discussions when I need to revisit them"*
Revisiting past work is a discrete, recurring event. Each time you go back to something from weeks or months ago, the context is either preserved or it's gone. Frequency captures how reliably the knowledge persists: "Often" means most of the time the context is there, "Rarely" means it's usually lost. This directly measures the consistency of Decide's structural advantage over tools where information disappears.

**Q8 - Autonomy**: **Frequency**
*"I am able to engage with team discussions and updates on my own schedule"*
Autonomy over your schedule fluctuates. Some days you engage with discussions when you choose to; other days, urgent requests or meeting-heavy schedules take that control away. Frequency captures how consistently you experience this autonomy: "Always" means you fully control your engagement timing, "Sometimes" means it's roughly half and half.

**Q9 - Collaboration quality**: **Agreement**
*"I am able to collaborate effectively with my colleagues on shared work"*
This is a general capability assessment - an overall judgment about whether collaboration works well. While collaboration happens in specific instances, the question frames it as a persistent evaluation ("I am able to"), not a per-instance report. Agreement captures the strength of that assessment: "I strongly agree" means collaboration is consistently smooth and productive.

---

### Final question and scale assignment

| # | Construct | Question | Scale | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|---|---|
| Q1 | Focus | I am able to focus on my most important work without unnecessary distractions | Frequency | Never | Rarely | Sometimes | Often | Always |
| Q2 | Informed | I feel well-informed about my team's work, priorities, and decisions | Agreement | Strongly disagree | Disagree | Neither agree nor disagree | Agree | Strongly agree |
| Q3 | Goal visibility | I have clear visibility into our goals and how we are progressing toward them | Agreement | Strongly disagree | Disagree | Neither agree nor disagree | Agree | Strongly agree |
| Q4 | Decision efficiency | When a decision needs to be made, I have the context I need to move forward efficiently | Frequency | Never | Rarely | Sometimes | Often | Always |
| Q5 | Decision process | I feel confident that important decisions are made with sufficient information and input | Agreement | Strongly disagree | Disagree | Neither agree nor disagree | Agree | Strongly agree |
| Q6 | Information access | I can quickly find the information I need when I need it | Frequency | Never | Rarely | Sometimes | Often | Always |
| Q7 | Knowledge persistence | I can understand the context behind past work and discussions when I need to revisit them | Frequency | Never | Rarely | Sometimes | Often | Always |
| Q8 | Autonomy | I am able to engage with team discussions and updates on my own schedule | Frequency | Never | Rarely | Sometimes | Often | Always |
| Q9 | Collaboration quality | I am able to collaborate effectively with my colleagues on shared work | Agreement | Strongly disagree | Disagree | Neither agree nor disagree | Agree | Strongly agree |

# 6 - Addressing team feedback on initial draft of questions

**Feedback on Q4 - "move forward efficiently"**

A coworker noted that "efficiently" doesn't sound like how people talk about decisions - nobody says "we need to decide this efficiently." They emphasized that people think about decision *speed*, and asked for a more conversational synonym.

The original reason we avoided speed language was that the original question ("faster than before") rewarded recklessness and used comparative framing. But our revised Q4 already mitigates the recklessness concern through its structure: "I have the context I need to..." frames speed as the *outcome* of having context, not a goal in itself. The adverb just describes what happens when you're not blocked.

"Quickly" is the most natural replacement. It's conversational (people say "we need to decide this quickly"), it's absolute framing (not comparative like "faster"), and the context-first structure keeps it from implying fast-for-the-sake-of-fast.

**Change:** "move forward efficiently" → "move forward quickly"

Revised Q4: *"When a decision needs to be made, I have the context I need to move forward quickly"*

---

**Feedback on Q7 - Frequency vs Agreement**

A coworker felt Q7 (knowledge persistence) reads more like an Agreement question than a Frequency question. After consideration, we agree.

The other Frequency questions (Q1, Q4, Q6, Q8) all measure capabilities that genuinely fluctuate instance by instance - focus varies daily, each decision moment has different context needs, each search attempt independently succeeds or fails, autonomy shifts with schedule demands. Q7 is different: knowledge persistence is a structural property of the tools and processes in place. Either the organization's knowledge base preserves context well or it doesn't. It doesn't vary much from one retrieval to the next - making it closer to Q2 (being informed) and Q3 (goal visibility), both persistent states measured with Agreement.

**Change:** Q7 scale from Frequency to Agreement

# 7 - When to use various scale types

A question that came up during review: since the questions could plausibly be answered with either scale, what's the actual basis for choosing one over the other?

Both scales *work* for any statement-format question. The distinction isn't about correctness - it's about which scale produces more useful data. The practical test is: **what does the midpoint tell you?**

- Frequency midpoint ("Sometimes") = "this is true roughly half the time." Concrete, quantifiable.
- Agreement midpoint ("Neither agree nor disagree") = "I'm somewhere in the middle." Vaguer - could mean "it varies," "I'm not sure," or "I don't feel strongly either way."

**Use Frequency when "sometimes" tells you something meaningfully clearer than "neither agree nor disagree."** This happens when the question describes something with a success rate across instances - how often can you focus? how often do you have context when a decision comes up? how often can you find what you need? For these, "often" vs "sometimes" vs "rarely" is exactly the data you want.

**Default to Agreement when the two midpoints are roughly equivalent in usefulness.** This happens when the question is about a general belief or overall assessment rather than a per-instance hit rate - how informed do you feel? how clear is your goal visibility? how confident are you in the decision process? For these, the strength of the respondent's overall conviction is the data you want, and Agreement is the standard Likert scale.

**Worked examples:**

Q1 (Focus) - landed on Frequency: "I can sometimes focus without distractions" tells you something actionable - about half their days are disrupted. "I neither agree nor disagree that I can focus" is harder to interpret. What does that mean? They're unsure? It varies? They don't care? Frequency wins because focus is about consistency across days, and "sometimes" gives you clearer data.

Q2 (Informed) - landed on Agreement: "I sometimes feel well-informed" and "I neither agree nor disagree that I feel well-informed" are roughly equally informative. Being informed is a general sense, not something with a clear hit rate. The midpoint test doesn't produce a clear winner, so we default to Agreement as the simpler, standard choice.

**The pattern across all 9 questions:**

The Frequency questions (Q1, Q4, Q6, Q8) all share a trait: they describe things where the success rate across instances is the thing you actually care about. How often can you focus? How often do you have decision context? How often can you find things? How often do you control your schedule? For those, "often" vs "sometimes" vs "rarely" is exactly the data you want.

The Agreement questions (Q2, Q3, Q5, Q7, Q9) are more about how strongly you believe something is true overall. The per-instance hit rate isn't really the point - it's the general assessment.
