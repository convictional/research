> **What this is:** an unpublished **marketing blog draft**, not research. It was written to
> support a product position (an email-focused, Slack-less way of working) and it selects
> literature accordingly. It states its own bias below, which is why it is kept — but read it as
> advocacy. For actual research in this repository, see the other experiment directories.

## Draft

I've worked in tech since 2016, when I joined Shopify and was introduced to Slack. I quickly found myself in dozens of channels, some work-related, many not, and noticed my interruptions were going up. It was easier to connect, but I was putting in longer hours to find quiet time to focus.

Fast forward to 2021: I joined Convictional, where one of the draws was that we don't use Slack. We're async by default. Since then, I've done some of my best work.

Today my role (and knowledge work in general) is changing again through AI. As AI continues to improve (both in intelligence and UX), the human role in knowledge work will become much more focused on judgement, creativity and trust-building; easy, repetitive tasks get handed off to the AI leaving us to do more deciding, thinking and aligning of our teams.

With this shift in mind, we wanted to understand why we all feel we do our best work in Slack-less environments and how that will translate to our vision for a unified inbox. What does the research actually say about real-time communication and cognition?

Full transparency: We've just launched an email-focused product, so we're biased. But we wanted to show our work and explore these questions from first principles. Below is what we found.

## The Connection Paradox

Slack and similar platforms promise better collaboration and reduced friction. By many measures, they succeed. Slack's own data shows that 87% of users report improved communication. The platform sees 5 billion actions per week across 47 million daily active users. Internal email drops by 32%, meetings decline by 19%, and information flows more freely ([Slack, 2024](#slack-2024)).

But there's an offsetting cost. The same connectivity that makes collaboration easier also fragments our attention. The average Slack user is connected to their workspace for [9+ hours per day](https://slack.com/blog/news/work-is-fueled-by-true-engagement), taking approximately 83-92 actions daily ([DemandSage, 2025](#demandsage-2025))—sending messages, checking channels, searching conversations. Each action is a potential context switch, a moment where focus fractures.

This isn't about whether Slack "works." It clearly does, for certain roles and definitions of working. The question is: what are we trading for that connectivity in knowledge work? As AI overtakes more task-based work that might traditionally have happened in Slack, what remains is making decisions and being creative, where distractions cost us far more.

## The Science of Interruption

### What Is an Interruption?

An interruption is any environmental cue that causes you to shift focus from one task to another. They can be planned, like meetings, or unplanned, like notification pings. Some interruptions relate to your current task while others pull you into different contexts. They can come from external sources or be self-initiated, like checking Slack out of curiosity or boredom.

Interruptions aren't inherently bad. If customers don't interrupt you when something's wrong, they churn. Some roles (e.g. support, ER triage) are designed around interruptions. The question is whether knowledge work requires the same constant reactivity.

### The Cognitive Costs

A highly cited 2008 study by [Gloria Mark, Daniela Gudith, and Ulrich Klocke](https://interruptions.net/literature/Mark-CHI08.pdf) found something surprising: knowledge workers completed tasks about 7% faster when interrupted versus when left undistracted ([see study details](#mark-2008-study)). But this speed came at the cost of higher workload, higher stress, more frustration, and changes to work style to compensate for the pressure.

More recent research from Gloria Mark and others reveals deeper patterns. Workers now switch tasks every **3 minutes** on average ([Mark et al., 2014](#mark-2014)), and nearly half of these interruptions are self-initiated. We interrupt ourselves. What's more, it takes an average of **23 minutes and 15 seconds** to fully refocus on a task after an interruption ([Mark, 2024](#mark-2024)).

That last finding is critical. If we're switching tasks every 3 minutes but need 23 minutes to fully refocus, we never reach full cognitive recovery before the next interruption arrives.

### Attention Residue

When you switch from Task A to Task B, your brain doesn't fully let go of Task A. Neural activity related to the first task remains active even as you engage new cognitive resources for the second task. Sophie Leroy coined the term "attention residue" for this phenomenon ([Leroy, 2009](#leroy-2009)): the cognitive remainder left behind when you switch contexts.

![][image1]
**Figure 1:** Visually, we can think of this like a funnel where sand from each task pours in. If not allowed to drain fully, the sand from each task mixes together, degrading your ability to think clearly about any single task.

The implications extend beyond stress. Studies in healthcare settings found that interruptions increase diagnostic errors by **12.1%** and prescription errors by **12.7%** after a single interruption ([see details](#westbrook-2017-study)). Mid-task interruptions are particularly costly, roughly doubling error rates compared to interruptions at natural task boundaries ([Bailey & Konstan, 2006](#bailey-2006)).

It’s important to note however, that these healthcare studies measured 1-6 interruptions in clinical settings, not the 80-165 interruptions typical in knowledge work. We can't extrapolate linearly. But the direction seems clear: interruptions degrade decision quality, and attention residue compounds with each context switch.

Applied to the future of knowledge work in which we spend more time making decisions and thinking creatively, the total cost of interruptions also goes up.

### The Recovery Window Impossibility

Let's visualize what this looks like in practice.

![][image2]
**Figure 2:** Each interruption triggers a recovery curve—your focus gradually returns toward baseline over 23 minutes. But with interruptions arriving every 3-10 minutes, new interruptions arrive before recovery completes. You're left in a chronic state of partial attention.

This is, of course, influenced by our willpower and discipline, but it's also a structural and design problem. When interruptions arrive faster than the brain can recover, sustained concentration becomes impossible.

Research on email and Slack usage suggests knowledge workers face approximately **165+ interruptions per day** (combining ~77 email checks and ~88 Slack checks) ([Mark et al., 2016](#mark-2016); estimated). That's roughly **17-20 interruptions per hour**, or one every 3-4 minutes closely matching empirical findings. Batching interruptions where possible helps to mitigate this, although you trade off additional context switching when you get back to your notifications.

## The Network Effect Amplifier

Chat apps are not inherently different from email in their ability to interrupt an individual. However, behaviorally we treat chat apps differently, with more allowance for casual conversation and long back and forth which has been mixed with historical content. It's the networked structure of channels and direct messages which creates a huge increase in the number of chats happening.

### The Mathematics of Connection

Consider a team of 10 people. How many possible groups can form?

- **45 pairs** (1:1 connections)
- **120 groups of 3**
- **210 groups of 4**
- **252 groups of 5**

Each group represents a potential channel or direct message, a context you might need to track.

![][image3]
**Figure 3:** The growth is explosive. With 50 people, you get 1,225 possible pairs, 19,600 possible groups of 3, and over 2.1 million possible groups of 5\.

Organizations don't form all possible groups, of course. But research on Enterprise Social Media Platforms (ESMPs) shows that Slack-like tools amplify network effects when compared with email. After adoption, communication ties increase by **7.8%** and one-to-many communications jump by **14.3%**. Typical organizations end up with **2-3× as many channels as employees** ([Ribeiro et al., 2025](#ribeiro-2025)). The result is that even with attempts at compartmentalization, the number of potential interruption sources grows much faster than the raw team size.

### How This Manifests

Just like with drawing a hand from a deck of cards, we can 'draw' people from the total team to create groups; and just like with decks of cards, the number groups (hands) that a person (card) can be in is enormous. We can count them in an organized way by swapping out one member at a time, shown below:

![][image4]
**Figure 4:** In a 16-person team, if you're a typical member, you're in:

- **15 possible pairs** (direct 1:1 connections)
- **105 possible groups of 3**
- **455 possible groups of 4**

That's **575 possible groups** containing you. Not all will be active channels or direct messages, but even if only 10% become active conversation threads, that's 57 contexts to track. What's more, these channels or direct messages don't break topics down within them. Direct messages are infinite, covering your entire conversation history in that group, meaning your attention is spread as soon as you enter that direct message or channel.

Inboxes work differently. They lack channel discovery and discourage sprawl. Most emails are 1:1 or small groups focused on a specific task, topic, decision or role. Threading happens by default, and async norms are culturally established. Inbox threads are topic-based, not group-oriented, and context can be brought in naturally without forcing a context switch.

Slack, by contrast, is designed around engagement in ways that encourage noise. Many-to-many channels may center around a group or topic but often stay broad. Real-time is the default, complete with status indicators and the ability to 'notify anyway' even when users set themselves to away. While useful or necessary for some roles, the truth is that most notifications in knowledge work can wait the 2-4 hours for your batch checking.

The internal focus combined with channel discoverability creates what researchers call "weak-tie formation": connections that wouldn't naturally form in email-based workflows and may not add real organizational value. More connections mean more potential interruptions.

Research confirms this: studies found that apps like Slack result in approximately **14% more notifications** from internal teammates compared to email alone ([Ribeiro et al., 2025](#ribeiro-2025)). Part of this comes from behavioral norms (chat feels more casual), but part is structural in the design of Slack leading to many potential conversations and self-interruption opportunities.

Now consider how this might change as AI accelerates our ability to to ship new ideas. The analogy I like to use is a team of people tasked with digging the deepest hole; AI gives everyone a backhoe but no additional direction on where to dig. Apps like Slack encourage digging holes in the wrong places through channel discoverability and the ease at which one-to-many communications sprawl.

### Your Interruption Profile

Want to estimate your own interruption frequency?

![][image5]
**Figure 5:** Based on your number of channels, direct messages, and email checks, this calculator estimates your daily interruptions and the cognitive recovery time required. The math is grounded in Slack's published data (92 messages per user per day, 62% in channels, 38% in direct messages) and Gloria Mark's 23-minute recovery finding.

## A Different Architecture

So what do we do about this? First, let's acknowledge the reality that some roles require frequent availability. Support teams, incident responders, and time-sensitive coordination genuinely benefit from real-time communication. The cognitive costs are necessary in order to perform that role.

But for knowledge work, and in particular the future of knowledge work focused on decision making, creative thinking and trust-building, there's growing evidence that we can do better.

### The Research on Batching

Studies by Kostadin Kushlev and Elizabeth Dunn (2015) and Patrick Fitz et al. (2019) found that **checking email and notifications 3 times per day** (morning, midday, end of day) significantly reduces stress without increasing FOMO or reducing responsiveness ([Kushlev & Dunn, 2015](#kushlev-2015); [Fitz et al., 2019](#fitz-2019)). Batching creates predictable windows for focus and recovery. Long uninterrupted work blocks allow full recovery. You process notifications in bulk, reducing the cost per context switch. Most importantly, **you retain control over when interruptions occur** preventing context switching from your high-value focused work.

Compare the following differences between two checking timelines:

* **Continuous checking:** 165 interruptions/day × 23 min recovery \= **63 hours** of theoretical recovery time needed (obviously impossible in an 8-hour workday)
* **3× daily batching:** 3 scheduled check-ins × 23 min recovery \= **69 minutes** of recovery time (easily accommodated)

The difference is both quantitative and qualitative; with batching, you can actually reach those mythical flow states, whereas with continuous checking, you can't. While it might not feel that different, in practice you’re able to focus for multiple hours more each work day, significantly improving performance and reducing stress. This is important as the nature of work increasingly emphasizes judgement and creativity more, thanks to AI.

### Async-First Design

As mentioned above, we don't use Slack or other real time chat apps at Convictional. Our communication happens primarily via email, directly on the task via in-line chat, and deliberate, focused meetings. We certainly still have our own interruptions, and it's easy enough to find yourself breaking the batching rule for checking email. Further, if we were to use Slack,  they [offer lots of ways to customize your notifications](https://slack.com/help/articles/201355156-Configure-your-Slack-notifications) or turn them off temporarily.

However, even with notifications silenced, Slack's design encourages self-interruption. Unread badges create visual pressure. Channel discoverability invites browsing. Real-time presence indicators (green status dots) create urgency. Search surfaces conversations you weren't tracking. Direct messages are group-centric instead of topic-centric, mixing contexts from your entire conversation history. Given that **49% of interruptions are self-initiated** ([Mark et al., 2014](#mark-2014)), design choices matter.

A unified inbox with your emails, chats and tasks, by contrast, batches by default. You check it rather than it interrupting you. Most emails are 1:1 or small groups, meaning fewer contexts to track. Threading and views support organization. Async norms are culturally established, and threads stay topic-based rather than group-oriented. This allows people to significantly increase focused time each day.

This isn't to say an inbox is perfect. It has its own issues, and there was a good reason Slack found product-market fit when they did. But based on how our brains operate, a single prioritized inbox better aligns with how attention and recovery work.

## Conclusion

Interruptions and distractions are inevitable in collaborative environments. But the frequency, timing, and structure of those interruptions shape cognitive outcomes more than we typically acknowledge. What's more, and as already mentioned, these interruptions become more costly as more knowledge work shifts to thinking, deciding, and aligning requiring un-interrupted focus in order to optimize.

Slack amplifies connectivity through network effects—and in doing so, amplifies interruption potential. The average knowledge worker now faces 165+ context switches per day, arriving too frequently to allow cognitive recovery. In future, we’ll likely face more as our teams iterate from idea to idea more quickly. The result: chronic attention residue, degraded creativity and decision quality, increased stress and frustration, and the feeling of being "busy" without accomplishing deep work.

There's no one-size-fits-all solution, but at Convictional, we're designing collaboration tools that embody these learnings. Tools that assume focus is valuable and interruptions should be deliberate. If you're tired of the constant noise, [check us out](http://convictional.com).

---

## Appendix: Limitations and Research Context

### Key Research Study Details

#### Mark et al. (2008): The Cost of Interrupted Work {#mark-2008-study}

**Study Design:** Participants were tasked with assuming an HR role at a craft supply company and using a cheat sheet to respond to emails after a vacation to clear an inbox. Different groups were exposed to varying levels of distraction (no distractions, in-context distractions, out-of-context distractions) while completing this task.

**Key Finding:** Interrupted groups completed the task ~7% faster than their non-distracted peers *without* an increase in errors (measured as typos, incorrect names, spelling/grammar errors). However, this speed came at the cost of significantly higher workload, stress, and frustration.

**Source:** Mark, G., Gudith, D., & Klocke, U. (2008). The cost of interrupted work: More speed and stress. *CHI '08: Proceedings of the SIGCHI Conference on Human Factors in Computing Systems*, 107-110.

#### Westbrook et al. (2017): Interruption Effects in Healthcare {#westbrook-2017-study}

**Study Design:** Observational study in clinical settings measuring the impact of interruptions on nurses administering medications and doctors making diagnostic decisions. Measured 1-6 interruptions in high-stakes, time-bounded tasks with clear error metrics.

**Key Findings:**
- Diagnostic errors increased **12.1%** after a single interruption
- Prescription errors increased **12.7%** after a single interruption
- Medication errors **doubled at 4 interruptions** and **tripled at 6 interruptions** within a 10-minute window

**Important Context:** This study measured healthcare-specific errors in clinical contexts. The magnitude of effects may not directly translate to knowledge work, though the direction (interruptions degrade decision quality) appears robust.

**Source:** Westbrook, J. I., et al. (2017). Association of interruptions with an increased risk and severity of medication administration errors. *Archives of Internal Medicine*, 170(8), 683-690.

---

### On Extrapolating Healthcare Research

Throughout this essay, we reference healthcare studies showing 12% error increases from interruptions. It's critical to understand what these studies measured, and what they didn't.

Westbrook et al. (2017) and related AHRQ studies measured 1-6 interruptions in clinical settings—nurses administering medications, doctors making diagnostic decisions. These are high-stakes, time-bounded tasks with clear error metrics.

Knowledge work is different:

- 80-165 interruptions per day (not 1-6)
- Tasks are less bounded (writing code, drafting strategy, analyzing data)
- Errors are often subtle or invisible (suboptimal decisions, more code bugs, missed insights)
- Recovery is different for every individual, we present averages

**We cannot extrapolate linearly.** If one interruption causes a 12% error increase, 118 interruptions don't cause a 1,416% increase, that's obviously not the case based on experience. However, we can be confident directionally as cognitive processes are physical, so while decision making environments are not always the same, reasoning processes are subject to the same constraints.

So, what we can say is:

- The direction is clear: interruptions degrade decision quality
- The mechanism (attention residue) applies generally
- Chronic interruption likely creates chronic cognitive deficit
- The magnitude in knowledge work is uncertain

When we provided the support organization calculation earlier, we framed it conservatively: **if** we could eliminate interruptions entirely (returning to a zero-interruption baseline), error rates might decrease by \~11% (calculated as 1 \- 1/1.12). This is the maximum theoretical benefit, not a prediction.

The actual impact depends on:

- Your baseline error rate
- What constitutes an "error" in your work
- Whether errors compound or occur independently
- Individual variation in susceptibility to interruption

We include the calculation not as precise prediction but as illustration: even modest improvements in decision quality can have significant cumulative impact at organizational scale.

### On Individual Variation

Gloria Mark's research provides averages: 23 minutes to refocus, 3-minute task switching. But individuals vary widely:

- Some people recover faster from interruptions
- Task complexity affects recovery time
- Prior experience with a domain reduces cognitive load
- Some individuals are more interruption-tolerant

The visualizations in this essay use population averages. Your mileage may vary.

### On Team Coordination Benefits

This essay focuses on individual cognitive costs. It doesn't fully account for:

- Coordination benefits of real-time communication
- Time-sensitive issues requiring immediate response
- Weak-tie connections that wouldn't form in async settings
- Team preferences and cultural factors

These trade-offs are real. The goal isn't to maximize individual focus at the expense of team effectiveness. It's to make the trade-offs explicit and conscious, rather than accepting default tool behaviors.

### On Network Effect Calculations

The combinatorial calculations (C(n,k), formally "N choose K," written as $\\binom{n}{k} \= \\frac{n\!}{k\!(n-k)\!}$) represent possible groups, not active groups. Real organizations don't form all possible combinations, or may form duplicates of some groups (e.g. different channels with the exact same members). We use these numbers to illustrate:

- How quickly potential communication surfaces grow
- Why larger teams feel more chaotic
- The structural reason Slack creates more interruptions than email

The interruption calculator makes assumptions about:

- Messages per channel/direct messages per day
- How messages translate to interruptions (not 1:1)
- Channel activity patterns

These are modeling choices based on Slack's published data (92 messages/day per user, 62% channels, 38% direct messages) but include uncertainty.

### On Attention Recovery Curve

In Leroy’s 2009 article, the specific shape of the recovery curve is not specified. However, an exponential decay has been used frequently in cognitive psychology when estimating attention. For this article, we use an exponential decay fitted to a decay period of 23 minutes (on an arbitrary scale of ‘100 attention units’). Specifically, the form of the equation is `1 - exp(-t / 7.67)`

### On Slack Specifically vs. ESMPs Generally

Some research cited (7.8% increase in ties, 14.3% increase in one-to-many communications) comes from studies of Enterprise Social Media Platforms, specifically Microsoft's Engage across 99 companies that were adopting it in addition to Email (Outlook) and Instant Messaging (Teams). Slack differs from these platforms:

- Channel-based rather than feed-based - however one-to-many posts in channels mimics the broadcast behaviour of Engage
- More work-focused rather than social-focused
- Different notification patterns

We believe the directional findings apply to Slack, but exact magnitudes may differ. Slack's specific impact on interruption frequency deserves more direct study.

---

## Complete References {#references}

- <span id="mark-2014"></span>Mark, G., Iqbal, S. T., Czerwinski, M., & Johns, P. (2014). Bored monkeys prefer more workload than less: The effect of interruptions on work. *Proceedings of CHI 2014*. [3-minute task switching; 49% self-interruption]
- <span id="mark-2024"></span>Mark, G. (2024). *Attention Span: A Groundbreaking Way to Restore Balance, Happiness and Productivity*. [23 min 15 sec recovery time; 47-second attention span]
- <span id="leroy-2009"></span>Leroy, S. (2009). Why is it so hard to do my work? The challenge of attention residue when switching between work tasks. *Organizational Behavior and Human Decision Processes*, 109(2), 168-181. [Coined "attention residue"]
- <span id="leroy-2018"></span>Leroy, S., & Glomb, T. M. (2018). Tasks interrupted: How anticipating time pressure on resumption of an interrupted task causes attention residue. *Organization Science*, 29(3), 380-397. [Ready-to-resume plans reduce residue]
- <span id="bailey-2006"></span>Bailey, B. P., & Konstan, J. A. (2006). On the need for attention-aware systems: Measuring effects of interruption on task performance, error rate, and affective state. *Computers in Human Behavior*, 22(4), 685-708. [Mid-task interruptions double error rates]
- <span id="kushlev-2015"></span>Kushlev, K., & Dunn, E. W. (2015). Checking email less frequently reduces stress. *Computers in Human Behavior*, 43, 220-228. [3× daily batching reduces stress]
- <span id="fitz-2019"></span>Fitz, N., Kushlev, K., Jagannathan, R., Lewis, T., Paliwal, D., & Ariely, D. (2019). Batching smartphone notifications can improve well-being. *Computers in Human Behavior*, 101, 84-94. [3× daily batching optimal]
- <span id="ahrq"></span>Agency for Healthcare Research and Quality (AHRQ). Diagnostic Safety Series on interruption effects in clinical settings.
- <span id="rubinstein-2001"></span>Rubinstein, J. S., Meyer, D. E., & Evans, J. E. (2001). Executive control of cognitive processes in task switching. *Journal of Experimental Psychology: Human Perception and Performance*, 27(4), 763-797. [40% productivity loss from task switching]
- <span id="lane-2024"></span>Lane, J. A., et al. (2024). *Teams in the Digital Workplace: Collaboration, Communication, and Coordination*. MIT Press.
- <span id="slack-2024"></span>Slack. (2024). *Work is Fueled by True Engagement*. Slack Blog. [87% improved communication, 5B actions/week, usage statistics]
- <span id="demandsage-2025"></span>DemandSage, StatsUp, Business of Apps. (2025). Slack usage statistics. [47M DAU, 92 messages/day, 83-92 actions/day]
- <span id="ribeiro-2025"></span>Ribeiro, Shapiro, Suri (2025); The Effects of Enterprise Social Media on Communication Networks. https://dl.acm.org/doi/10.1145/3717867.3717875 [7.8% increase in ties, 14.3% increase in one-to-many, 2-3× channels per employee]
- <span id="mark-2016"></span>Mark, G. et al. (2016); Email Duration, Batching and Self-interruption: Patterns of Email Use on Productivity and Stress. https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/Email20Duration20Camera20Ready20submission3-1.pdf [77 email checks/day baseline]
