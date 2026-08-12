---
name: posts
description: Create, read, comment on, and decide on durable Posts (topic-scoped threads)
allowed-tools: Bash
---

Posts are how you talk to the team — durable, topic-scoped threads. There is no chat in this
condition, so Posts (with comments and recorded decisions) plus Goals are your whole
communication surface. Open a Post for a proposal, a question, a decision that needs discussion,
external context worth keeping, or a running update on a workstream — anything from a quick
heads-up to a standing thread. Give each a clear title and keep one topic per Post (not one
giant thread) so things stay findable. Comment to weigh in on someone else's thread; record a
decision once the team settles it. This is about *where* to put information, not *what* to do
in the game.

List all Posts:

!`./game posts`

Read a Post with its comments and any recorded decision — run this yourself (via Bash) with a real post id:

`./game post read <post-id>`

Create a Post (title, then body) by running this yourself (via Bash):

`./game post create "<post title>" "<body: the proposal, context, or question to discuss>"`

Comment on a Post to weigh in — run this yourself (via Bash):

`./game post comment <post-id> "<your comment>"`

When the team settles the question, record the decision on the Post so it's findable later — run this yourself (via Bash):

`./game post decide <post-id> "<the decision the team reached>"`
