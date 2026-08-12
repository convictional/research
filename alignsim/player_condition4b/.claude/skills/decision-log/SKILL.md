---
name: decision-log
description: View the running log of decisions the team has recorded on Posts
allowed-tools: Bash
---

The decision log is the running record of every decision the team has recorded on a Post (via
`./game post decide`), oldest first. Each entry shows the decision itself, who recorded it, the
turn it was decided, and the Post it lives on — so you can see what's already been settled without
opening every thread. Open the underlying Post with `./game post read <post-id>` for the full
discussion behind a decision. This is about *where* settled decisions live, not *what* to decide
in the game.

View the decision log:

!`./game decisions`
