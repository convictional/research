"""
Pair mining for contrastive learning.

Implements unsupervised positive/negative pair mining strategies from PLAN.md section 5:
- GitHub: same_thread pairs, parent↔reply relationships
- Docs: adjacent sections, same heading pairs
- Email: subject↔body, thread pairs
- etc.
"""
