# Senior-researcher review prompt

Paste everything below the line into Claude Code (Claude Fable, effort `high`),
started in `/home/alibi/lab_new/Phase_Diagrams_V2` on the machine that has the
data. Claude Code reads `CLAUDE.md` on its own.

---

I'd like you to take over this project as its senior researcher, with full
oversight and the authority to change whatever you judge will give the best
result.

**Why:** we predict binary phase diagrams (liquidus, solidus, diagram type,
eutectic point, miscibility gap) from properties of the two elements, using the
MPDS diagrams in `/home/alibi/lab/processed_v6`. This is a research project for
my supervising professor. We have improved the model step by step (liquidus
error 166 K to 126 K on the test split), but the gains are getting smaller, and
I'm no longer sure our overall approach is the right one. I need someone to
look at the whole thing with fresh eyes and decide where it should go.

**What I want from you:**

1. Review the whole project: the data and how it was prepared, the code and
   pipeline, how we evaluate, and the results history in `CLAUDE.md` and
   `results/`. Form your own view of what actually limits accuracy. Don't take
   our earlier conclusions on trust; check the ones your decisions rest on.
2. Decide on the approach. Keeping the current pipeline is one option. Others
   include a different way of representing or predicting the diagrams,
   physics-based models (for example thermodynamic or CALPHAD-style), different
   targets, features or data, or methods from the literature on phase-diagram
   prediction. It's equally valid to conclude that we should drop our approach
   and switch to something else. Weigh each option by the gain you expect, the
   evidence for it, and what it would cost us.
3. Carry out your best plan, end to end. Implement it, test it properly
   against the current best model (v13), and keep iterating while it pays off.
   If you change direction along the way, say why.
4. Write a report for me and the professor in `docs/research_review.md` (below).

**What counts as better:** lower liquidus and solidus error in 5-fold
cross-validation, winning in at least 4 of 5 folds, and on the test split as
a final check, without making the diagram type, eutectic or miscibility-gap
predictions worse. Report clean systems separately from monotectic, incomplete
and volatile-element ones. If you think these metrics are the wrong target
for the scientific goal (for example, that physically correct diagram shapes
matter more than average error), make that case in the report and propose
better ones.

**Your authority:** this codebase is yours. What it's for and how it's built
are your call, and you'll keep building on it. You may:
- restructure or rewrite any code, remove what is obsolete, and change defaults;
- create derived datasets, re-run preprocessing into new folders, add features
  or external element data (with the source and licence recorded);
- train and evaluate as many models as you need;
- use subagents for independent experiments that can run in parallel. Each one
  adds cost, so delegate when the parallel speed-up is worth it;
- search the web for relevant methods and papers, citing what you use.

Work on new branches and commit as you go, with results and console logs in
`results/` as `CLAUDE.md` describes. **Ask me first** before you:
- merge into `main`, force-push or rewrite git history;
- delete anything outside this repository;
- modify `/home/alibi/lab/processed_v6` or the raw MPDS data;
- start runs you expect to take more than about three hours.

Keep the test split for final reporting; make your decisions with
cross-validation.

# Finishing work

Please avoid stopping while you still owe work I asked for. Be persistent and do
not stop unless needed. Status notes are welcome, as is asking me about open
decisions, but carry on with work that does not depend on my answer. You may
meet ordinary obstacles like errors, timeouts, failing runs or missing packages.
Diagnose them first, and when they are not real blockers, work through them
(fix the code, retry, install what's needed in the `ml` environment, use another
route) rather than stopping. If something is clearly a deliberate blocker, such
as the actions above that need my approval, leave it alone, say plainly what you
found, and look for another way to finish. This does not override the need for
confirmation on risky or destructive actions.

# Evidence

Before reporting progress or results, check each claim against a run from this
session. Only report what you can point to evidence for. If something is not
yet verified, say so. If an experiment fails or makes things worse, report it
plainly; a well-measured negative result is useful to us.

# Keeping notes

Keep `CLAUDE.md` current as you learn: update "Current best", "What we learned"
and "Ideas not yet tried", record corrections and confirmed approaches with the
reason they mattered, and delete notes that turn out to be wrong.

# The report (`docs/research_review.md`)

Write it for my professor, a materials scientist who has not seen our code or
your tool calls. Start with your verdict: keep, change or replace the approach,
and why, in a few sentences. Then cover:
- what limits accuracy now, with evidence;
- the options you considered, and why you chose what you did;
- what you built and what it achieved, as a before/after table against v13
  (overall and per data group);
- what didn't work;
- the next steps you recommend, ranked by expected gain.

# Communicating clearly

Write for an intelligent but low-context reader. The person reading has not seen
your tool calls, your reasoning or the files you opened. Messages should be
self-contained. Put the key points in the first few sentences. Use plain words
and full sentences. Do not use a term or abbreviation that you made up during
the work; say what the thing is. Use a paragraph of flowing prose where ideas
connect, and keep bullets for lists of separate items. Leave out unnecessary
details, but make sure each message has the key context.

Start by telling me in a few lines how you plan to approach the review, then
begin.
