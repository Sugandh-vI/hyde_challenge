# Build an agent that plays Catan

**17/10/2026: 10:00 to 19:30 IST, submission from 11:00 to 17:00. Team size: 3-4.**

You will build a program that plays 2-player Catan. It plays against our two bots, F and O, and against the
programs the other teams build.

**New to Catan?** Read the game rules in the official rulebook,
[CATAN base game rules (catan.com, PDF)](https://www.catan.com/sites/default/files/2021-06/catan_base_rules_2020_200707.pdf).
Its almanac is a reference for the details. Our game differs
from that rulebook in two ways only: **2 players** instead of 3 or 4, and **no trading between
players**. Trading with the bank and ports works as the rulebook says. Everything else is as the
engine implements it; where the engine and the rulebook disagree, the engine decides.

Your agent runs on one CPU core, inside the time limit of section 5. You may use any language model
to write code and to analyze results, we provide you free GLM-5.3 Flash, frontier in open weights.

---

## 1. The game

Standard Catan with these settings.

| Setting | Value |
|---|---|
| Players | 2 |
| Victory points to win | 10 |
| Engine | catanatron, the exact build we give you |
| Board | official spiral layout, randomized board seeds |
| Trade between players | off |
| Trade with the bank and ports | on |

Our engine is the catanatron library, pinned to one version and patched by us. The library also
ships a few agents of its own. We call one of them, its `ValueFunctionPlayer`, **F**.

**Use the engine we give you. Do not install catanatron from PyPI.** The action space changed
between catanatron releases. `pip install -r requirements.txt` installs the right build.

---

## 2. Environment

Catan is an imperfect information game and we keep it that way.

Your agent **sees**, through the `obs` object it receives:

| What | Where |
|---|---|
| The board: tiles, numbers, ports, the robber, every settlement, city and road | `obs.board` |
| Your own hand: exact resource cards and exact development cards | `obs.my_resources`, `obs.my_devs` |
| Your own victory points, public and with your hidden cards | `obs.my_public_vps`, `obs.my_actual_vps` |
| The bank: the exact count of each resource left | `obs.bank` |
| The opponent's **public** victory points | `obs.opp_public_vps` |
| The opponent's exact **resource** hand | `obs.opp_resources` |
| The **number** of unplayed development cards the opponent holds | `obs.opp_dev_count` |
| Every development card either player has played | `obs.played_devs` |
| The turn number | `obs.turn` |
| Every move so far, with every dice roll | the `state.action_records` of a world from `obs.sample_world` |
| Your color, and the CPU you have left | `obs.me`, `obs.cpu_left` |

Your agent **does not see** which development cards the opponent holds unplayed. It therefore does
not see hidden victory point cards.

**Why the opponent's resources are not hidden.** There are 19 cards of each resource. You know the
bank and your own hand, so the opponent's hand is exactly `19 - bank - mine` for every resource.
Hiding it would hide nothing, so we compute it for you. The development deck is the only real
hidden information in 2-player Catan.

### 2.1 Searching under imperfect information

Call `obs.sample_world(rng)`. It returns a complete catanatron `Game` that agrees with everything
you can see. The opponent's unplayed development cards are drawn at random from the cards nobody
has shown.

Sample several worlds and combine the results if you want to act well under uncertainty. How many
worlds you sample, and how you combine them, is your decision.

---

## 3. Engine interface

### 3.1 Actions

The engine gives you a list of legal `Action` objects. Each carries a color, a type and a value.

| Action type | What `value` holds |
|---|---|
| `ROLL`, `END_TURN`, `BUY_DEVELOPMENT_CARD` | `None` |
| `PLAY_KNIGHT_CARD`, `PLAY_ROAD_BUILDING` | `None` |
| `BUILD_SETTLEMENT`, `BUILD_CITY` | a node id, `int` |
| `BUILD_ROAD` | a node pair, `(int, int)` |
| `MOVE_ROBBER` | `(coordinate, victim_color_or_None)` |
| `DISCARD_RESOURCE`, `PLAY_MONOPOLY` | a resource name, for example `"WHEAT"` |
| `PLAY_YEAR_OF_PLENTY` | two resource names |
| `MARITIME_TRADE` | the cards you give, padded with `None`, then the card you get |

The engine does not generate trades between players.

**You return an index into the list.** You never build an `Action` yourself.

### 3.2 Forced moves

When the engine offers exactly one legal action, the harness plays it. It does not call your agent.
In F-against-F play, 47% of all decisions have exactly one legal action. You do not need to handle
them, and they do not count against your clock.

### 3.3 State

`game.state` holds the board, a flat `player_state` dictionary, the resource bank and the
development deck. Your agent reads the observation object instead, which hides what section 2 says
is hidden.

The engine also offers `create_sample_vector`, which returns 614 numbers describing the position.
A look at it can be helpful.

---

## 4. Submission

One directory containing `agent.py` and the files it uses, uploaded as a zip: more Python files
of your own, and data files such as weights or tables. Total size limit 100 MB.

**Python only.** Compiled code is not accepted: an upload holding a shared library, a compiled
extension or an executable is rejected. Your agent may not start other programs. numpy, scipy and
PyTorch already run their heavy work in compiled code.

Your agent may import the Python standard library and the packages pinned in `requirements.txt`:
numpy, scipy, scikit-learn and PyTorch (CPU build). Nothing else is installed on our machine.

Your agent runs with its own directory as the working directory. Load your files relative to it,
for example `os.path.join(os.path.dirname(__file__), "weights.pt")`. On our machine that directory
is read-only. Write temporary files to `tempfile.gettempdir()`: a directory private to your agent's
process, deleted when the process ends.

```python
class Agent:
    def __init__(self):
        """Load weights or tables here. Called once, when your agent's process starts."""

    def decide(self, obs, playable_actions) -> int:
        """Return the INDEX of your chosen action in playable_actions."""
```

One process plays many games in a row, all against the same opponent, and the harness restarts it
after a crash. So one `Agent` object sees many games, and it may learn from them: adapting to an
opponent is allowed. Nothing carries over from one process to the next.

### 4.1 Two complete agents ship with the kit

**`examples/random_agent.py`** is legal, runs, and scores 0.000 against F.

```python
import random


class Agent:
    def __init__(self):
        self.rng = random.Random(0)

    def decide(self, obs, playable_actions) -> int:
        return self.rng.randrange(len(playable_actions))
```

**`examples/f_agent.py` is F itself**, written against this interface. F is the bot you have to
beat, and we give you its code. It scores each position with a weighted sum of 13 features and
plays the move that leads to the best-scoring position. It looks only one move ahead.

```python
import random

from catanatron.players.value import DEFAULT_WEIGHTS, get_value_fn

VALUE_FN = get_value_fn("base_fn", DEFAULT_WEIGHTS)


class Agent:
    def __init__(self):
        self.rng = random.Random(0)

    def decide(self, obs, playable_actions) -> int:
        world = obs.sample_world(self.rng)
        best_i, best_v = 0, None
        for i, action in enumerate(playable_actions):
            successor = world.copy()
            try:
                successor.execute(action)
            except Exception:
                continue
            v = VALUE_FN(successor, obs.me)
            if best_v is None or v > best_v:
                best_i, best_v = i, v
        return best_i
```

### 4.2 Rules

1. Return an integer index into `playable_actions`.
2. No network access. Read only your own directory and the installed packages; write only to
   `tempfile.gettempdir()`.
3. If your agent raises an exception or returns an illegal index, we play a random legal action and
   record it. Above 2% of your decisions, we disqualify the submission.
4. Each team has **7 scored submissions**. Before we score an upload, it must load and play two
   test games without an error. An upload that fails this is reported back to you with the error and
   does not use one of your 7. Each team may upload at most 10 times in all, failed uploads
   included. While 2 of your uploads wait to be scored, the next one waits its turn.
5. The practice harness on your own machine is unlimited.

---

## 5. Time limits

### 5.1 Our machine

Every team gets the same evaluation environment:

* An AWS c7a.48xlarge: AMD EPYC 9R14 processors, 192 cores, no hyperthreading.
* **One core per game, and 2 GB of memory per agent**.
* Python 3.11, with exactly the packages and versions of `requirements.txt`.
* The patched catanatron build of this kit.

Timings on your own laptop are for your use only. **Our machine is the reference**, and the
leaderboard reports the numbers measured on it.

### 5.2 What we cap

We cap your agent's **own CPU time in each game**. Every thread and child process your agent starts
counts towards it. The engine and your opponent do not.

The cap works like a chess clock with an increment. In each game your agent starts with 300 ms and
gains 12.5 ms for every decision it is asked to make, up to a ceiling of 1.5 seconds:

```
allowance after k decisions = min(300 ms + 12.5 ms x k, 1,500 ms)
```

A game asks your agent for about 56 decisions on average, so the allowance averages about
**1 second per game**. A long game gets proportionally more, up to the ceiling, which is reached
at 96 decisions: no game gives your agent more than **1.5 seconds** in total. The 300 ms at the
start is there for the opening placement, the decision with the most legal moves. You may spend the
allowance unevenly, for example by thinking longer in the opening.

**When your agent has used up its allowance, we play uniformly random legal moves for the rest of
that game.** The leaderboard shows in how many games this happened.

`obs.cpu_left` tells your agent, at every decision, how many seconds of the allowance remain,
including the 12.5 ms for that decision. Use it to budget your search.

Three more limits stop a broken agent from holding up a match:

* One decision may take at most 3 seconds. An agent that does not answer in time is stopped and
  restarted, and random moves finish that game.
* Time your agent spends waiting without using the CPU, for example sleeping, is limited to 3 seconds
  per game beyond its CPU allowance.
* Loading your agent, that is importing `agent.py` and running `__init__`, may take up to 60 seconds
  and does not count against the clock.

For scale, mean CPU per game through the evaluation script, on one idle core of the processor our
machine uses (AMD EPYC 9R14):

| | per game | per decision |
|---|---|---|
| `random_agent.py` | 11 ms | 0.6 ms |
| `f_agent.py`, which is F | 158 ms | 2.9 ms |
| the allowance | about 1,000 ms, at most 1,500 ms | 12.5 ms |

So you get about six times F's compute. A few things to keep in mind:

1. **Your machine is probably faster per core than ours.** A recent Apple laptop ran F in 96 ms per
   game, 1.65 times faster.
2. **A laptop running other jobs can report two to three times the CPU** for the same games.

So if you develop on a recent laptop, plan for our machine to charge about twice(some overhead due to non-idle cores) what yours
reports, and use the leaderboard's "games over the clock" column to check.

Receiving the observation costs under 1 ms per decision, and it is charged to your agent.

**Exactly one core.** Threads share it, so they buy you nothing. Your agent cannot start
worker processes: Python's `multiprocessing` refuses inside the harness, so a pool or a data loader
with workers fails when your agent loads.

### 5.3 Your results and the leaderboard

Sign in to the team portal with your team name and your team leader's phone number. It shows every
upload you made, what became of it, and for each scored submission, measured on our machine:

| Column | Meaning |
|---|---|
| Win rate against F | over 1,000 games, with its 95% interval |
| Win rate against O | over 1,000 games, with its 95% interval |
| Mean agent CPU per game | your agent's own CPU, as in section 5.2 |
| Games over the clock | games in which your agent used up its allowance |
| Faults | exceptions, illegal indices, timeouts and crashes |
| Submission number | 1 to 7 |

The portal's leaderboard shows the top 10 teams, each by its best submission, on the public set of
boards. The top-30 cut and strength use a private set, and the group stage new boards (section 8),
so do not overfit on the leaderboard.

---

## 6. The evaluation script

`catan_kit/evaluate.py` is **the script we score with**. The only things that change on the day are
the board seeds, the dice key and the opponents.

Before anything else, check your install:

```
python -m catan_kit.selfcheck
```

Then play your agent against F:

```
python -m catan_kit.evaluate --a my_team/agent.py --b examples/f_agent.py --games 1000
```

The script plays every board twice, once from each seat, with the same dice both times. The engine
stops a game at 1,000 turns; a game with no winner counts as a loss for both sides, here and in
every score we compute. It prints
your win rate with its 95% interval, your mean and worst CPU per game, the games over the clock,
your faults, and how long your agent took to load (see README.md).

**How the script runs your agent.**

1. Your agent runs in its own process, separate from the engine and from the opponent. Both our
   bots run through the same harness, under the same clock.
2. For each decision it receives a copy of the game. In that copy the opponent's unplayed development
   cards and the order of the development deck are redrawn at random, so the copy holds nothing that
   section 2 hides. Changing the copy changes nothing in the real game.
3. Your agent's CPU time is read from the operating system, from outside your process.
4. In scored games the dice come from a secret key, so knowing the board does not tell you the rolls.
5. Python's `random` module in your agent's process is reseeded at the start of every game, from
   that game's seed. The engine uses it when a move has a random outcome, such as the card a robber
   steals, so an agent that executes such moves in its search still replays exactly.
6. On our machine, each team's agent also runs as its own user, with no network, no access to any
   files but its own, one CPU core and 2 GB of memory.

Run it against F before every submission. If your agent stays inside the clock on our machine, its
win rate against F on the leaderboard should fall inside the interval the script prints.

---

## 7. Your opponents

**F** is catanatron's `ValueFunctionPlayer`. Here is the rest of the library measured against it,
at 10 victory points, over 1,000 games with seats alternated, through the same harness your agent
runs in: the same hidden information (each library bot plays on a world sampled from what it can
see, as your agent can) and the same CPU allowance (section 5.2). Read the numbers as a ranking.

| Bot | Win rate against F | Games in which its allowance ran out |
|---|---|---|
| `RandomPlayer` | 0.001 | 0 of 1,000 |
| `VictoryPointPlayer` | 0.001 | 0 of 1,000 |
| `WeightedRandomPlayer` | 0.001 | 0 of 1,000 |
| `MCTSPlayer`, 25 simulations | 0.001 | 1,000 of 1,000 |
| `AlphaBetaPlayer`, depth 3 | 0.001 | 1,000 of 1,000 |
| `SameTurnAlphaBetaPlayer`, depth 2 | 0.080 | 415 of 1,000 |

The three search bots lose because they run out of allowance and then play random moves
(section 5.2). A search that fits the allowance is the point of the exercise.

You also play **O**, our own trained bot, which beats F by some margin.

Both bots are anchors. You may submit a copy of F. It is legal.

---

## 8. The funnel

All times are IST. Submissions open at 11:00. Every submission is scored twice: on a **public** set
of boards, which the leaderboard shows, and on a **private** set, which nobody sees during the day.

### Checkpoint, 15:30

15:30 is a checkpoint, not a cut. No team is out at 15:30, and uploads stay open until 17:00.

### Last upload, 17:00

Uploads close at 17:00; we do not accept later ones. You do not pick a submission: the group stage
plays your best one by the mean of its two private-set win rates.

### The top 30 and the group stage

At 17:00 we rank the teams by the mean of two win rates on the **private** set, against F
and against O, each team by its best submission, and take the top 30. They split into 3 groups of
10, balanced by the same number: groups are filled from the top, A, B, C, C, B, A, A, B, C, and so
on. The group stage uses new boards that no submission has played. Inside a group, every team plays
every other team 1,000 games, with seats alternated. **The top 2 of each group by robustness
(section 9.1) present.**

The top 30 and the group results are announced at 17:50.

### Reports and presentations

Every team in the top 30 uploads its report by **17:30**. The six presenting teams present from
18:00, in alphabetical order, 10 minutes each, with live questions.

**Do not be discouraged by the leaderboard.** It is measured on the public set, which you may
overfit without meaning to, and leaderboard strength carries at most 25% of the weight.

## 9. How we score you

We mark four things: robustness, strength, the report, and the presentation with its questions.
Robustness and strength together carry about the same weight as the report and the presentation
together.

### 9.1 Robustness

Robustness is your mean win rate against the half of your group opponents you do worst against.

For example, in a group of 10 you have 9 opponents, and your robustness is your mean win rate against
the 5 you do worst against.

### 9.2 Strength

Your win rates against the two anchor bots, on the private set, for the submission that plays the
group stage:

```
strength = 0.5 * win_rate_vs_F + 0.5 * win_rate_vs_O
```

### 9.3 Report

**Up to 4 pages**, not counting references, on A4 in a single column with text
of at least 11 pt. There is no minimum. Upload it as a PDF by 17:30, the same way as a submission (`~/upload.sh report.pdf`).
Every team in the top 30 writes one, and you have from 17:00 to finish it. We mark the reports of
the six teams that present.

We strongly advise you to keep analysing & experimenting throughout the hackathon to figure out why things fail or work. High win rate without good reasoning will only fetch marks in the first two components.

Write it for a reader who knows machine learning but has not seen your agent. We mark four things:
how clearly you explain your method, how well you justify your design decisions, insights you can draw from the whole journey and the quality of
your experiments. We'll reveal the detailed rubric after the event.

### 9.4 Presentation and questions

Six minutes, then four minutes of questions. You may present from
your report or slides as you like. 

We mark how well you defend the decisions in your report, what you learned that holds beyond this
game, and your answers to open questions about the work. The questions carry the most weight.

---

## 10. The day

| Time (IST) | Event |
|---|---|
| 10:00 | Session starts: who we are, Catan, this problem statement, and the coding-assistant inference you can use |
| 11:00 | **Submissions open.** Leaderboard opens |
| 15:30 | **Checkpoint.** No cut; uploads stay open until 17:00 |
| 17:00 | **Last upload** |
| 17:30 | **Reports due**, from every team in the top 30 |
| 17:50 | The top 30 and the group-stage results announced |
| 18:00 | Presentations by the top 2 of each group, in alphabetical order |
| 19:30 | End |

There is no earlier mandatory submission. You have the evaluation script, so you can check your own
agent whenever you want.

### About reproducibility(OPTIONAL)

Three separate things had to be fixed before the same seeds gave the same games twice. All are
handled for you by the kit.

**1. A bug in the engine.** The function listing bank and port trades returned them in a Python
`set`, and those entries contain `None`. On Python 3.11 and older, `hash(None)` comes from the
memory address of `None`, which changes every time a process starts, so the order of the list
changed between runs. When two moves scored equally, a bot picked a different one and the game
diverged. `catan_kit` sorts that list at import.

**2. Python's hash randomization.** The engine also iterates sets of strings, and Python randomizes
string hashes per process. Measured: the same 60-game run gave 3953, 3785 and 3822 agent decisions
on three tries. With `PYTHONHASHSEED=0` it gives 3901 every time. That variable only takes effect
at interpreter start, so `catan_kit.evaluate` and `catan_kit.selfcheck` restart themselves with it
set. **If you write your own driver script, call `catan_kit._patch.ensure_hash_seed()` at the top
of it**, or run it with `PYTHONHASHSEED=0`. Put the rest of the script under
`if __name__ == "__main__":`. The harness starts each agent in a new Python process, which imports
your script again. Without the guard, that import tries to start the games again and the script
stops with a `RuntimeError` about the bootstrapping phase.

**3. A hidden reseed.** The first call to catanatron's `get_feature_ordering`, which every feature
vector uses, builds a throwaway game with no seed, and that reseeds Python's global `random` in the
middle of whatever game is running. So the first game in each process played differently from the
same seed played later. The kit restores the random state around that call.

With all three in place, and with the seeding in section 6, a run replays exactly, and so does any
single game on its own, so you can replay a game your agent lost.

**Use Python 3.11.** We score on it, and catanatron needs at least 3.11. On 3.11 the same seeds gave
the same games on macOS and on Linux in our check: eight games of O against F, with the same
winner, length and number of decisions. Your rank does not depend on your local version, because
we score every submission on our own machine.

---

## 11. Fair play

1. **Your team's own work.** Do not share code, weights, data or submissions with another team.
   Do not submit another team's agent, or a modified copy of it.
2. **Built during the event.** Everything you submit must be built during the event, except public
   material: libraries, papers and open-source code. Name every piece of public material you used
   in your report.
3. **Language models.** You may use a language model to write code and to analyze results. Your
   agent may not call a language model, or any other service, while it plays.
4. **Play through the interface.** Your agent may use only the observation and the action list it
   receives. It is cheating to try to get information that section 2 hides, to interfere with the
   opponent, the evaluation harness, the time measurement or our machine, to try to learn the
   evaluation seeds or the dice key, or to use a submission to extract information from our machine.
5. **Colluding is cheating.** Losing on purpose, or arranging
   results with another team, is cheating.
6. **Your report describes what you did.** Every number in it must come from a run you made. We may
   rerun your agent and your experiments.
7. **Penalties.** We disqualify a submission or a team that breaks these rules. The organizers
   decide every case these rules do not cover, and their decision is final. If you are not sure
   whether something is allowed, ask us before you do it.

---

## 12. Reading

Some literature references. We suggest you to actively search for more as you figure out works well here. You don't have the time to try everything, your taste matters here.

**Overviews**

| Reading | Why |
|---|---|
| Yannakakis and Togelius, *Artificial Intelligence and Games*, Springer 2018, [free online](https://gameaibook.org) | A textbook on the whole field: search, learning, and how they combine |
| Browne et al., *A Survey of Monte Carlo Tree Search Methods*, IEEE Transactions on Computational Intelligence and AI in Games 2012 | Monte Carlo tree search and its many variants, in one place |

**Search, with dice and without**

| Reading | Why |
|---|---|
| Ballard, *The \*-minimax search procedure for trees containing chance nodes*, Artificial Intelligence 1983 | Alpha-beta extended to dice rolls: expectimax, with pruning |
| Kocsis and Szepesvári, *Bandit based Monte-Carlo planning*, ECML 2006 | UCT: spend the search where it is still uncertain |
| Campbell, Hoane, Hsu, *Deep Blue*, Artificial Intelligence 2002 | A hand-tuned evaluation function plus deep search beat a world champion. No learning at all |

**Hidden information**

| Reading | Why |
|---|---|
| Long, Sturtevant, Buro, Furtak, *Understanding the success of perfect information Monte Carlo sampling in game tree search*, AAAI 2010 | When that shortcut works and when it fails |
| Moravčík et al., *DeepStack*, Science 2017, [arXiv:1701.01724](https://arxiv.org/abs/1701.01724) | Poker. Acting well when you cannot see the opponent's cards |
| Brown and Sandholm, *Libratus*, Science 2018 | Also poker, and **not** reinforcement learning. Regret minimization plus search |

**Learning an evaluation, or a policy**

| Reading | Why |
|---|---|
| Buro, *Improving heuristic mini-max search by supervised learning*, Artificial Intelligence 2002 | Othello. A linear evaluation over simple features, fitted to game outcomes, beat the world champion |
| Hansen and Ostermeier, *Completely derandomized self-adaptation in evolution strategies*, Evolutionary Computation 2001 | CMA-ES: tune the weights of an evaluation such as F's directly on win rate, no gradients needed |
| Tesauro, *Temporal Difference Learning and TD-Gammon*, CACM 1995 | Backgammon, a dice game, learned by self-play. The network scores the position **after** a move rather than scoring moves, which handles a changing action list |
| Ross, Gordon, Bagnell, *A reduction of imitation learning and structured prediction to no-regret online learning*, AISTATS 2011 | DAgger: copying an expert works better when the expert also labels the positions your own agent reaches |
| Hinton, Vinyals, Dean, *Distilling the knowledge in a neural network*, 2015, [arXiv:1503.02531](https://arxiv.org/abs/1503.02531) | Train a small, fast model to copy a large one |

**Search and learning together**

| Reading | Why |
|---|---|
| Silver et al., *Mastering the game of Go with deep neural networks and tree search*, Nature 2016 | A learned value function guides the search, and the search improves the training data |
| Anthony, Tian, Barber, *Thinking Fast and Slow with Deep Learning and Tree Search*, 2017, [arXiv:1705.08439](https://arxiv.org/abs/1705.08439) | The same loop stated plainly: search, train on what the search found, search again |
| Silver et al., *A general reinforcement learning algorithm that masters chess, shogi and Go*, [Science 2018](https://www.science.org/doi/10.1126/science.aar6404) | No task specific insight is actually needed to master it |
| Schrittwieser et al., *MuZero*, Nature 2020, [arXiv:1911.08265](https://arxiv.org/abs/1911.08265) | Planning with a **learned** model of the rules |
| Schmid et al., *Student of Games*, 2021, [arXiv:2112.03178](https://arxiv.org/abs/2112.03178) | One algorithm for both perfect and imperfect information games |

Catan-specific work exists and you are free to find it.

---

## 13. Advice

### Before you train

1. **Spend some time understanding the game before jumping to optimisation.** 
2. **Look at the state your agent receives before you train on it.** Ask whether it contains what
   your method needs.
3. **Write down what you will compare before you run it.** One change at a time, against a version
   you have already measured.

### Measuring

4. **Don't ditch the approaches that don't top the leaderboard, use them for measuring robustness.** Section 9.1 is built so a narrow agent
   loses to a well-rounded one.
5. **Use enough games.** Evaluate on at least 400 games during training and 1,000 before
   you submit. A difference smaller than the confidence interval is not meaningful.
6. **Compare versions on the same seeds & separate eval seeds from train.**


### Time and compute

13. **Play games in parallel processes, with one thread each.** Set `torch.set_num_threads(1)` in
    every worker. More busy processes than cores slows every one of them.
14. **Keep checking your agent's CPU per game against the clock**, If a bigger model plays better but runs out of time, train a smaller one to copy it.
    Load and warm up everything in `__init__`, which is not charged to any game: work left for the
    first `decide` is charged to your first game.
15. **Use more than one machine.** Your laptops, and free CPU sessions on Kaggle or Google Colab, can
    play games and run training alongside each other. They are not our machine: check timings on
    the leaderboard.

### Habits that save the day

16. **Save checkpoints often**, with the settings and seed that produced them.
17. **Log every run's numbers to a file**: losses, win rate, games per second, CPU per game. Keep one
    table of all runs. You will need it for the report.
18. **Seed everything.** The same seed gives the same game, so a result you cannot reproduce is a
    bug to find.
19. **Submit early.** Your first submission tests the whole path from your folder to the
    leaderboard. You have seven.
20. **Write the report as you go.** Record each experiment when it ends, including the ones that
    failed. Section 9.3 rewards an honest failure.
