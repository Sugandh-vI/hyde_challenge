# Catan agent hackathon: starter kit

Build a program that plays 2-player Catan. It will play against our two bots, F and O, and against the other
teams' programs.

**Read [PROBLEM_STATEMENT.md](PROBLEM_STATEMENT.md) first.** This file is only for getting you started.

---

## Your team machine

Every team has its own machine for the hackathon: 8 CPU cores, 16 GB of memory, with the kit installed
and ready. You reach it with SSH through our gateway, using your team's username and password.

> [!TIP]
> Run **one command at a time**: paste a line, press Enter, wait for it to finish, then paste the next.
> Pasting several lines at once is the most common reason a setup goes wrong.

### Log in

Your username is `team` followed by your team name, all lower case, with no spaces. For example, the
team *blanc* logs in as `teamblanc`. Your initial password is your team name (all lower case, no
spaces) followed by the last 4 digits of your team leader's registered phone number, written twice. If
that number ends in 1234, team *blanc* starts with `blanc12341234`. Change it as soon as you are in,
as described below.

**Mac or Linux:** open Terminal. **Windows 10 or 11:** open PowerShell (SSH is built in).

```bash
ssh team<yourteam>@32.190.233.160
```

The first time, SSH asks whether you trust the gateway. Check that the fingerprint is exactly this one,
then type `yes`:

```
ED25519 key fingerprint is SHA256:Z8tsuVPiJ3dyI2vOQt2wV8hcBQxyaZOLpB2H4rnvvBE
```

Then enter your password. Nothing shows while you type it; that is normal.

#### Change your password

Do this once you are logged in. It asks for the current password, then the new one twice:

```bash
passwd
```

A new password must be at least 8 characters, differ from the old one and not be a palindrome, or
`passwd` refuses it and the old one stays.

### If the login does not work

- **"Connection closed by 32.190.233.160 port 22" right after the password**

  The password was wrong. A wrong password does not say "Permission denied": the connection just closes.
  Run the `ssh` command again and retype it carefully: it is case-sensitive, and nothing shows while you
  type. If you changed the password, use the new one.

- **"WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!" or "Host key verification failed"**

  Your computer remembers an older key for the gateway. Remove it (this works on Mac, Linux and Windows
  PowerShell):

  ```bash
  ssh-keygen -R 32.190.233.160
  ```

  Then log in again, check the fingerprint (`SHA256:Z8tsuVPiJ3dyI2vOQt2wV8hcBQxyaZOLpB2H4rnvvBE`) and
  type `yes`. If the fingerprint is different, do not type yes: tell the organisers.

- **"client_global_hostkeys_prove_confirm: server gave bad signature for ECDSA key 0"**

  Harmless: you are logged in and everything works. To stop the message, add these two lines to the file
  `~/.ssh/config` on your own computer (on Windows: `C:\Users\<you>\.ssh\config`; create the file if it
  does not exist):

  ```
  Host 32.190.233.160
    UpdateHostKeys no
  ```

- **Can I log in with my own SSH key instead of the password?**

  No. The gateway passes your login to your machine by username and password only, so adding a key to
  `~/.ssh/authorized_keys` does not change how you log in. Use your password.

- **"ssh: command not found" or "'ssh' is not recognized" (Windows)**

  Install the OpenSSH client: Settings, System, Optional features, Add a feature, **OpenSSH Client**.
  Then open a new PowerShell window.

### On your machine

The kit is in `~/catan-agent-hackathon`, with its Python environment ready. `~/START_HERE.txt` has the
short version.

Go to the kit and activate its environment (do this every time you log in):

```bash
cd ~/catan-agent-hackathon && source .venv/bin/activate
```

Check that everything works:

```bash
python -m catan_kit.selfcheck
```

Play your agent against another one, for example F against the random agent:

```bash
python -m catan_kit.evaluate --a examples/f_agent.py --b examples/random_agent.py --games 20
```

### Submit

Your machine has `~/upload.sh`, which sends a file to us. Upload a zip of your agent folder, or your
report as a PDF:

```bash
~/upload.sh my_team.zip
```

```bash
~/upload.sh report.pdf
```

It prints `uploaded to ...` when the upload worked. The team portal then shows your submission being
queued, scored and done, with its results. A zip can be at most 100 MB. Every upload counts towards your
team's limit, failed ones included, so test locally with `evaluate` first. The latest PDF you upload is
your report.

### Team portal

Your submissions, their scores and the live leaderboard are on the team portal:
**https://main.d2sgimrfzis7m5.amplifyapp.com/**

Sign in with your team name written the way it appears in your username after `team` (all lower case,
no spaces: `blanc`, not `teamblanc`), and your team leader's full registered phone number.

---

## 1. Install

**On your team machine the kit is already installed**, in `~/catan-agent-hackathon` with its
virtual environment and a passed self-check. Follow [Your team machine](#your-team-machine) below
to log in, and skip to section 3.

Each team machine has 8 CPU cores and 16 GB of memory. To work on your own computer as well:

You need **Python 3.11** and **git** (pip uses it to fetch the pinned engine). catanatron needs
at least Python 3.11, and we score on 3.11.

```bash
git clone https://github.com/pbhar-hyde/catan-agent-hackathon.git
cd catan-agent-hackathon
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu    # Linux and Windows only
pip install -r requirements.txt
```

The `torch` line installs the CPU build of PyTorch. Without it, pip on Linux fetches the much
larger CUDA build. On macOS, skip it.

If the install fails:

* **`python3.11: command not found`.** Many systems ship another version, for example Ubuntu 24.04
  ships 3.12. [uv](https://docs.astral.sh/uv/) installs 3.11 next to it: `uv python install 3.11`,
  then `uv venv --python 3.11 .venv` in place of the `python3.11 -m venv` line.
* **pip cannot find `git`.** Install git, then open a new terminal. On Windows: Git for Windows.
* **Windows PowerShell refuses to run `activate`.** Run
  `Set-ExecutionPolicy -Scope Process RemoteSigned` first, in the same window.
* **No PyTorch for your machine.** PyTorch 2.8 has no build for Intel Macs. Develop on Linux, in
  Google Colab or on Kaggle, or write an agent that does not import torch.
* **Google Colab and Kaggle** may run another Python version. The kit installs and passes its
  self-check on Python 3.12 too, but we score on 3.11, so check timings on the leaderboard.

Run every command from the root of this repository: that is where Python finds `catan_kit`.

The engine version is pinned. **Do not upgrade it.** The action space changed between catanatron
releases, so an agent written against another version produces actions our scorer rejects.

## 2. Check the install

```bash
python -m catan_kit.selfcheck
```

Fix any failure before you write agent code. A broken environment produces plausible numbers, not
error messages.

## 3. Play a game

```bash
python -m catan_kit.evaluate --a examples/f_agent.py --b examples/random_agent.py --games 200
```

```
A = examples/f_agent.py
B = examples/random_agent.py
A win rate          1.0000 +-0.0000   (200/200)
A mean CPU / game       97.7 ms   (limit 300 ms + 12.5 ms per decision, at most 1500 ms)
A worst CPU / game     262.5 ms
A games over budget        0      (random moves played after the limit: 0)
A errors                   0      illegal index or exception, of 12611 decisions
A timeouts/crashes    0 / 0     (a silent agent is killed after 3 s)
A load time             0.00 s    (limit 60 s)
wall clock              30.2 s
```

Measured on an idle laptop. Your numbers will differ, and a busy machine reports more CPU for
the same work. The leaderboard reports ours.

This is the script we score with. In the final evaluation we change only the board seeds, the dice key and
the opponents.

## 4. Write your agent

Make a folder for your agent, for example `my_team/`, copy `examples/random_agent.py` into it as
`agent.py`, and edit it.

```python
class Agent:
    def __init__(self):
        """Load weights or tables here. Called once, when your agent's process starts."""

    def decide(self, obs, playable_actions) -> int:
        """Return the INDEX of your chosen action in playable_actions."""
```

Then measure it against F:

```bash
python -m catan_kit.evaluate --a my_team/agent.py --b examples/f_agent.py --games 1000
```

## 5. Submit

Zip the folder, with `agent.py` at the top level of the zip and every file it loads beside it, up
to 100 MB, and upload it from your team machine:

```bash
~/upload.sh my_team.zip
```

Your report goes the same way, as a PDF: `~/upload.sh report.pdf`. Results appear on the team
portal (see [Team portal](#team-portal) above): sign in with your team name and your team leader's
phone number. PROBLEM_STATEMENT.md, section 4, has the rules.

Any operating system's zip works: we run your Python files directly, on Linux, with nothing to
build or start. Two things differ from Windows there: file names are case sensitive (`Weights.pt`
is not `weights.pt`), and paths must be built with `os.path.join`, never with `\`.

## 6. What is in here

| Path | What it is |
|---|---|
| `PROBLEM_STATEMENT.md` | the rules, scoring and schedule |
| `catan_kit/obs.py` | what your agent sees, and `sample_world` for search |
| `catan_kit/evaluate.py` | the scoring script |
| `catan_kit/match.py` | the harness it uses: one process per agent, the CPU clock, the masked copy |
| `catan_kit/host.py` | the agent's side of the harness, run in the agent's own process |
| `catan_kit/sandbox.py` | extra isolation used only on our machine, off on yours |
| `catan_kit/selfcheck.py` | install check |
| `catan_kit/_patch.py` | a determinism fix for the engine, applied at import |
| `examples/random_agent.py` | plays a random legal move and scores 0.000 against F |
| `examples/f_agent.py` | **F itself**, one of the two bots you have to beat |


## 7. Coding assistant (OpenCode)

OpenCode is a coding assistant that runs in your terminal, connected to our inference server. Your
team's API key is sent to your team leader: paste it where it says `<your key>` below.

> [!WARNING]
> Your machine's shell is **bash**, so everything goes in `~/.bashrc` (not `~/.zshrc`). Run each command
> on its own, in this order.

1. Install OpenCode:

   ```bash
   curl -fsSL https://opencode.ai/install | bash
   ```

2. Load the new settings, so the shell can find `opencode`:

   ```bash
   source ~/.bashrc
   ```

3. Add the `opencode-hyde` command, which keeps the Hyde setup separate from any other OpenCode settings:

   ```bash
   echo "alias opencode-hyde='XDG_CONFIG_HOME=~/.hyde-opencode/config XDG_DATA_HOME=~/.hyde-opencode/data opencode'" >> ~/.bashrc
   ```

4. Save your API key (replace `<your key>`, keep the quotes):

   ```bash
   echo 'export HYDE_INFERENCE_API_KEY="<your key>"' >> ~/.bashrc
   ```

5. Load the settings again:

   ```bash
   source ~/.bashrc
   ```

6. Log in to the inference server. It should end with "Logged into https://inference.hyde.tech":

   ```bash
   opencode-hyde auth login https://inference.hyde.tech
   ```

7. Start it from your kit folder:

   ```bash
   cd ~/catan-agent-hackathon && opencode-hyde
   ```

After the first time, you only need step 7.

- **"opencode-hyde: command not found"**

  Run `source ~/.bashrc`, or log out and in again. If it still fails, check that the alias was saved with
  `grep opencode-hyde ~/.bashrc`; if nothing is printed, repeat step 3.

- **"opencode: command not found"**

  The install did not finish, or the settings were not loaded. Run step 2, and if that does not help,
  repeat step 1.

> [!CAUTION]
> Keep your API key to your team. Do not share it, put it in your code, or include it in your submission
> zip.

## Need help?

Ask an organiser or a mentor in the room, and tell them your team name and the exact error message.

## License

GPL-3.0-or-later, because this kit builds on catanatron. See [LICENSE](LICENSE) and
[NOTICE.md](NOTICE.md).
