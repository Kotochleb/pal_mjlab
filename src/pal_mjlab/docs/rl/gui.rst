.. _launcher_gui:

Using the GUI
==============

The launcher is a Tkinter application (``menu.py``) that wraps common
mjlab commands — listing tasks, training, deploying policies, managing the
``tsp`` job queue, and syncing checkpoints from a remote machine — behind a
set of buttons, so you don't have to write or remember bash commands.

Every command the GUI runs is echoed into a terminal pane before it starts,
and its live output is streamed back as it runs, so the GUI never hides what
is actually being executed underneath.


Layout
------

The window is split into two areas:

- **Left panel** — terminal selector, action buttons, job queue controls and
  the list of currently running processes.

- **Right panel** — four independent terminal panes arranged in a 2x2 grid
  (**Terminal 1**–**Terminal 4**).

|

.. important::

   Every button runs its command in whichever terminal is currently
   selected via the **Select Terminal** radio buttons. Pick a terminal
   *before* clicking a button, otherwise output will land in the wrong
   pane.

Terminals
---------

Each of the four terminals behaves independently:

- Output is appended live as the underlying process produces it.
- Command lines you launched are shown in a distinct color, and the final
  status (``✓ done`` / ``✗ exited <code>``) is shown in another color once
  the process finishes.
- To keep memory bounded, each terminal keeps at most 500 lines and
  silently drops the oldest ones once that limit is exceeded.
- Terminals are read-only consoles — you can select/copy text
  (Ctrl+C / Ctrl+A) but not type into them.
- **Clear terminal** wipes the currently selected terminal.

|

Actions
-------

+-------------------------+------------------------------------------------+
| Button                  | What it does                                   |
+=========================+================================================+
| List tasks              | Runs ``uv run list-envs`` and parses the       |
|                         | resulting table into the list of tasks used by |
|                         | the Train and Deploy dialogs.                  |
+-------------------------+------------------------------------------------+
| Train Policy            | Opens the training dialog (see below).         |
+-------------------------+------------------------------------------------+
| Deploy                  | Opens the deploy dialog to run a policy from a |
|                         | saved checkpoint, or a scripted agent.         |
+-------------------------+------------------------------------------------+
| Sync Checkpoints        | ``rsync``'s the ``logs/`` folder from the      |
|                         | configured remote machine into the local       |
|                         | ``./logs`` directory.                          |
+-------------------------+------------------------------------------------+
| Setup Remote Folder     | Configures the SSH user / host / remote path   |
|                         | used by the Remote checkbox and Sync           |
|                         | Checkpoints.                                   |
+-------------------------+------------------------------------------------+
| Remote (checkbox)       | When checked, the *next* launched command runs |
|                         | over SSH on the configured remote machine      |
|                         | instead of locally.                            |
+-------------------------+------------------------------------------------+

You must click **List tasks** at least once before **Train Policy** or
**Deploy** will let you pick an environment.

|

Train Policy
~~~~~~~~~~~~

The training dialog collects:

- **Experiment Name** and **Custom Job Name** — used to label the run and
  build the ``tsp`` job label.
- **Select Environment** — one of the tasks discovered by List tasks.
- **Options** — a free-form list of ``key value`` overrides (one per line,
  without the leading ``--``), appended to the training command as
  ``--key value``. Click **Load Default** to pre-fill this box with the
  environment's default reward weights (``env.rewards.<name>.weight
  <value>``) so you can tweak them instead of typing them from scratch.
- **Description** — free text saved alongside the run for your own
  reference.

Confirming the dialog:

- Submits the job through ``tsp`` (so it queues rather than blocking the
  terminal), with ``--env.scene.num-envs 4096``, ``--agent.run-name
  <job>``, ``--agent.logger tensorboard`` and ``--agent.save-interval 500``
  always applied on top of your custom options.
- Appends a timestamped entry — environment, job name, full option string
  and description — to ``training_snapshots/<date>.txt``, so past runs stay
  documented even if you forget what you changed.

|

Deploy
~~~~~~

The deploy dialog lets you pick:

- **Task** — the environment to deploy on.
- **Checkpoint** — the latest checkpoint found for each ``experiment/run``
  folder under ``./logs`` (only the highest-step checkpoint per run is
  shown).
- **Agent Zero** / **Agent Random** — check either box to run a scripted
  baseline agent instead of a checkpoint, useful for sanity-checking the
  environment itself.

|

Job Queue (tsp)
----------------

``tsp`` (task-spooler) is used to queue training jobs so several runs can be
submitted without immediately competing for resources. The queue can be
inspected and managed from the left panel:

+------------------+---------------------------------------------------+
| Button           | Effect                                            |
+==================+===================================================+
| Check tsp        | Opens a small picker: **All jobs**, **Queued      |
|                  | only** or **Running only**, each running ``tsp``  |
|                  | (optionally piped through ``grep``).              |
+------------------+---------------------------------------------------+
| Check tsp -t     | Tails the output of the currently running/most    |
|                  | recent ``tsp`` job.                               |
+------------------+---------------------------------------------------+
| Remove tsp Job   | Prompts for a job ID and runs ``tsp -r <id>`` to  |
|                  | cancel it.                                        |
+------------------+---------------------------------------------------+

|

Running Processes
------------------

Every command launched from the GUI (local or remote) is tracked in the
**Running Processes** list, labeled by job/command name. Selecting an entry
and clicking **Terminate Selected** kills that process:

- **Local processes** — the full process tree (all child processes) is
  sent ``SIGTERM``, then ``SIGKILL`` if it hasn't exited after a short
  grace period.
- **Remote processes** — the same tree-kill logic is applied over SSH on
  the remote machine.

Closing the GUI (**X Quit**) or sending it a termination signal cleans up
every tracked process the same way, so nothing is left running unattended.

|

Remote Execution
-----------------

Remote support lets training or deployment run on another machine over SSH
instead of the local one:

1. Click **Setup Remote Folder** and provide the SSH user, host/IP, and the
   remote path to the project. The GUI verifies the path exists before
   saving it to ``remote_config.txt``.
2. Check the **Remote** checkbox before clicking an action button (Train
   Policy, Deploy, etc.).
3. You'll be prompted for the SSH password each time a remote command is
   launched — it is not stored on disk.

Once setup is done, the info is stored in a file, so it is not needed to
setup every time unless the remote changes.

.. important::

   The **Remote** checkbox only affects the *next* command you launch, not
   **Sync Checkpoints**, which always talks to the configured remote
   machine regardless of the checkbox state.

**Sync Checkpoints** additionally lets you filter which runs get pulled by
matching a substring against ``{run_date}_{run_name}`` folder names, so you
don't have to download every run in ``logs/`` on the remote machine.