# Skyway Live Scheduler

The Live Scheduler is a real-time, human-in-the-loop autonomous agent designed to schedule astronomical observations.
It interfaces directly with the telescope's control system, generating dynamic observation schedules based on live telemetry, machine learning inference, and observer approval.

## Architecture

The system is built as a modular, continuous state machine:

* **`TelescopeClient` ([`client.py`](./client.py)):** The hardware abstraction layer. Handles all synchronous communication with the telescope (polling exposure status and telemetry, and submitting observations to the queue).
* **`ModelRunner` ([`model_runner.py`](./model_runner.py)):** The inference engine. Keeps the ML model weights loaded in memory and generates "chunks" of proposed observations based on current state features.
* **`UserInterface` ([`interface.py`](./interface.py)):** The human-in-the-loop presentation layer. Currently implemented as a blocking Command Line Interface (CLI) that displays Pandas DataFrames and Matplotlib sky plots for user approval.
* **`ProgressManager` ([`progress_manager.py`](./progress_manager.py)):** Tracks observing progress, including checking start/stop conditions, logging completed observations, and resuming previous status if the system was restarted. Logs status to JSON Lines (`.jsonl`) files using a noon-to-noon session ID.
* **`SchedulerOrchestrator` ([`orchestrator.py`](./orchestrator.py)):** The central control loop that ties all the above components together.

## Setup & Configuration

Before running the scheduler, ensure your configuration parameters are set.
The default configuration file is located at [`skyway/configs/live_scheduler_default.yaml`](../configs/live_scheduler_default.yaml).

Key parameters to check:
* `model_path`: Path to the trained AI model.
* `chunk_size`: The number of future observations to generate and propose to the user at one time. Only the first of these is ever submitted for observing, while the rest are used to predict where future paths may lead.
* `output_directory`: Where the `history.jsonl` observing logs will be saved.

## Running the Scheduler

The system is designed to be run from a local Linux machine at the observatory, typically accessed via VNC. 

To start the observation loop, run the entry-point script:

```bash
python scripts/run_live_scheduler.py
```

## The Control Flow
1. **Proposal**: The system will fetch telemetry, generate a chunk of observations, and present the proposed chunk plan to the user.
2. **Approval**: The system will pause and ask for `Y`/`N` approval.
   * `Y`: The system will proceed to waiting for schedule submission.
   * `N`: The chunk is rejected and the system will regenerate a new proposal after masking the first selected action from the rejected proposal.
3. **Execution**: While an observation is exposing, the system continually polls the telescope control system. Once the exposure is finished, it logs the completion to disk and submits the first observation of the approved chunk. If significant telemetry changes are encountered while waiting, the system regenerates a chunk before submission.

## Safety & Interruptions
* **Big Red Stop Button (Emergency Stop)**: To safely and immediately halt the system at any point, press `Ctrl+C` in the terminal. This will trigger a `KeyboardInterrupt` that catches the loop, halts new submissions, and safely exits. This will only stop the AI scheduling agent from making inference and submitting new observations, so the telescope will continue completing its current observing queue.
* **Soft Interrupt (User Changes Mind While Waiting)**: ***Note: currently a placeholder, pending future Web UI development.*** This will allow a user to scrap the previously approved chunk plan while the agent is otherwise waiting for submission.
