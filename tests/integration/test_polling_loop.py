import unittest
from unittest.mock import patch, MagicMock
import json
import pandas as pd

from skyway.math import units
from skyway.live_scheduler.orchestrator import SchedulerOrchestrator
from skyway.live_scheduler.client import BlancoSCLTelescopeClient
from skyway.live_scheduler.model_runner import MockModelRunner


class SCLServerSimulation:
    """A helper class to simulate the stateful responses of the SCLN TCP/IP server."""

    def __init__(self):
        self.simulated_time = 0
        self.exposures = 0

    def advance_time(self, seconds):
        """Called by the mocked time.sleep to advance the server's internal clock."""
        self.simulated_time += seconds

    def send_command(self, cmd_str, timeout=1.5):
        cmd = json.loads(cmd_str)

        if cmd.get("command") == "EXPOSE":
            self.exposures += 1
            self.simulated_time = 0  # Reset the clock for the new exposure
            return json.dumps({"status": "EXPOSURE SUBMITTED"})

        elif cmd.get("command") == "TELEMETRY":
            # the first exposure is submitted immediately without waiting
            # for all subsequent exposures, wait 3 seconds of simulated time to be ready
            if self.exposures > 0:
                is_ready = self.simulated_time >= 3
            else:
                is_ready = True

            return json.dumps({
                "type": "TELEMETRY",
                "readyToExpose": is_ready,
                "telemetry": {"ra": 0.0, "dec": 0.0}
            })


class TestOrchestratorPollingLoop(unittest.TestCase):

    @patch('skyway.live_scheduler.client.SCL')
    @patch('time.sleep')
    def test_polling_and_rejection_loop(self, mock_sleep, MockSCL):
        """
        Integration test verifying the orchestrator correctly handles:
        1. A user rejecting a chunk (generating a new one).
        2. Submitting an observation.
        3. Polling the server while waiting for the exposure to finish.
        4. Exiting gracefully when the end condition is met.
        """

        # 1. SETUP THE NETWORK SIMULATION
        # link our custom state machine to the mock SCL's send_command method
        simulated_server = SCLServerSimulation()
        mock_scl_instance = MockSCL.return_value
        mock_scl_instance.is_connected.return_value = True
        mock_scl_instance.send_command.side_effect = simulated_server.send_command

        # link the mocked time.sleep directly to the server's clock
        mock_sleep.side_effect = simulated_server.advance_time

        # 2. SETUP THE MOCK COMPONENTS
        
        with patch('skyway.live_scheduler.client.ephemerides.equatorial_to_topographic',
                   return_value=(0.0, -30 * units.deg)), \
             patch('skyway.live_scheduler.client.DatabaseSeeing') as MockSeeing:
            MockSeeing.return_value.update.return_value = False
            MockSeeing.return_value.raw = pd.DataFrame()
            client = BlancoSCLTelescopeClient(
                propid="2019A-0305",
                server_ip="observer4.ctio.noao.edu",
                server_port=20000
            )
        model = MockModelRunner()

        # mock the CLI UI to Reject the first proposal, then Approve the next three
        mock_ui = MagicMock()
        mock_ui.get_user_decision.side_effect = [False, True, True, True]
        mock_ui.check_for_replan_signal.return_value = False

        # Mock the Progress Manager
        mock_progress = MagicMock()
        mock_progress.check_start_condition.return_value = True
        mock_progress.completed_fields = pd.DataFrame(
            columns=["field_id", "ra", "dec", "filter"]
        )

        # define the dynamic end condition: Stop after 3 observations are completed
        # define "completed" as submitting the observations and waited for all to finish
        def dynamic_end_condition():
            return simulated_server.exposures >= 3
        mock_progress.check_end_condition.side_effect = dynamic_end_condition

        # 3. ASSEMBLE AND RUN
        orchestrator = SchedulerOrchestrator(
            client=client,
            model=model,
            ui=mock_ui,
            progress=mock_progress,
            chunk_size=3,
            observing_poll_rate_sec=1
        )

        # This will run until our dynamic_end_condition returns True
        orchestrator.run()

        # 4. ASSERTIONS

        # A. Verify was asked for chunk decisions 4 times (1 rejection + 3 approvals)
        self.assertEqual(mock_ui.get_user_decision.call_count, 4)

        # B. Verify 3 exposures were actually sent to the telescope
        self.assertEqual(simulated_server.exposures, 3)

        # C. Verify the time.sleep was called correctly.
        # Exposure 1: No wait, submits immediately
        # Exposure 2: Waits 3 times before submitting
        # Exposure 3: Waits 3 times before submitting
        # Total sleeps = 6.
        self.assertEqual(mock_sleep.call_count, 6)

        # Mathematically prove it was instructed to sleep for exactly 1 second each time
        mock_sleep.assert_called_with(1)

        # D. Verify the progress manager logged the completions
        # (It logs the *previous* observation when the *next* one starts, plus we didn't
        # write the teardown logic for the final obs, so it should be called 2 times)
        self.assertEqual(mock_progress.record_completion.call_count, 2)

if __name__ == '__main__':
    unittest.main()
