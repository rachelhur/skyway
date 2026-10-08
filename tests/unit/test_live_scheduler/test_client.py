import unittest
from unittest.mock import patch
import json
import pandas as pd
from skyway.math import units

from skyway.live_scheduler.client import BlancoSCLTelescopeClient

class TestBlancoSCLTelescopeClient(unittest.TestCase):

    # @patch intercepts the SCL import inside your client file
    @patch('skyway.live_scheduler.client.DatabaseSeeing')
    @patch('skyway.live_scheduler.client.ephemerides.equatorial_to_topographic',
           return_value=(0.0, -30 * units.deg))
    @patch('skyway.live_scheduler.client.SCL')
    def setUp(self, MockSCL, mock_sun_altaz, MockSeeing):
        """This runs before every single test."""

        # empty seeing history instead of the DES database
        MockSeeing.return_value.update.return_value = False
        MockSeeing.return_value.raw = pd.DataFrame()

        # grab a reference to the dummy SCL instance
        self.mock_scl_instance = MockSCL.return_value
        
        # force the dummy socket to say "Yes, I am connected"
        self.mock_scl_instance.is_connected.return_value = True
        
        # instantiate client using the new parameters
        self.client = BlancoSCLTelescopeClient(
            propid="2019A-0305", 
            server_ip="observer4.ctio.noao.edu", 
            server_port=20000
        )

    def test_submit_observation_success(self):
        """Test that the client handles a successful exposure command."""
        
        fake_server_response = json.dumps({
            "target": "DECamAISched",
            "transaction_id": "0",
            "propid": "2019A-0305",
            "status": "EXPOSURE SUBMITTED",
            "error_code": 0,
            "message": "Command Succeeded",
            "data": {}
        })
        self.mock_scl_instance.send_command.return_value = fake_server_response

        # mock scheduler observation (using arbitrary values for ra/dec)
        ra = 169.188
        dec = 1.441
        obs_row = {
            "ra": ra * units.degree, 
            "dec": dec * units.degree, 
            "filter": "g", 
            "expTime": 90,
            "field_name": "test_target"
        }

        response = self.client.submit_observation(obs_row)

        self.assertEqual(response["status"], "EXPOSURE SUBMITTED")

        # verify the outgoing payload matches the daytime testing hardcodes
        called_args = self.mock_scl_instance.send_command.call_args[0][0]
        sent_payload = json.loads(called_args)
        
        self.assertEqual(sent_payload["command"], "EXPOSE")
        self.assertEqual(sent_payload["parameters"]["expType"], "dark")
        self.assertEqual(sent_payload["parameters"]["filter"], "block")
        self.assertEqual(sent_payload["parameters"]["expTime"], "90")
        self.assertEqual(sent_payload["parameters"]["object"], "test_target")
        self.assertEqual(sent_payload["parameters"]["propid"], "2019A-0305")
        self.assertEqual(sent_payload["parameters"]["comment"], "DO NOT USE")
        self.assertEqual(sent_payload["parameters"]["ra"], ra)
        self.assertEqual(sent_payload["parameters"]["dec"], dec)

    def test_submit_observation_failure(self):
        """Test that the client gracefully handles an error from the control system."""
        
        fake_server_response = json.dumps({
            "type": "RESPONSE",
            "source": "SISPI",
            "target": "DECamAISched",
            "transaction_id": "0",
            "propid": "Unknown",
            "status": "FAILED",
            "error_code": 0,
            "message": "FAILED: execute: insufficient privileges to execute commands",
            "data": "No data"
        })
        self.mock_scl_instance.send_command.return_value = fake_server_response

        obs_row = {"ra": 0.0, "dec": 0.0, "filter": "u", "expTime": 10}
        response = self.client.submit_observation(obs_row)

        self.assertEqual(response["status"], "FAILED")

    def test_check_exposure_status_ready(self):
        """Test the boolean parsing of the telemetry endpoint."""
        
        fake_server_response = json.dumps({
            "type": "TELEMETRY",
            "telemetry": {},
            "readyToExpose": True
        })
        self.mock_scl_instance.send_command.return_value = fake_server_response

        is_ready = self.client.check_exposure_status()
        self.assertTrue(is_ready)

    def test_check_exposure_status_not_ready(self):
        """Test the boolean parsing of the telemetry endpoint."""
        
        fake_server_response = json.dumps({
            "type": "TELEMETRY",
            "telemetry": {},
            "readyToExpose": False
        })
        self.mock_scl_instance.send_command.return_value = fake_server_response

        is_ready = self.client.check_exposure_status()
        self.assertFalse(is_ready)

    def test_telemetry_network_timeout(self):
        """Test how the client handles a complete network failure."""
        
        # Simulate the SCL client returning None (which happens on a socket timeout)
        self.mock_scl_instance.send_command.return_value = None

        is_ready = self.client.check_exposure_status()
        self.assertFalse(is_ready)

    def test_get_telemetry_success(self):
        """Test fetching and parsing RA and DEC from telemetry."""
        
        fake_server_response = json.dumps({
            "type": "TELEMETRY",
            "telemetry": {
                "ra": 150.5,
                "dec": -30.2
            }
        })
        self.mock_scl_instance.send_command.return_value = fake_server_response
        
        telemetry = self.client.get_telemetry()
        
        self.assertEqual(telemetry["pointing_ra"], 150.5)
        self.assertEqual(telemetry["pointing_dec"], -30.2)

if __name__ == '__main__':
    unittest.main()
