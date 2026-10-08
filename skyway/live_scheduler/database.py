"""
Simple interface to a postgres database taking connection information from
'.desservices.ini'. Based on obztak/utils/database.py.

For more documentation on desservices, see here:
https://opensource.ncsa.illinois.edu/confluence/x/lwCsAw
"""

import logging
import os
import psycopg2
import platform

try:
    from configparser import RawConfigParser
except ImportError:
    from ConfigParser import RawConfigParser


class Database(object):
    """
    A simple interface to a postgres database taking connection information from
    '.desservices.ini'. Based on obztak/utils/database.py.
    """

    def __init__(self, dbname=None):
        """
        Initialize the database connection.

        Arguments
        ---------
        dbname: str [None]
            Name of the database section in '.desservices.ini' to use for connection.
            Default is to auto-detect based on hostname.
        """
        self.dbname = self.parse_dbname(dbname)
        self.conninfo = self.parse_config(section=self.dbname)
        self.connection = None
        self.cursor = None
        self.connect()

    def __str__(self):
        """String representation of the database connection."""
        return str(self.connection)

    def parse_dbname(self, dbname):
        """
        Determine the database name to use based on the hostname or provided argument.

        Arguments
        ---------
        dbname: str [None]
            Optional database name to use. If None, will auto-detect based on hostname.
        """

        # use provided dbname if given
        if dbname is not None:
            return dbname

        # auto-detect based on hostname
        hostname = platform.node()
        if hostname in (
            "observer2.ctio.noao.edu",
            "observer3.ctio.noao.edu",
            "observer6",
        ):
            return "db-ctio"
        return "db-fnal"

    def parse_config(self, filename=None, section="db-fnal"):
        """
        Parse the configuration file and return the connection information.

        Arguments
        ---------
        filename: str [None]
            Path to the configuration file. Default guesses possible default locations.
        section: str ["db-fnal"]
            Name of the section in the configuration file to use for connection info.
        """
        # determine path to configuration file
        if filename is None:
            if os.getenv("DES_SERVICES"):
                filename = os.getenv("DES_SERVICES")
            elif os.path.exists(".desservices.ini"):
                filename = os.path.expandvars("$PWD/.desservices.ini")
            else:
                filename = os.path.expandvars("$HOME/.desservices.ini")

        # parse the configuration file
        logging.debug(".desservices.ini: %s", filename)
        if not os.path.exists(filename):
            raise IOError("%s does not exist" % filename)
        parser = RawConfigParser()
        parser.read(filename)

        # extract connection information from the specified section
        return {
            "host": parser.get(section, "server"),
            "dbname": parser.get(section, "name"),
            "user": parser.get(section, "user"),
            "password": parser.get(section, "passwd"),
            "port": parser.get(section, "port"),
        }

    def connect(self):
        """Establish a connection to the database using parsed connection info."""
        logging.debug("Connecting to: %s", self.dbname)
        self.connection = psycopg2.connect(**self.conninfo)
        self.cursor = self.connection.cursor()
        logging.debug("Connection: %s", str(self.connection).split("'")[1])

    def disconnect(self):
        """Close the database connection and cursor if they are open."""
        if self.cursor is not None:
            self.cursor.close()
            self.cursor = None
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def execute(self, query):
        """Execute a SQL query and return the results, with error handling."""
        self.cursor.execute(query)
        try:
            return self.cursor.fetchall()
        except Exception as exc:
            self.reset()
            raise exc

    def reset(self):
        """Reset the database connection by disconnecting and reconnecting."""
        self.connection.reset()

    def query(self, query):
        """Convenience method to execute a query and return results."""
        return self.execute(query)

    def __del__(self):
        """Ensure the database connection is closed when the object is deleted."""
        self.disconnect()
