"""The trace monitor exporter: each turn of an AI Suite agent recorded and sent to the trace monitor.

recorder.py keeps what a turn did while it runs; the record is built and sent after the turn. setup.py makes the parts
for a loaded entry when its options turn the export on; turn.py takes each turn through them. No Home Assistant
imports here: decider/ imports recorder.py, and with it this package.
"""
