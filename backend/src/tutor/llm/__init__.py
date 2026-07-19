"""LLM-backed implementations of the diagnostician and lesson-writer ports.

Everything generated here passes a verification gate (restricted SymPy parse,
schema validation, answer-leak checks) before reaching a student, and every
adapter falls back to the deterministic template ports on failure — a bad
model response can never block a session.
"""
