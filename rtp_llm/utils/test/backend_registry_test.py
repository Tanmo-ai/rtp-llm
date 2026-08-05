import unittest

from rtp_llm.utils.backend_registry import (
    register_backend_hook,
    reset_backend_registrations,
    run_backend_registrations,
)


class BackendRegistryTest(unittest.TestCase):
    def setUp(self):
        reset_backend_registrations()

    def tearDown(self):
        reset_backend_registrations()

    def test_hook_runs_with_context_when_slot_drained(self):
        seen = []
        register_backend_hook("linear", lambda factory: seen.append(factory))

        self.assertEqual(seen, [], "hook must not run before the slot is drained")
        run_backend_registrations("linear", factory="LinearFactory")
        self.assertEqual(seen, ["LinearFactory"])

    def test_hooks_run_in_registration_order(self):
        seen = []
        register_backend_hook("moe", lambda: seen.append("first"))
        register_backend_hook("moe", lambda: seen.append("second"))

        run_backend_registrations("moe")
        self.assertEqual(seen, ["first", "second"])

    def test_draining_twice_does_not_rerun_hooks(self):
        calls = []
        register_backend_hook("linear", lambda: calls.append(1))

        run_backend_registrations("linear")
        run_backend_registrations("linear")
        self.assertEqual(calls, [1])

    def test_other_slots_are_untouched(self):
        calls = []
        register_backend_hook("attention", lambda: calls.append(1))

        run_backend_registrations("linear")
        self.assertEqual(calls, [])

    def test_registering_after_drain_raises(self):
        run_backend_registrations("linear")

        # A hook recorded after its slot was drained would never run, so
        # surface it instead of silently dropping the backend.
        with self.assertRaises(RuntimeError):
            register_backend_hook("linear", lambda: None)

    def test_hook_exception_propagates(self):
        def broken():
            raise ValueError("backend is broken")

        register_backend_hook("linear", broken)

        # Swallowing this would leave the factory silently selecting a
        # different implementation, i.e. wrong numerics instead of a crash.
        with self.assertRaises(ValueError):
            run_backend_registrations("linear")

    def test_draining_slot_without_hooks_is_noop(self):
        run_backend_registrations("nobody_registered_here")


if __name__ == "__main__":
    unittest.main()
