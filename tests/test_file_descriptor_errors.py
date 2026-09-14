"""Resource exhaustion must not be reported as corrupt models or invalid input."""

import errno
import unittest

from modiff.server import WebServer, classify_hf_download_error


class FileDescriptorErrorTests(unittest.TestCase):
    def test_wrapped_native_file_open_failure_precedes_unicode_input_error(self):
        native = RuntimeError("Unable to open weights: Too many open files (24)")
        decoded = UnicodeDecodeError("utf-8", b"\xbb", 0, 1, "invalid start byte")
        decoded.__context__ = native
        wrapped = OSError("Unable to load weights")
        wrapped.__cause__ = decoded
        result = WebServer._classify_exception(object.__new__(WebServer), wrapped)
        self.assertEqual(result["error_code"], "file_descriptor_limit")
        self.assertEqual(result["category"], "runtime_error")
        self.assertIn("open-file", result["recovery_hint"])

    def test_operating_system_limits_are_classified_without_english_message(self):
        for code in (errno.EMFILE, errno.ENFILE):
            with self.subTest(errno=code):
                error = OSError(code, "localized error")
                result = WebServer._classify_exception(object.__new__(WebServer), error)
                self.assertEqual(result["error_code"], "file_descriptor_limit")
                status, download_code, message, retryable = classify_hf_download_error(error)
                self.assertEqual((status, download_code, retryable), (503, "huggingface_file_descriptor_limit", False))
                self.assertIn("open-file", message)

    def test_download_wrapper_and_causal_cycle_are_bounded(self):
        root = OSError(errno.EMFILE, "file limit")
        wrapped = RuntimeError("Hub cache access failed")
        wrapped.__cause__ = root
        root.__context__ = wrapped
        self.assertEqual(classify_hf_download_error(wrapped)[1], "huggingface_file_descriptor_limit")

    def test_unrelated_unicode_and_memory_errors_keep_their_classification(self):
        server = object.__new__(WebServer)
        error = UnicodeDecodeError("utf-8", b"\xbb", 0, 1, "invalid start byte")
        self.assertEqual(server._classify_exception(error)["error_code"], "invalid_node_input")
        self.assertEqual(server._classify_exception(RuntimeError("HIP out of memory"))["error_code"], "cuda_oom")
