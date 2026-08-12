import unittest
from types import SimpleNamespace

from modiff.server import WebServer, classify_hf_download_error


class FakeResponse:
    def __init__(self, status_code):
        self.status_code = status_code


class GatedRepoError(Exception):
    response = FakeResponse(403)


class RepositoryNotFoundError(Exception):
    response = FakeResponse(404)


class HuggingFaceDownloadErrorTests(unittest.TestCase):
    def test_gated_repository_has_actionable_non_retryable_error(self):
        status, code, message, retryable = classify_hf_download_error(GatedRepoError('restricted'))
        self.assertEqual(status, 403)
        self.assertEqual(code, 'huggingface_access_required')
        self.assertIn('read token', message)
        self.assertFalse(retryable)

    def test_network_failure_is_retryable(self):
        status, code, message, retryable = classify_hf_download_error(ConnectionError('connection reset'))
        self.assertEqual(status, 503)
        self.assertEqual(code, 'huggingface_network_error')
        self.assertIn('connection reset', message)
        self.assertTrue(retryable)

    def test_missing_repository_is_not_misreported_as_a_token_problem(self):
        error = RepositoryNotFoundError(
            'Repository Not Found. If this is a private or gated repo, make sure you are authenticated.'
        )
        status, code, message, retryable = classify_hf_download_error(error)
        self.assertEqual(status, 404)
        self.assertEqual(code, 'huggingface_repo_not_found')
        self.assertIn('not found', message)
        self.assertFalse(retryable)


class HuggingFaceDownloadInputTests(unittest.IsolatedAsyncioTestCase):
    async def test_download_endpoint_rejects_windows_backslash_repo_escapes_before_queueing(self):
        server = object.__new__(WebServer)
        request = SimpleNamespace(
            can_read_body=False,
            query={"repo_id": r"unit\..\..\outside"},
        )

        response = await WebServer.hf_download(server, request)

        self.assertEqual(response.status, 400)
        self.assertIn(b"invalid_huggingface_repo_id", response.body)

    async def test_download_endpoint_rejects_mutable_or_malformed_revisions(self):
        server = object.__new__(WebServer)
        for revision in ("main", "A" * 40, "a" * 39, " a" * 20):
            with self.subTest(revision=revision):
                request = SimpleNamespace(
                    can_read_body=False,
                    query={"repo_id": "unit/exact-model", "revision": revision},
                )

                response = await WebServer.hf_download(server, request)

                self.assertEqual(response.status, 400)
                self.assertIn(b"invalid_huggingface_revision", response.body)


if __name__ == '__main__':
    unittest.main()
