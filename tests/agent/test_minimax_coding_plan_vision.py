"""Tests para _MiniMaxCodingPlanVisionClient y _AsyncMiniMaxCodingPlanVisionClient.

Usa unittest.mock para simular httpx.Client — sin llamadas reales a la API.
"""

import sys
import os
import asyncio
import unittest.mock
from types import SimpleNamespace

# Agregar el paquete agent al path para poder importar auxiliary_client
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agent.auxiliary_client import (
    _MiniMaxCodingPlanVisionClient,
    _AsyncMiniMaxCodingPlanVisionClient,
)


class TestMiniMaxCodingPlanVisionClient(unittest.TestCase):
    """Tests para el cliente síncrono _MiniMaxCodingPlanVisionClient."""

    def setUp(self):
        """Configurar cliente con mocks."""
        self.api_key = "test-api-key-12345"
        self.api_host = "https://api.minimax.io"
        self.client = _MiniMaxCodingPlanVisionClient(
            api_key=self.api_key, api_host=self.api_host
        )

    # ------------------------------------------------------------------
    # Test 1: stream=True levanta NotImplementedError
    # ------------------------------------------------------------------
    def test_stream_true_raises_not_implemented_error(self):
        """Verifica que stream=True cause NotImplementedError."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe esto."},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/image.png"},
                    },
                ],
            }
        ]
        with self.assertRaises(NotImplementedError) as ctx:
            self.client.chat.completions.create(
                model="MiniMax-CodingPlan-VLM",
                messages=messages,
                stream=True,
            )
        self.assertIn(
            "Streaming is not supported", str(ctx.exception)
        )

    # ------------------------------------------------------------------
    # Test 2: stream=False (default) no levanta error
    # ------------------------------------------------------------------
    def test_stream_false_does_not_raise(self):
        """Verifica que stream=False (default) no lance excepciones."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe esto."},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/image.png"},
                    },
                ],
            }
        ]
        # Mock de httpx.Client para evitar llamada real
        mock_response = unittest.mock.MagicMock()
        mock_response.raise_for_status = unittest.mock.MagicMock()
        mock_response.json.return_value = {"content": "Descripción de prueba."}

        with unittest.mock.patch.object(
            self.client._httpx, "Client"
        ) as mock_client_class:
            mock_client_instance = unittest.mock.MagicMock()
            mock_client_instance.post.return_value = mock_response
            mock_client_class.return_value.__enter__ = unittest.mock.MagicMock(
                return_value=mock_client_instance
            )
            mock_client_class.return_value.__exit__ = unittest.mock.MagicMock(
                return_value=False
            )

            # No debe lanzar error
            result = self.client.chat.completions.create(
                model="MiniMax-CodingPlan-VLM",
                messages=messages,
                stream=False,  # default
            )
            self.assertIsNotNone(result)

    # ------------------------------------------------------------------
    # Test 3: Sin imagen levanta ValueError
    # ------------------------------------------------------------------
    def test_missing_image_raises_value_error(self):
        """Sin image_url debe levantar ValueError."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Solo texto, sin imagen."}
                ],
            }
        ]
        with self.assertRaises(ValueError) as ctx:
            self.client.chat.completions.create(
                model="MiniMax-CodingPlan-VLM",
                messages=messages,
            )
        self.assertIn(
            "image_url", str(ctx.exception)
        )

    # ------------------------------------------------------------------
    # Test 4: URL de imagen OpenAI procesada correctamente
    # ------------------------------------------------------------------
    def test_openai_image_url_parsed_correctly(self):
        """Mensaje con image_url remota se procesa sin conversión local."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "¿Qué hay en esta imagen?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://example.com/foto.jpg"
                        },
                    },
                ],
            }
        ]

        mock_response = unittest.mock.MagicMock()
        mock_response.raise_for_status = unittest.mock.MagicMock()
        mock_response.json.return_value = {
            "content": "Es una foto de un gato."
        }

        with unittest.mock.patch.object(
            self.client._httpx, "Client"
        ) as mock_client_class:
            mock_client_instance = unittest.mock.MagicMock()
            mock_client_instance.post.return_value = mock_response
            mock_client_class.return_value.__enter__ = unittest.mock.MagicMock(
                return_value=mock_client_instance
            )
            mock_client_class.return_value.__exit__ = unittest.mock.MagicMock(
                return_value=False
            )

            result = self.client.chat.completions.create(
                model="MiniMax-CodingPlan-VLM",
                messages=messages,
            )

            # Verificar que se hizo el POST con la URL correcta
            mock_client_instance.post.assert_called_once()
            call_args = mock_client_instance.post.call_args
            # post() recibe URL como primer positional arg
            call_positional = call_args[0]
            self.assertIn("vlm", call_positional[0])

            # Verificar que la respuesta tiene la estructura esperada
            self.assertTrue(hasattr(result, "choices"))
            self.assertEqual(
                result.choices[0].message.content, "Es una foto de un gato."
            )

    # ------------------------------------------------------------------
    # Test 5: Path de imagen local convertido a data URI
    # ------------------------------------------------------------------
    def test_local_image_path_converted_to_data_uri(self):
        """Path local se convierte a data URI usando mock de open()."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe esto."},
                    {
                        "type": "image_url",
                        "image_url": {"url": "/tmp/test_image.png"},
                    },
                ],
            }
        ]

        mock_response = unittest.mock.MagicMock()
        mock_response.raise_for_status = unittest.mock.MagicMock()
        mock_response.json.return_value = {"content": "Imagen local procesada."}

        # Simular contenido binario de una imagen PNG
        fake_image_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00"

        with unittest.mock.patch(
            "agent.auxiliary_client.open",
            unittest.mock.mock_open(read_data=fake_image_bytes),
        ), unittest.mock.patch(
            "agent.auxiliary_client.os.path.isfile", return_value=True
        ), unittest.mock.patch.object(
            self.client._httpx, "Client"
        ) as mock_client_class:
            mock_client_instance = unittest.mock.MagicMock()
            mock_client_instance.post.return_value = mock_response
            mock_client_class.return_value.__enter__ = unittest.mock.MagicMock(
                return_value=mock_client_instance
            )
            mock_client_class.return_value.__exit__ = unittest.mock.MagicMock(
                return_value=False
            )

            result = self.client.chat.completions.create(
                model="MiniMax-CodingPlan-VLM",
                messages=messages,
            )

            # Verificar que se hizo POST
            mock_client_instance.post.assert_called_once()
            call_args = mock_client_instance.post.call_args
            # content se pasa como keyword arg
            body_str = call_args[1]["content"]
            self.assertIn("data:image/png;base64,", body_str)

            # Verificar respuesta
            self.assertEqual(
                result.choices[0].message.content, "Imagen local procesada."
            )

    # ------------------------------------------------------------------
    # Test 6: Respuesta con interfaz OpenAI esperada
    # ------------------------------------------------------------------
    def test_response_has_expected_interface(self):
        """La respuesta debe tener .choices[0].message.content."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Cuenta sobre la imagen."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://example.com/img.jpg"
                        },
                    },
                ],
            }
        ]

        mock_response = unittest.mock.MagicMock()
        mock_response.raise_for_status = unittest.mock.MagicMock()
        mock_response.json.return_value = {
            "content": "Respuesta del modelo vision."
        }

        with unittest.mock.patch.object(
            self.client._httpx, "Client"
        ) as mock_client_class:
            mock_client_instance = unittest.mock.MagicMock()
            mock_client_instance.post.return_value = mock_response
            mock_client_class.return_value.__enter__ = unittest.mock.MagicMock(
                return_value=mock_client_instance
            )
            mock_client_class.return_value.__exit__ = unittest.mock.MagicMock(
                return_value=False
            )

            result = self.client.chat.completions.create(
                model="MiniMax-CodingPlan-VLM",
                messages=messages,
            )

            # Interfaz OpenAI completa
            self.assertTrue(hasattr(result, "choices"))
            self.assertTrue(len(result.choices) == 1)
            self.assertTrue(hasattr(result.choices[0], "message"))
            self.assertEqual(
                result.choices[0].message.role, "assistant"
            )
            self.assertEqual(
                result.choices[0].message.content, "Respuesta del modelo vision."
            )
            self.assertTrue(hasattr(result, "usage"))
            self.assertTrue(hasattr(result.usage, "prompt_tokens"))
            self.assertTrue(hasattr(result.usage, "completion_tokens"))
            self.assertTrue(hasattr(result.usage, "total_tokens"))


class TestAsyncMiniMaxCodingPlanVisionClient(unittest.TestCase):
    """Tests para el cliente asíncrono _AsyncMiniMaxCodingPlanVisionClient."""

    def setUp(self):
        """Configurar cliente síncrono y su wrapper asíncrono."""
        self.api_key = "test-api-key-async"
        self.api_host = "https://api.minimax.io"
        self.sync_client = _MiniMaxCodingPlanVisionClient(
            api_key=self.api_key, api_host=self.api_host
        )
        self.async_client = _AsyncMiniMaxCodingPlanVisionClient(
            sync_client=self.sync_client
        )

    # ------------------------------------------------------------------
    # Test 7: Async create delega al cliente síncrono via asyncio.to_thread
    # ------------------------------------------------------------------
    def test_async_client_delegates_to_sync(self):
        """Verifica que _async_create llama al sync via asyncio.to_thread."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe esto asíncronamente."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://example.com/async_img.png"
                        },
                    },
                ],
            }
        ]

        mock_response = unittest.mock.MagicMock()
        mock_response.raise_for_status = unittest.mock.MagicMock()
        mock_response.json.return_value = {
            "content": "Respuesta asíncrona."
        }

        with unittest.mock.patch.object(
            self.sync_client._httpx, "Client"
        ) as mock_client_class:
            mock_client_instance = unittest.mock.MagicMock()
            mock_client_instance.post.return_value = mock_response
            mock_client_class.return_value.__enter__ = unittest.mock.MagicMock(
                return_value=mock_client_instance
            )
            mock_client_class.return_value.__exit__ = unittest.mock.MagicMock(
                return_value=False
            )

            # Llamada asíncrona
            result = asyncio.run(
                self.async_client.chat.completions.create(
                    model="MiniMax-CodingPlan-VLM",
                    messages=messages,
                )
            )

            # Verificar resultado
            self.assertIsNotNone(result)
            self.assertEqual(
                result.choices[0].message.content, "Respuesta asíncrona."
            )

            # Verificar que se hizo exactamente una llamada POST (delegada)
            mock_client_instance.post.assert_called_once()


if __name__ == "__main__":
    unittest.main()
