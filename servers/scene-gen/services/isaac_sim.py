"""TCP socket bridge to Isaac Sim, adapted from SAGE server/isaacsim/isaac_mcp/server.py."""

import socket
import json
import time
import logging
import os
from typing import Dict, Any, Optional

logger = logging.getLogger("scene-gen.isaac_sim")


class IsaacConnection:
    """Manages TCP socket connection to Isaac Sim extension."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        timeout: float = 300.0,
    ):
        self.host = host or os.environ.get("ISAAC_SIM_HOST", "localhost")
        self.port = port or int(os.environ.get("ISAAC_SIM_PORT", "8080"))
        self.timeout = timeout
        self._sock: Optional[socket.socket] = None

    def connect(self) -> bool:
        if self._sock:
            return True
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.connect((self.host, self.port))
            logger.info("Connected to Isaac Sim at %s:%s", self.host, self.port)
            return True
        except Exception as e:
            logger.error("Failed to connect to Isaac Sim: %s", e)
            self._sock = None
            return False

    def disconnect(self):
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            finally:
                self._sock = None

    def _receive_full_response(self, buffer_size: int = 16384) -> bytes:
        """Receive a complete JSON response, potentially in multiple chunks."""
        chunks: list[bytes] = []
        self._sock.settimeout(self.timeout)
        try:
            while True:
                try:
                    time.sleep(0.5)
                    chunk = self._sock.recv(buffer_size)
                    if not chunk:
                        if not chunks:
                            raise ConnectionError("Connection closed before receiving data")
                        break
                    chunks.append(chunk)
                    # Try to parse accumulated data as JSON
                    data = b"".join(chunks)
                    json.loads(data.decode("utf-8"))
                    return data
                except json.JSONDecodeError:
                    continue
                except socket.timeout:
                    break
        except socket.timeout:
            pass

        if chunks:
            data = b"".join(chunks)
            json.loads(data.decode("utf-8"))  # will raise if incomplete
            return data
        raise ConnectionError("No data received from Isaac Sim")

    def send_command(self, command_type: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send a command to Isaac Sim and return the result dict."""
        if not self._sock and not self.connect():
            raise ConnectionError("Not connected to Isaac Sim")

        command = {"type": command_type, "params": params or {}}
        try:
            self._sock.sendall(json.dumps(command).encode("utf-8"))
            response_data = self._receive_full_response()
            response = json.loads(response_data.decode("utf-8"))

            if response.get("status") == "error":
                raise RuntimeError(response.get("message", "Unknown Isaac Sim error"))
            return response.get("result", {})
        except (socket.timeout, ConnectionError, BrokenPipeError, ConnectionResetError) as e:
            self._sock = None
            raise ConnectionError(f"Isaac Sim communication error: {e}") from e
        except json.JSONDecodeError as e:
            self._sock = None
            raise RuntimeError(f"Invalid JSON from Isaac Sim: {e}") from e


def create_scene(conn: IsaacConnection, scene_save_dir: str, room_dict_path: str) -> Dict[str, Any]:
    """Create a single-room scene in Isaac Sim."""
    return conn.send_command("create_single_room_layout_scene_from_room", {
        "scene_save_dir": scene_save_dir,
        "room_dict_save_path": room_dict_path,
    })


def simulate_physics(conn: IsaacConnection) -> Dict[str, Any]:
    """Run physics simulation and return stability results."""
    return conn.send_command("simulate_the_scene")


def export_usd(conn: IsaacConnection, output_path: str) -> Dict[str, Any]:
    """Export the current scene as a USD file."""
    return conn.send_command("export_usd", {"output_path": output_path})
